import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer

from attention_approximation.modeling_llama import LlamaForCausalLM as TeacherModel
from attention_approximation.modeling_llama_approximated import LlamaForCausalLM as StudentModel



def apply_repetition_penalty(logits: torch.Tensor, generated: torch.Tensor, penalty: float):
    """Down-weight logits for tokens that already appeared to discourage loops."""
    if penalty <= 1.0:
        return logits
    logits = logits.clone()
    unique_tokens = generated[0].tolist()
    for token_id in set(unique_tokens):
        logits[0, token_id] /= penalty
    return logits


def sample_next_token(logits: torch.Tensor, temperature: float, top_k: int):
    if temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / temperature
    if 0 < top_k < logits.size(-1):
        top_values, top_indices = torch.topk(logits, top_k)
        probs = torch.softmax(top_values, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        return top_indices.gather(-1, next_token)

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple generation script for distilled checkpoints.")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints_full/last.pt"))
    parser.add_argument("--model-config", dest="model_config", default=Path("data/MobileLLM/config.json"), type=Path)
    parser.add_argument("--seq-length", dest="seq_length", default=512, type=int)
    parser.add_argument("--factorization-rank", dest="factorization_rank", default=16, type=int)
    parser.add_argument("--layer-sharing", dest="layer_sharing", action="store_true")
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--prompt", default="In a shocking finding, scientists discovered a herd of unicorns living in a remote", type=str)
    parser.add_argument("--max-new-tokens", dest="max_new_tokens", default=100, type=int)
    parser.add_argument("--temperature", type=float, default=0.8, help="0 uses greedy decoding.")
    parser.add_argument("--top-k", dest="top_k", type=int, default=0, help="Top-k filtering for sampling.")
    parser.add_argument("--repetition-penalty", dest="repetition_penalty", type=float, default=1.2, help=">1.0 to penalize repeated tokens.")
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found at {args.checkpoint}")
    model_ckpt = torch.load(args.checkpoint, map_location="cpu")
    ckpt_config = model_ckpt.get("config", {})

    config = TeacherModel.config_class.from_json_file(args.model_config)
    config.seq_length = int(ckpt_config.get("seq_length", args.seq_length))
    config.factorization_rank = int(ckpt_config.get("factorization_rank", args.factorization_rank))
    config.layer_sharing = bool(ckpt_config.get("layer_sharing", args.layer_sharing))

    device = torch.device(args.device)

    model = StudentModel(config)
    missing, unexpected = model.load_state_dict(model_ckpt["model_state_dict"], strict=False)
    transferred = len(model.state_dict()) - len(missing)
    print(f"Transferred {transferred}/{len(model.state_dict())} items from pretrained weights")
    if missing:
        print(f"[warn] Missing keys ({len(missing)}): {missing[:8]}{' …' if len(missing) > 8 else ''}")
    if unexpected:
        print(f"[warn] Unexpected keys ({len(unexpected)}): {unexpected[:8]}{' …' if len(unexpected) > 8 else ''}")

    model.to(device)
    print("Loaded model")
    tokenizer = AutoTokenizer.from_pretrained("facebook/MobileLLM-350M-layer-share", use_fast=False, legacy=False)

    # Example input
    text = args.prompt
    with torch.inference_mode():
        generated = tokenizer(text, return_tensors="pt").input_ids.to(device)
        max_allowed = config.seq_length - generated.shape[1] - 1
        max_new_tokens = min(args.max_new_tokens or max_allowed, max_allowed)
        output_tokens = generated.clone()

        for _ in range(max_new_tokens):
            outputs = model(input_ids=output_tokens)
            logits = outputs.logits[:, -1, :]  # last token logits
            logits = apply_repetition_penalty(logits, output_tokens, args.repetition_penalty)

            next_token = sample_next_token(
                logits,
                temperature=args.temperature,
                top_k=args.top_k,
            )
            output_tokens = torch.cat([output_tokens, next_token], dim=-1)
            # Stop if EOS generated
            if next_token.item() == tokenizer.eos_token_id:
                break

    print(tokenizer.decode(output_tokens[0], skip_special_tokens=True))

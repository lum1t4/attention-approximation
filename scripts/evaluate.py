import argparse
from attention_approximation.evaluation import LlamaApproxEvalAdapter
from pathlib import Path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the approximated LLaMA student model with lm_eval.")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints_full/last.pt"))
    parser.add_argument("--config", type=Path, default=Path("data/MobileLLM/config.json"))
    parser.add_argument("--tasks", type=str, default="lambada_openai")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--precision", type=str, choices=['fp32', 'fp16', 'bf16'], default="fp32")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on the number of evaluation samples.")
    parser.add_argument("--bootstrap_iters", type=int, default=1000)
    parser.add_argument("--tokenizer", type=str, default="facebook/MobileLLM-350M-layer-share")
    parser.add_argument(
        "--max-gen-toks",
        type=int,
        default=None,
        help="Maximum number of tokens to generate when tasks request generation.",
    )

    args = parser.parse_args()

    adapter = LlamaApproxEvalAdapter(
        args.checkpoint,
        args.config,
        tokenizer=args.tokenizer,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed
    )

    adapter.evaluate(args.tasks, num_fewshot=args.num_fewshot, limit=args.limit, bootstrap_iters=args.bootstrap_iters)

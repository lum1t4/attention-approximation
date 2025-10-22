from contextlib import nullcontext
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F
from lm_eval import evaluator, utils
from lm_eval.api.model import TemplateLM
from transformers import AutoTokenizer

from attention_approximation.modeling_llama_approximated import LlamaForCausalLM
from attention_approximation.pytorch import init_seeds
# https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/model_guide.md


class LlamaApproxEvalAdapter(TemplateLM):
    """TemplateLM adapter for the approximated LLaMA student model."""

    def __init__(
        self,
        weights_path: str | Path,
        config_path: str | Path,
        tokenizer: str = "facebook/MobileLLM-350M-layer-share",
        batch_size: int = 1,
        max_length: int = 512,
        max_gen_toks: int | None = None,
        seed: int = 0,
        device: torch.device | str = "cpu",
        precision: Literal['fp32', 'fp16', 'bf16'] = "fp32"
    ):
        super().__init__()

        self.weights_path = Path(weights_path)
        self.config_path = Path(config_path)

        self.device: torch.device = torch.device(device) if isinstance(device, str) else device
        self.seed = seed
        self.batch_size_per_gpu = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=False, legacy=False)
        mapping = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }

        assert precision.lower() in mapping.keys(), f"Precision {precision} is not supported"
        assert weights_path.exists(), f"Checkpoint not found at {weights_path}"
        assert config_path.exists(), f"Config not found at {config_path}"
        self.dtype = mapping[precision.lower()]
        self._load_model()
        self.max_gen_toks = max_gen_toks or self.max_length
        init_seeds(seed)
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True

    def _load_model(self):
        ckpt = torch.load(self.weights_path, map_location="cpu")
        ckpt_config = ckpt.get("config", {})
        config = LlamaForCausalLM.config_class.from_json_file(self.config_path)
        self.max_length = int(ckpt_config.get("seq_length", 512))
        #  getattr(config, "seq_length", config.max_position_embeddings)
        config.factorization_rank = int(ckpt_config.get("factorization_rank",16))
        config.layer_sharing = getattr(config, "layer_sharing", False)
        model = LlamaForCausalLM(config)
        missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
        transferred = len(model.state_dict()) - len(missing)
        print(f"Transferred {transferred}/{len(model.state_dict())} items from pretrained weights")
        self.model: LlamaForCausalLM = model.eval().to(self.device, dtype=self.dtype)

    def autocast(self):
        if self.device.type == "cuda" and self.dtype in (torch.float16, torch.bfloat16):
            return torch.amp.autocast(device_type=self.device.type, dtype=self.dtype)
        return nullcontext()

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
        raise NotImplementedError("`loglikelihood_rolling` is not implemented for this model.")

    def greedy_generate(self, prompt: str) -> str:
        encoded = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = encoded.input_ids.to(self.device)
        if input_ids.size(1) == 0:
            if self.bos_token_id is None:
                raise ValueError("Tokenizer has no BOS token and prompt produced an empty sequence.")
            input_ids = torch.tensor([[self.bos_token_id]], device=self.device)

        generated = input_ids
        with torch.inference_mode():
            for _ in range(self.max_gen_toks):
                if generated.size(1) >= self.max_length:
                    break
                with self.autocast():
                    logits = self.model(input_ids=generated).logits
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=-1)
                if next_token.squeeze().item() == self.eot_token_id:
                    break
        new_tokens = generated[0, input_ids.size(1) :].tolist()
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def generate_until(self, requests):
        res = []
        reqs = [req.args for req in requests]

        def _collate(x):
            toks = self.tok_encode(x[0])
            return (len(toks), x[0])

        reord = utils.Reorderer(reqs, _collate)
        for context, gen_kwargs in reord.get_reordered():
            out_str = self.greedy_generate(context)
            for term in gen_kwargs["until"]:
                out_str = out_str.split(term)[0]
            res.append(out_str)
        return reord.get_original(res)

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def pad_token_id(self):
        if self.tokenizer.pad_token_id is not None:
            return self.tokenizer.pad_token_id
        elif self.tokenizer.eos_token_id is not None:
            return self.tokenizer.eos_token_id
        else:
            return 0

    @property
    def bos_token_id(self):
        return self.tokenizer.bos_token_id

    def tok_encode(self, string: str) -> list[int]:
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens)

    def _loglikelihood_tokens(self, requests, disable_tqdm: bool = False):
        if len(requests) == 0:
            return []

        results = [None for _ in range(len(requests))]
        ordered_indices = sorted(
            range(len(requests)),
            key=lambda i: len(requests[i][1]) + len(requests[i][2]),
            reverse=True,
        )

        B = self.batch_size_per_gpu
        for start in range(0, len(ordered_indices), B):
            end = min(start + B, len(ordered_indices))
            batch_indices = ordered_indices[start:end]

            sequences: list[list[int]] = []
            continuation_tokens: list[list[int]] = []
            context_lens: list[int] = []
            seq_lens: list[int] = []
            max_len = 0

            for ordered_idx in batch_indices:
                _, context, continuation = requests[ordered_idx]
                context_tokens = list(context)
                cont_tokens = list(continuation)

                if len(cont_tokens) == 0:
                    sequences.append(context_tokens)
                    continuation_tokens.append(cont_tokens)
                    context_lens.append(len(context_tokens))
                    seq_len = len(context_tokens)
                    seq_lens.append(seq_len)
                    max_len = max(max_len, seq_len)
                    continue

                if not context_tokens:
                    if self.bos_token_id is None:
                        raise ValueError("Received empty context and tokenizer has no BOS token.")
                    context_tokens = [self.bos_token_id]

                total_len = len(context_tokens) + len(cont_tokens)
                if total_len > self.max_length:
                    overflow = total_len - self.max_length
                    if overflow >= len(context_tokens):
                        raise ValueError(
                            f"Sequence length {total_len} exceeds model limit {self.max_length} "
                            "and continuation cannot be truncated."
                        )
                    context_tokens = context_tokens[overflow:]
                    total_len = len(context_tokens) + len(cont_tokens)

                sequences.append(context_tokens + cont_tokens)
                continuation_tokens.append(cont_tokens)
                context_lens.append(len(context_tokens))
                seq_lens.append(total_len)
                max_len = max(max_len, total_len)

            if max_len == 0:
                for dest, ordered_idx in enumerate(batch_indices):
                    results[ordered_idx] = (0.0, True)
                continue

            batch_size = len(batch_indices)
            batch_input = torch.full(
                (batch_size, max_len),
                fill_value=self.pad_token_id,
                dtype=torch.long,
                device=self.device,
            )
            attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=self.device)

            for row, seq in enumerate(sequences):
                if not seq:
                    continue
                length = len(seq)
                batch_input[row, :length] = torch.tensor(seq, dtype=torch.long, device=self.device)
                attention_mask[row, :length] = 1

            with torch.no_grad():
                with self.autocast():
                    logits = self.model(input_ids=batch_input, attention_mask=attention_mask).logits

            logprobs = F.log_softmax(logits, dim=-1)
            greedy_tokens = logprobs.argmax(dim=-1)

            for row, ordered_idx in enumerate(batch_indices):
                cont_tokens = continuation_tokens[row]
                cont_len = len(cont_tokens)
                ctx_len = context_lens[row]
                seq_len = seq_lens[row]

                if cont_len == 0:
                    results[ordered_idx] = (0.0, True)
                    continue

                start_pos = ctx_len - 1
                end_pos = seq_len - 1

                logprob_slice = logprobs[row, start_pos:end_pos, :]
                greedy_slice = greedy_tokens[row, start_pos:end_pos]

                target_tokens = torch.tensor(cont_tokens, dtype=torch.long, device=self.device)
                gathered = logprob_slice.gather(1, target_tokens.unsqueeze(-1)).squeeze(-1)

                total_logprob = float(gathered.sum().detach().cpu())
                greedy_match = bool(torch.equal(greedy_slice[:cont_len], target_tokens))
                results[ordered_idx] = (total_logprob, greedy_match)

        return results

    def evaluate(self, tasks: str | list[str], num_fewshot: int = 0, limit: int | None = None, bootstrap_iters: int = 1000) -> dict:
        with torch.inference_mode():
            if isinstance(tasks, str):
                tasks = [task.strip() for task in tasks.split(",") if task.strip()]
                print(f"Running tasks: {tasks}")
            return evaluator.simple_evaluate(
                model=self,
                tasks=tasks,
                num_fewshot=num_fewshot,
                limit=limit,
                bootstrap_iters=bootstrap_iters,
                numpy_random_seed=self.seed,
                torch_random_seed=self.seed,
                fewshot_random_seed=self.seed,
            )

from transformers import AutoModelForCausalLM, AutoTokenizer
from attention_approximation.modeling_llama import LlamaForCausalLM as TeacherModel
from attention_approximation.modeling_llama_approximated import LlamaForCausalLM as StudentModel

import torch
model_path = "checkpoints_full_model/whole_20.pt"
model_config = "data/MobileLLM/config.json"


seq_length = 512
factorization_rank = 16
layer_sharing = False
device = "cuda"

config = TeacherModel.config_class.from_json_file(model_config)
config.seq_length = seq_length
config.factorization_rank = factorization_rank
config.layer_sharing = layer_sharing



model = StudentModel(config)

# Load model checkpoint state dict
# from distill_whole_model import TrainingConfig
model_ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
print("Loaded model checkpoint from", model_path)
model.load_state_dict(model_ckpt["model_state_dict"], strict=False)
model.to(device)


# Example input
tokenizer = AutoTokenizer.from_pretrained("facebook/MobileLLM-350M-layer-share", use_fast=False, legacy=False)

class LAForGeneration:
    def __init__(self, model, tokenizer, device="cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
    
    def generate(self, prompt, max_new_tokens=50): # greedy decoding
        generated = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        for _ in range(max_new_tokens):
            with torch.inference_mode():
                outputs = model(input_ids=generated)
                logits = outputs.logits[:, -1, :]  # last token logits
                next_token = torch.argmax(logits, dim=-1).unsqueeze(-1)

            generated = torch.cat([generated, next_token], dim=-1)
            # stop if EOS generated
            if next_token.item() == tokenizer.eos_token_id:
                break

        return self.tokenizer.decode(generated[0], skip_special_tokens=True)
    
    def loglikelihood(self, requests: list) -> list[tuple[float, bool]]:
        results = []
        for context, continuation in requests:
            inputs = self.tokenizer(context + continuation, return_tensors="pt").to(self.device)
            with torch.no_grad():
                logits = self.model(**inputs).logits
            cont_ids = inputs["input_ids"][0, -len(self.tokenizer.encode(continuation)):]
            log_probs = torch.nn.functional.log_softmax(logits[0, -len(cont_ids)-1:-1], dim=-1)
            score = log_probs.gather(1, cont_ids.unsqueeze(-1)).sum().item()
            results.append((score, True))
        return results


    def loglikelihood_rolling(self, requests: list) -> list[float]:
        """
        requests: ["sequence1", "sequence2", ...]
        Returns: [score1, score2, ...] where each score is the total loglikelihood of the sequence.
        """
        results = []
        for string in requests:
            tokens = self.tokenizer(string, return_tensors="pt").input_ids[0].to(self.device)
            nll = 0.0

            for i in range(1, len(tokens)):
                inp = tokens[:i].unsqueeze(0)
                target = tokens[i]

                with torch.no_grad():
                    logits = self.model(input_ids=inp).logits
                log_probs = torch.nn.functional.log_softmax(logits[0, -1, :], dim=-1)
                nll += log_probs[target].item()

            results.append(nll)
        return results


    def generate_until(self, requests: list) -> list[str]:
        """
        requests: [(prompt, {"until": ["stop1", "stop2"]}), ...]
        Returns: [generated_text1, generated_text2, ...]
        """
        results = []
        for context, args in requests:
            until = args.get("until", None)
            generated = self.tokenizer(context, return_tensors="pt").input_ids.to(self.device)

            output_text = context
            for _ in range(model.config.seq_length):  # cap generation
                with torch.inference_mode():
                    logits = self.model(input_ids=generated).logits
                next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)
                generated = torch.cat([generated, next_token], dim=-1)
                decoded = self.tokenizer.decode(generated[0], skip_special_tokens=True)

                # check stopping condition
                stop_hit = False
                if until:
                    for s in until:
                        if decoded.endswith(s):
                            decoded = decoded[: -len(s)]
                            stop_hit = True
                            break
                if next_token.item() == self.tokenizer.eos_token_id:
                    stop_hit = True

                output_text = decoded
                if stop_hit:
                    break

            results.append(output_text)
        return results




# Usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
la_for_generation = LAForGeneration(model, tokenizer, device=device)
result = la_for_generation.generate("Hello, my dog is", max_new_tokens=50)
print(result)
requests = [("context1", "continuation1"), ("context2", "continuation2")]
results = la_for_generation.loglikelihood(requests)
print(results)
requests = ["input1", "input2"]
results = la_for_generation.loglikelihood_rolling(requests)
print(results)
requests = [("input1", {"until": "stop1"}), ("input2", {"until": "stop2"})]
results = la_for_generation.generate_until(requests)
print(results)


import code; code.interact(local=locals())
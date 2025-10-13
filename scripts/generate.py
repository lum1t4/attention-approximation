from transformers import AutoModelForCausalLM, AutoTokenizer
from attention_approximation.modeling_llama import LlamaForCausalLM as TeacherModel
from attention_approximation.modeling_llama_approximated import LlamaForCausalLM as StudentModel

import torch
model_path = "checkpoints_full_model/whole_1070.pt"
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


text = "In a shocking finding, scientists discovered a herd of unicorns living in a remote"

with torch.inference_mode():
    generated = tokenizer(text, return_tensors="pt").input_ids.to(device)
    max_new_tokens = config.seq_length - generated.shape[1] - 1
    for _ in range(max_new_tokens):
            with torch.inference_mode():
                outputs = model(input_ids=generated)
                logits = outputs.logits[:, -1, :]  # last token logits
                next_token = torch.argmax(logits, dim=-1).unsqueeze(-1)

            generated = torch.cat([generated, next_token], dim=-1)
            # stop if EOS generated
            if next_token.item() == tokenizer.eos_token_id:
                break

    print(tokenizer.decode(generated[0], skip_special_tokens=True))
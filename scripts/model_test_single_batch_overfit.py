
import torch
from attention_approximation.data import DistributedDataLoader
from attention_approximation.pytorch import init_seeds, device_memory_used, get_num_params

from attention_approximation.modeling_llama import LlamaForCausalLM as LLM0
from attention_approximation.modeling_llama_approximated import LlamaForCausalLM as LLM1
from attention_approximation.modeling_llama_approximated_1 import LlamaForCausalLM as LLM2

init_seeds(1337)

device = torch.device('cuda:0')
model_config_path = "data/MobileLLM/config.json"
data_path = "data/minipile"

config = LLM0.config_class.from_json_file(model_config_path)
config.factorization_rank = 8
config.layer_sharing = False
config.seq_length = 512

# model = LlamaForCausalLM(config)
model = LLM2(config)

loader = DistributedDataLoader(data_path, 4, 512, "train")
model.to(device)


print(f"Model {get_num_params(model)}")
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

x, y = loader.next_batch()
x = x.to(device, non_blocking=True)
y = y.to(device, non_blocking=True)

for i in range(1000):
    optimizer.zero_grad()
    outputs = model(x).logits
    loss = torch.nn.functional.cross_entropy(outputs.view(-1, outputs.size(-1)), y.view(-1))
    loss.backward()
    optimizer.step()
    if i <= 20 or i % 50:
        print(
            f"step {i}, loss = {loss.item()}, "
            f"mem = {device_memory_used(device)}"
        )

from attention_approximation.modeling_llama_approximated_1 import LlamaForCausalLM
import torch
from attention_approximation.data import DistributedDataLoader

device = torch.device('cuda:0')
model_config_path = "data/MobileLLM/config.json"
data_path = "data/minipile"

config = LlamaForCausalLM.config_class.from_json_file(model_config_path)
config.factorization_rank = 16
config.layer_sharing = False
config.seq_length = 512

model = LlamaForCausalLM(config)
loader = DistributedDataLoader(data_path, 4, 512, "train")
model.to(device)



model.eval()  # Set to eval mode to avoid batch norm issues
# Load batch
x, y = loader.next_batch()
x = x.to(device)

# Get embeddings and detach to make it a leaf variable
embeddings = model.model.embed_tokens(x).detach().clone()
embeddings.requires_grad = True

# Forward pass through the rest of the model
hidden_states = embeddings

_, seq_len, hidden_size = hidden_states.size()
device = hidden_states.device
grid_z, grid_y, grid_x = torch.meshgrid(
    torch.arange(config.num_attention_heads, dtype=torch.long, device=device),
    torch.arange(seq_len, dtype=torch.long, device=device),
    torch.arange(seq_len, dtype=torch.long, device=device),
    indexing="ij"
)
all_indices = torch.stack([grid_z, grid_y, grid_x], dim=-1).view(-1, 3)

causal_mask = model.model._update_causal_mask(
    attention_mask=None,
    input_tensor=hidden_states,
)

for layer in model.model.layers:
    hidden_states = layer(
        hidden_states,
        attention_mask=causal_mask,
        output_attentions=None,
        all_indices=all_indices
    )[0]

hidden_states = model.model.norm(hidden_states)
outs = torch.nn.functional.linear(hidden_states, model.model.embed_tokens.weight)

# Test: loss depends only on batch element 2
test_batch_idx = 2
loss = outs[test_batch_idx].sum()

# Backward pass
loss.backward()

# Verify that only embeddings[test_batch_idx] has non-zero gradients
print(f"Testing gradient dependencies for batch index {test_batch_idx}:")
for i in range(embeddings.shape[0]):
    if i == test_batch_idx:
        has_nonzero = (embeddings.grad[i] != 0).any().item()
        print(f"  embeddings.grad[{i}] has non-zero values: {has_nonzero} (expected: True)")
        assert has_nonzero, f"Expected non-zero gradients for batch {i}"
    else:
        all_zero = (embeddings.grad[i] == 0).all().item()
        print(f"  embeddings.grad[{i}] is all zeros: {all_zero} (expected: True)")
        assert all_zero, f"Expected zero gradients for batch {i}, but found non-zero values!"

print("\n✓ Test passed! No information leakage across batch dimension.")
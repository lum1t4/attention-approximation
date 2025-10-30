import torch
import time
from attention_approximation.modeling_llama import LlamaForCausalLM as LlamaBase
from attention_approximation.modeling_llama_approximated import LlamaForCausalLM as LlamaApprox
import safetensors.torch

# ---------- CONFIG ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seq_len = 512
batch_size = 2
n_warmup = 5
n_trials = 20

model_config_path = "data/MobileLLM/config.json"
model_weights_path = "data/MobileLLM/model.safetensors"
approx_checkpoints = "checkpoints/CF128/last.pt"

# ---------- LOAD BASE MODEL ----------
print("\n=== Loading BASE model ===")
config_base = LlamaBase.config_class.from_json_file(model_config_path)
msd_base = safetensors.torch.load_file(model_weights_path, device='cpu')
model_base = LlamaBase(config_base).to(device)
model_base.load_state_dict(msd_base, strict=True)
model_base.eval()

# ---------- LOAD APPROX MODEL ----------
print("\n=== Loading APPROX model ===")
config_approx = LlamaApprox.config_class.from_json_file(model_config_path)
config_approx.factorization_rank = 128
config_approx.layer_sharing = False
config_approx.seq_length = seq_len

msd_approx = torch.load(approx_checkpoints, map_location='cpu')
model_approx = LlamaApprox(config_approx).to(device)
model_approx.load_state_dict(msd_approx, strict=False)
# print("Compiling APPROX model with torch.compile()...")
# model_approx = torch.compile(model_approx)
# print("Done compiling.")
model_approx.eval()

# ---------- INPUT ----------
input_ids = torch.randint(0, config_base.vocab_size, (batch_size, seq_len), device=device)

def benchmark_model(model, name):
    print(f"\nBenchmarking {name}...")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    # Warmup
    for _ in range(n_warmup):
        _ = model(input_ids)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # Benchmark loop
    timings = []
    with torch.no_grad():
        for _ in range(n_trials):
            start_event.record()
            _ = model(input_ids)
            end_event.record()
            torch.cuda.synchronize()
            elapsed = start_event.elapsed_time(end_event)  # milliseconds
            timings.append(elapsed)

    avg_time = sum(timings) / len(timings)
    std_time = (sum((t - avg_time)**2 for t in timings) / len(timings)) ** 0.5
    peak_mem = torch.cuda.max_memory_allocated() / 1e6

    print(f"  Avg inference time: {avg_time:.2f} ± {std_time:.2f} ms")
    print(f"  Peak GPU memory:   {peak_mem:.2f} MB")

# ---------- RUN ----------
benchmark_model(model_base, "BASE model")
benchmark_model(model_approx, "APPROX model")

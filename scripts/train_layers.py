"""
This script is used to train replaced attention layers in a LLaMA model
using L2 loss to distill the attention outputs from the original attention layers.
"""

from transformers import LlamaConfig
from attention_approximation.modeling_llama import LlamaForCausalLM as TeacherModel
from attention_approximation.pytorch import intersect_dicts
from attention_approximation.utils import LOGGER
import safetensors
from copy import copy
import torch
from collections.abc import Callable
from torch import nn
from attention_approximation.modeling_llama_approximated import LlamaApproximatedAttention
from pathlib import Path
from attention_approximation.pytorch import WORLD_SIZE, LOCAL_RANK, RANK
from attention_approximation.data import DataLoaderLite
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import time


# Configs
model_config_path = "data/MobileLLM/config.json"
model_weights_path = "data/MobileLLM/model.safetensors"
data_path = Path("data/edu_fineweb10B")
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# Training hyperparameters
max_steps: int = 10000
batch_size: int = 2
seq_length: int = 128
gradient_accumulation_steps: int = 4
log_interval: int = 10
save_interval: int = 1000
val_interval: int = 250
learning_rate: float = 1e-3
min_learning_rate: float = 1e-5
weight_decay: float = 1e-5
warmup_steps: int = 100
grad_clip: float = 1.0
use_wandb: bool = False  # Set to True if you have wandb installed
checkpoint_dir = Path("checkpoints")
checkpoint_dir.mkdir(exist_ok=True)

# Instantiate teacher
config = LlamaConfig().from_json_file(model_config_path)

# Student config
student_config = copy(config)
student_config.factorization_rank = 16  # Low-rank factorization
student_config.layer_sharing = False
student_config.seq_length = seq_length

# Load teacher model
model = TeacherModel(config)
checkpoint = safetensors.torch.load_file(model_weights_path)
csd = intersect_dicts(checkpoint, model.state_dict())
model.load_state_dict(csd, strict=False)
LOGGER.info(f"Transferred {len(csd)}/{len(model.state_dict())} items from pretrained weights")
model.eval().to(device)


class AttentionDistillationWrapper(nn.Module):
    """Wrapper to distill teacher attention to student attention using L2 loss."""

    def __init__(self, student_att: Callable, teacher_att: nn.Module, config: LlamaConfig):
        super().__init__()
        self.student_att = student_att(config=config, all_indices=config.all_indices)
        self.teacher_att = teacher_att
        # Freeze teacher attention
        for param in self.teacher_att.parameters():
            param.requires_grad = False
        self.last_distillation_loss = 0.0  # Store loss for collection

    def forward(self, *args, **kwargs):
        # Get student outputs (trainable)
        student_outputs = self.student_att(*args, **kwargs)
        # Get teacher outputs (frozen)
        with torch.no_grad():
            teacher_outputs = self.teacher_att(*args, **kwargs)

        # Extract hidden states from outputs
        student_hidden_states = student_outputs[0] if isinstance(student_outputs, tuple) else student_outputs
        teacher_hidden_states = teacher_outputs[0] if isinstance(teacher_outputs, tuple) else teacher_outputs
        # Compute L2 distillation loss (normalized)
        # self.last_distillation_loss = torch.linalg.vector_norm(student_hidden_states - teacher_hidden_states, dim=-1).mean() * (teacher_hidden_states.size(-1) ** -0.5)
        self.last_distillation_loss = torch.nn.functional.mse_loss(student_hidden_states, teacher_hidden_states)
        # Return teacher outputs (to maintain compatibility with the rest of the model)
        return teacher_outputs


def patch_model(model: nn.Module):
    # Create all_indices for the approximated attention
    grid_y, grid_x = torch.meshgrid(torch.arange(seq_length, dtype=torch.long), torch.arange(config.hidden_size, dtype=torch.long), indexing="ij")
    all_indices = torch.stack([grid_y, grid_x], dim=-1).view(-1, 2)
    student_config.all_indices = all_indices.to(device)

    # Replace all attention layers with distillation wrapper
    model.requires_grad_(False)  # Freeze entire model first
    for i, layer in enumerate(model.model.layers):
        original_attn = layer.self_attn
        layer.self_attn = AttentionDistillationWrapper(
            student_att=LlamaApproximatedAttention,
            teacher_att=original_attn,
            config=student_config
        )
        LOGGER.info(f"Replaced attention in layer {i}")

    model = model.to(device)
    # Collect all student parameters
    student_params = []
    for layer in model.model.layers:
        student_params.extend(layer.self_attn.student_att.parameters())
    LOGGER.info(f"Total student parameters to train: {sum(p.numel() for p in student_params):,}")
    return model, student_params


model, new_params = patch_model(model)
# Setup optimizer and scheduler
optimizer = optim.AdamW(new_params, lr=learning_rate, weight_decay=weight_decay)
scheduler = CosineAnnealingLR(optimizer, T_max=max_steps, eta_min=min_learning_rate)

# Setup data loaders
train_loader = DataLoaderLite(
    path=data_path,
    batch_size=batch_size,
    seq_len=seq_length,
    process_rank=RANK if RANK >= 0 else 0,
    num_processes=WORLD_SIZE if WORLD_SIZE > 0 else 1,
    split='train'
)

val_loader = DataLoaderLite(
    path=data_path,
    batch_size=batch_size,
    seq_len=seq_length,
    process_rank=RANK if RANK >= 0 else 0,
    num_processes=WORLD_SIZE if WORLD_SIZE > 0 else 1,
    split='val'
)

# Initialize wandb if requested
if use_wandb and (LOCAL_RANK in {-1, 0}):
    try:
        import wandb
        wandb.init(
            project="llama-attention-distillation",
            config={
                "max_steps": max_steps,
                "batch_size": batch_size,
                "seq_length": seq_length,
                "learning_rate": learning_rate,
                "factorization_rank": student_config.factorization_rank,
                "gradient_accumulation_steps": gradient_accumulation_steps,
            }
        )
    except ImportError:
        LOGGER.warning("wandb not installed. Disabling wandb logging.")
        use_wandb = False


def train_step(model, batch, optimizer, scheduler, step):
    """Single training step."""
    x, y = batch
    x, y = x.to(device), y.to(device)

    # Forward pass through model
    outputs = model(x, labels=y)

    # Collect attention distillation losses from all layers
    total_att_loss = 0
    num_att_layers = 0

    for layer in model.model.layers:
        if hasattr(layer.self_attn, 'last_distillation_loss'):
            total_att_loss += layer.self_attn.last_distillation_loss
            num_att_layers += 1


    loss = total_att_loss / (gradient_accumulation_steps * num_att_layers)
    # Backward pass
    loss.backward()
    # Gradient clipping and optimization
    if (step + 1) % gradient_accumulation_steps == 0:
        torch.nn.utils.clip_grad_norm_(new_params, grad_clip)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    return loss.item() * gradient_accumulation_steps


def validate(model, val_loader, num_batches=10):
    """Validation loop."""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for _ in range(num_batches):
            x, y = val_loader.next_batch()
            x, y = x.to(device), y.to(device)

            outputs = model(x, labels=y)
            total_loss += outputs.loss.item()

    model.train()
    return total_loss / num_batches


def save_checkpoint(model, optimizer, scheduler, step, checkpoint_dir):
    """Save training checkpoint."""
    # Extract student state dict
    student_state = {}
    for i, layer in enumerate(model.model.layers):
        layer_state = layer.self_attn.student_att.state_dict()
        for k, v in layer_state.items():
            student_state[f"layer_{i}.self_attn.{k}"] = v

    checkpoint = {
        'step': step,
        'model_state_dict': student_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }

    checkpoint_path = checkpoint_dir / f"checkpoint_step_{step}.pt"
    torch.save(checkpoint, checkpoint_path)
    LOGGER.info(f"Saved checkpoint to {checkpoint_path}")


def main():
    LOGGER.info("Starting training...")
    model.train()
    start_time = time.time()
    running_loss = 0

    for step in range(max_steps):
        LOGGER.info(f"Step {step+1}/{max_steps}")
        # Get batch
        batch = train_loader.next_batch()

        # Training step
        loss, att_loss = train_step(model, batch, optimizer, scheduler, step)
        running_loss += loss
        # Logging
        if step % log_interval == 0:
            avg_loss = running_loss / log_interval
            elapsed = time.time() - start_time
            tokens_per_sec = (batch_size * seq_length * log_interval) / elapsed

            LOGGER.info(
                f"Step {step+1}/{max_steps} | "
                f"Loss: {avg_loss:.4f} | "
                f"Learning rate: {scheduler.get_last_lr()[0]:.2e} | "
                f"Tokens/s: {tokens_per_sec:.0f}"
            )
            running_loss = 0
            start_time = time.time()

        # Validation
        if (step + 1) % val_interval == 0:
            val_loss = validate(model, val_loader)
            LOGGER.info(f"Validation Loss: {val_loss:.4f}")

        # Save checkpoint
        if (step + 1) % save_interval == 0:
            save_checkpoint(model, optimizer, scheduler, step + 1, checkpoint_dir)

    # Final checkpoint
    save_checkpoint(model, optimizer, scheduler, max_steps, checkpoint_dir)
    LOGGER.info("Training completed!")



if __name__ == "__main__":
    main()
"""
This script is used to train replaced attention layers in a LLaMA model
using L2 loss to distill the attention outputs from the original attention layers.
"""

import time
import argparse
from copy import copy
from pathlib import Path
from dataclasses import dataclass
from collections.abc import Callable

import torch
import torch.optim as optim
import torch.distributed as dist
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: N817
from transformers import LlamaConfig
import safetensors
from attention_approximation.modeling_llama import LlamaForCausalLM as TeacherModel
from attention_approximation.modeling_llama_approximated import LlamaApproximatedAttention
from attention_approximation.pytorch import device_parse, intersect_dicts, WORLD_SIZE, LOCAL_RANK, RANK, rank_zero_only, de_parallel
from attention_approximation.utils import LOGGER
from attention_approximation.data import DistributedDataLoader
import numpy as np


DDP_ENABLED = WORLD_SIZE > 1

@dataclass
class TrainingConfig:
    """Stores all hyperparameters and configuration settings."""
    # Model and Data Paths
    model_config_path: str = "data/MobileLLM/config.json"
    model_weights_path: str = "data/MobileLLM/model.safetensors"
    data_path: str = "data/edu_fineweb10B"
    checkpoint_dir: str = "checkpoints"
    val: str = "test"
    device: str = "cuda"
    # Training Hyperparameters
    max_steps: int = 10000
    batch_size: int = 2
    seq_length: int = 128
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-3
    min_learning_rate: float = 1e-5
    weight_decay: float = 1e-5
    warmup_steps: int = 100
    grad_clip: float = 1.0

    # Logging and Saving
    log_interval: int = 10
    save_interval: int = 1000
    val_interval: int = 250
    val_batches: int = 10

    # Student Model Configuration
    factorization_rank: int = 16
    layer_sharing: bool = False


@rank_zero_only
def print0(s:str):
    LOGGER.info(s)

class AttentionDistillationWrapper(nn.Module):
    """
    Wrapper to distill teacher attention to student attention using L2 loss.
    This module replaces the standard attention layer during training.
    """
    def __init__(self, student_att: Callable, teacher_att: nn.Module, config: LlamaConfig):
        super().__init__()
        self.student_att = student_att(config=config, all_indices=config.all_indices)
        self.teacher_att = teacher_att
        # Freeze teacher attention parameters
        for param in self.teacher_att.parameters():
            param.requires_grad = False
        self.last_distillation_loss = 0.0  # Buffer to store loss for collection

    def forward(self, *args, **kwargs):
        # Get student outputs (trainable)
        student_outputs = self.student_att(*args, **kwargs)

        # Get teacher outputs (frozen, no gradients)
        with torch.no_grad():
            teacher_outputs = self.teacher_att(*args, **kwargs)
        # Extract hidden states from outputs
        student_hidden_states = student_outputs[0] if isinstance(student_outputs, tuple) else student_outputs
        teacher_hidden_states = teacher_outputs[0] if isinstance(teacher_outputs, tuple) else teacher_outputs
        # Compute MSE loss for distillation
        self.last_distillation_loss = torch.nn.functional.mse_loss(student_hidden_states, teacher_hidden_states)
        # Return teacher outputs to maintain compatibility with the rest of the model's forward pass
        return teacher_outputs


class TrainContext:
    config: TrainingConfig
    model: nn.Module | nn.parallel.DistributedDataParallel
    device: torch.device

    def __init__(self, config: TrainingConfig):
        self.config = config


def patch_model(state, model: nn.Module) -> tuple[nn.Module, list[nn.Parameter]]:
    """Replaces attention layers with the distillation wrapper."""
    config = model.config
    student_config = copy(config)
    student_config.factorization_rank = state.config.factorization_rank
    student_config.layer_sharing = state.config.layer_sharing
    student_config.seq_length = state.config.seq_length

    grid_y, grid_x = torch.meshgrid(
        torch.arange(state.config.seq_length, dtype=torch.long),
        torch.arange(config.hidden_size, dtype=torch.long),
        indexing="ij"
    )
    all_indices = torch.stack([grid_y, grid_x], dim=-1).view(-1, 2)
    student_config.all_indices = all_indices.to(state.device)

    model.requires_grad_(False)  # Freeze the entire model initially
    for i, layer in enumerate(model.model.layers):
        layer.self_attn = AttentionDistillationWrapper(
            student_att=LlamaApproximatedAttention,
            teacher_att=layer.self_attn,
            config=student_config
        )
        if LOCAL_RANK in {-1, 0}:
            LOGGER.info(f"Replaced attention in layer {i}")

    model = model.to(state.device)

    student_params = []
    for layer in model.model.layers:
        student_params.extend(layer.self_attn.student_att.parameters())
    print0(f"Total student parameters to train: {sum(p.numel() for p in student_params):,}")
    return model, student_params


def setup_model(state):
    """Loads, patches, and prepares the model for training."""
    # Load teacher config and model
    config = LlamaConfig().from_json_file(state.config.model_config_path)
    model = TeacherModel(config)

    checkpoint = safetensors.torch.load_file(state.config.model_weights_path)
    csd = intersect_dicts(checkpoint, model.state_dict())
    model.load_state_dict(csd, strict=False)
    print0(f"Transferred {len(csd)}/{len(model.state_dict())} items from pretrained weights")
    # Ensure all processes have the same initial weights if in DDP mode
    if DDP_ENABLED:
        dist.barrier()
    model.eval() # Teacher parts of the model should be in eval mode
    model, student_params = patch_model(state, model)
    if DDP_ENABLED:
        # DDP requires find_unused_parameters=True because the final model output
        # does not depend on the student parameters, only the distillation loss does.
        model = DDP(model, device_ids=[LOCAL_RANK], find_unused_parameters=True)
    return model, student_params


def setup_optimizer(state):
    """Initializes the AdamW optimizer."""
    return optim.AdamW(state.student_params, lr=state.config.learning_rate, weight_decay=state.config.weight_decay)


def setup_scheduler(state):
    """Initializes the cosine annealing learning rate scheduler."""
    max_steps = state.config.max_steps
    mlr = state.config.min_learning_rate
    return CosineAnnealingLR(state.optimizer, T_max=max_steps, eta_min=mlr)

def setup_dataloaders(state):
    """Initializes the data loaders for training and validation."""
    train_loader = DistributedDataLoader(
        path=Path(state.config.data_path),
        batch_size=state.config.batch_size,
        seq_len=state.config.seq_length,
        split='train'
    )
    val_loader = DistributedDataLoader(
        path=Path(state.config.data_path),
        batch_size=state.config.batch_size,
        seq_len=state.config.seq_length,
        split=state.config.val
    )
    return train_loader, val_loader


def training_step(state: TrainContext, batch, step: int):
    """Performs a single training step, including forward and backward passes."""
    x, y = batch
    x, y = x.to(state.device), y.to(state.device)

    # Forward pass is performed through the DDP-wrapped model
    state.model(x)

    # The model to access layers is state.model.module in DDP
    model = de_parallel(state.model)
    # Collect attention distillation losses from all layers
    total_att_loss = 0
    num_att_layers = 0

    for layer in model.model.layers:
        if hasattr(layer.self_attn, 'last_distillation_loss'):
            total_att_loss += layer.self_attn.last_distillation_loss
            num_att_layers += 1

    assert num_att_layers > 0, "No attention layers found for distillation."
    loss = total_att_loss / (state.config.gradient_accumulation_steps * num_att_layers)
    # Backward pass
    loss.backward()

    # Gradient clipping and optimization step
    if (step + 1) % state.config.gradient_accumulation_steps == 0:
        torch.nn.utils.clip_grad_norm_(state.student_params, state.config.grad_clip)
        state.optimizer.step()
        state.scheduler.step()
        state.optimizer.zero_grad()

    return loss.item() * state.config.gradient_accumulation_steps


@torch.inference_mode()
def validate(state: TrainContext) -> float:
    """Runs the validation loop and returns the average loss."""
    # Ensure model is in a consistent state for validation
    state.model.eval()
    total_loss = torch.tensor(0.0, device=state.device, requires_grad=False)
    for _ in range(state.config.val_batches):
        x, y = state.val_loader.next_batch()
        x, y = x.to(state.device), y.to(state.device)
        _ = state.model(x, labels=y)
        model = de_parallel(state.model)
        # Collect distillation losses for validation
        step_loss = torch.tensor(0.0, device=state.device, requires_grad=False)
        num_att_layers = 0
        for layer in model.model.layers:
            if hasattr(layer.self_attn, 'last_distillation_loss'):
                step_loss += layer.self_attn.last_distillation_loss
                num_att_layers += 1

        if num_att_layers > 0:
            total_loss += (step_loss / num_att_layers)

    # Aggregate losses across all DDP processes
    if DDP_ENABLED:
        total_loss_tensor = torch.tensor(total_loss, device=state.device)
        dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
        total_loss = total_loss_tensor.item()

    return total_loss / (state.config.val_batches * WORLD_SIZE)


@rank_zero_only
def save_checkpoint(state, step):
    """Saves a training checkpoint, only on the master process."""
    model = de_parallel(state.model) if DDP_ENABLED else state.model
    ckpt = Path(state.config.checkpoint_dir)
    ckpt.mkdir(exist_ok=True)
    ckpt = ckpt / f"checkpoint_step_{step}.pt"
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': state.optimizer.state_dict(),
        'scheduler_state_dict': state.scheduler.state_dict(),
    }, ckpt)
    LOGGER.info(f"Saved checkpoint to {ckpt}")


def train(state):
    """The main training loop."""
    if DDP_ENABLED:
        dist.init_process_group(backend='nccl')
        print0(f"DDP enabled. Rank: {RANK}, World Size: {WORLD_SIZE}")
        state.device = torch.device(f"cuda:{LOCAL_RANK}")
        torch.cuda.set_device(state.device)
    else:
        state.device = device_parse(state.config.device)
        print0(f"Using device: {state.device}")

    state.model, state.student_params = setup_model(state)
    # Initialize components
    state.optimizer = setup_optimizer(state)
    state.scheduler = setup_scheduler(state)
    state.train_loader, state.val_loader = setup_dataloaders(state)
    print0("Starting training...")
    state.optimizer.zero_grad()

    running_loss = 0
    start_time = time.time()

    for step in range(state.config.max_steps):
        batch = state.train_loader.next_batch()
        distillation_loss = training_step(state, batch, step)
        running_loss += distillation_loss
        if LOCAL_RANK in {-1, 0} and step % state.config.log_interval == 0:
            avg_loss = running_loss / state.config.log_interval
            elapsed = time.time() - start_time
            tokens_per_sec = (state.config.batch_size * state.config.seq_length * state.config.log_interval * WORLD_SIZE) / elapsed
            print0(f"Step {step+1}/{state.config.max_steps} | Distill Loss: {avg_loss:.4f} | LR: {state.scheduler.get_last_lr()[0]:.2e} | Tokens/s: {tokens_per_sec:.0f}")
            running_loss = 0
            start_time = time.time()

        if step % state.config.val_interval == 0:
            val_loss = validate(state)
            print0(f"Validation Distill Loss: {val_loss:.4f}")

        if step % state.config.save_interval == 0:
            save_checkpoint(state, step + 1)

    print0("Training completed!")
    if DDP_ENABLED:
        dist.destroy_process_group()



if __name__ == "__main__":
    """Parses command-line arguments, creates config and context, then starts training."""
    parser = argparse.ArgumentParser(description="Train replaced attention layers in a LLaMA model via distillation.")
    # Model and Data Paths
    parser.add_argument("--model_config_path", type=str, default="data/MobileLLM/config.json", help="Path to the LLaMA model config JSON file.")
    parser.add_argument("--model_weights_path", type=str, default="data/MobileLLM/model.safetensors", help="Path to the pretrained model weights (safetensors).")
    parser.add_argument("--data_path", type=str, default="data/edu_fineweb10B", help="Path to the training/validation dataset shards.")
    parser.add_argument("--val", type=str, default="test", help="Dataset split to use for validation.")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to train on (cuda, cpu, mps).")

    # Training Hyperparameters
    parser.add_argument("--max_steps", type=int, default=10000, help="Total number of training steps.")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size per process.")
    parser.add_argument("--seq_length", type=int, default=128, help="Sequence length for training.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Steps to accumulate gradients before optimizer step.")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Initial learning rate.")
    parser.add_argument("--min_learning_rate", type=float, default=1e-5, help="Minimum learning rate for cosine annealing scheduler.")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay for optimizer.")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Number of warmup steps for scheduler.")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Maximum gradient norm for clipping.")

    # Logging and Saving
    parser.add_argument("--log_interval", type=int, default=10, help="How often to log training metrics (in steps).")
    parser.add_argument("--save_interval", type=int, default=1000, help="How often to save checkpoints (in steps).")
    parser.add_argument("--val_interval", type=int, default=250, help="How often to run validation (in steps).")
    parser.add_argument("--val_batches", type=int, default=10, help="Number of validation batches to average over.")

    # Student Model Configuration
    parser.add_argument("--factorization_rank", type=int, default=16, help="Factorization rank for approximated attention.")
    parser.add_argument("--layer_sharing", action="store_true", help="Enable layer sharing in student attention.")

    args = parser.parse_args()
    train_config = TrainingConfig(**vars(args))
    state = TrainContext(train_config)
    train(state)


""""
python scripts/distill_individual_layers.py --model_config_path 'data/MobileLLM/config.json' --model_weights_path 'data/MobileLLM/model.safetensors' --data_path 'data/minipile' --checkpoint_dir 'checkpoints' --device 'cuda' --max_steps 10000 --batch_size 2 --seq_length 128 --gradient_accumulation_steps 4 --learning_rate 1e-3 --min_learning_rate 1e-5 --weight_decay 1e-5 --warmup_steps 100 --grad_clip 1.0 --log_interval 10 --save_interval 1000 --val_interval 250 --val_batches 10 --factorization_rank 16 --layer_sharing
"""

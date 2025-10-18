# scripts/distill_full_model.py
"""
This script trains a student LLaMA model with approximated attention layers
by distilling knowledge from a larger, frozen teacher model.
The distillation is performed using a combination of KL Divergence and Cross-Entropy loss.
"""

import argparse
import time
from copy import copy
from dataclasses import dataclass, fields
from pathlib import Path

import safetensors
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR

from attention_approximation.modeling_llama import LlamaForCausalLM as TeacherModel
from attention_approximation.modeling_llama_approximated import LlamaForCausalLM as StudentModel
from attention_approximation.pytorch import (
    LOCAL_RANK,
    RANK,
    WORLD_SIZE,
    de_parallel,
    device_parse,
    device_memory_clear,
    device_memory_used,
    device_synchronize,
    init_seeds,
    intersect_dicts,
    rank_zero_only,
)
from attention_approximation.utils import LOGGER, yaml_load
import gc
from attention_approximation.base import BaseTrainConfig, BaseTrainContext, setup_dataloaders, save_checkpoint, print0


@dataclass
class TrainingConfig(BaseTrainConfig):
    """Stores all hyperparameters and configuration settings."""
    alpha: float = 0.5
    temperature: float = 2.0
    from_checkpoint: str = "checkpoints/checkpoint_last.pt"
    

class TrainContext(BaseTrainContext):
    teacher: nn.Module



def setup_models(ctx: TrainContext):
    """Loads teacher and student models"""
    from types import SimpleNamespace

    # Teacher load
    config = TeacherModel.config_class.from_json_file(ctx.config.model_config_path)
    config.use_cache = False
    teacher = TeacherModel(config)
    ckpt = safetensors.torch.load_file(ctx.config.model_weights_path)
    csd = intersect_dicts(ckpt, teacher.state_dict())
    teacher.load_state_dict(csd, strict=False)
    print0(f"Transferred {len(csd)}/{len(teacher.state_dict())} items from pretrained weights")
    teacher = torch.compile(teacher.to(ctx.device)).eval()

    # Student model
    ckpt = torch.load(ctx.config.from_checkpoint, map_location="cpu")
    prev_config = SimpleNamespace(**ckpt['config'])

    ctx.config.factorization_rank = prev_config.factorization_rank
    ctx.config.layer_sharing = prev_config.layer_sharing
    ctx.config.seq_length = prev_config.seq_length

    print0("Updated context from checkpoint config (prev_config):")
    print0("\t - ctx.config.factorization_rank <- prev_config.factorization_rank")
    print0("\t - ctx.config.layer_sharing <- prev_config.layer_sharing")
    print0("\t - ctx.config.seq_length <- prev_config.seq_length")

    config.factorization_rank = ctx.config.factorization_rank
    config.layer_sharing = ctx.config.layer_sharing
    config.seq_length = ctx.config.seq_length
    
    ckpt = ckpt['model_state_dict']
    ckpt = {k.replace(".self_attn.student_att.", ".self_attn."): v for k, v in ckpt.items()}
    model = StudentModel(config)
    csd = intersect_dicts(ckpt, model.state_dict())
    # Ensure we can load all weights into student model
    model.load_state_dict(csd, strict=True)
    LOGGER.info(f"Transferred {len(csd)}/{len(model.state_dict())} items from pretrained weights")
    model = torch.compile(model.to(ctx.device))

    ctx.model, ctx.teacher = model, teacher
    if WORLD_SIZE > 1:
        ctx.model = DDP(ctx.model, device_ids=[LOCAL_RANK])
    return ctx


def calculate_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.5,
    temperature: float = 2.0,
):
    """Calculates combined loss"""
    student_logits_flat = student_logits.view(-1, student_logits.size(-1))
    teacher_logits_flat = teacher_logits.view(-1, teacher_logits.size(-1))
    # Cross-Entropy Loss
    loss_ce = F.cross_entropy(student_logits_flat, targets.view(-1))
    # KL Divergence Distillation Loss
    student_log_probs = F.log_softmax(student_logits_flat / temperature, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits_flat / temperature, dim=-1)
    kl_loss = F.kl_div(student_log_probs, teacher_log_probs, reduction="batchmean", log_target=True)
    loss_kl = kl_loss * (temperature**2)
    return alpha * loss_kl + (1.0 - alpha) * loss_ce, loss_ce, loss_kl


def training_step(ctx: TrainContext, step: int):
    """Performs a single training step, including forward and backward passes with AMP."""

    config = ctx.config
    ctx.model.train()
    ctx.optimizer.zero_grad()

    cum_loss = 0.0
    cum_ce_loss = 0.0
    cum_kl_loss = 0.0

    t0 = time.time()
    for k in range(ctx.config.grad_accum_steps):
        x, y = next(ctx.train_iterator)
        x = x.to(ctx.device, non_blocking=True)
        y = y.to(ctx.device, non_blocking=True)
        if WORLD_SIZE > 1:
            # Avoid unnecessary gradient syncs
            ctx.model.require_backward_grad_sync = (k == ctx.config.grad_accum_steps - 1)

        # Forward pass with AMP
        with torch.autocast(device_type=ctx.device.type, dtype=ctx.dtype, enabled=config.amp):
            with torch.inference_mode():
                teacher_logits = ctx.teacher(x).logits
            logits = ctx.model(x).logits
            loss, loss_ce, loss_kl = calculate_loss(
                logits, teacher_logits, y, config.alpha, config.temperature
            )
            loss = loss / ctx.config.grad_accum_steps
            cum_loss += loss.detach()
            cum_ce_loss += loss_ce.detach() / ctx.config.grad_accum_steps
            cum_kl_loss += loss_kl.detach() / ctx.config.grad_accum_steps
            # Backward pass
            ctx.scaler.scale(loss).backward()

    if WORLD_SIZE > 1:
        dist.all_reduce(cum_loss, op=dist.ReduceOp.AVG)
        dist.all_reduce(cum_ce_loss, op=dist.ReduceOp.AVG)
        dist.all_reduce(cum_kl_loss, op=dist.ReduceOp.AVG)

    cum_loss = cum_loss.item()

    # Optimization step
    ctx.scaler.unscale_(ctx.optimizer)
    norm = torch.nn.utils.clip_grad_norm_(ctx.model.parameters(), max_norm=ctx.config.grad_clip)
    ctx.scaler.step(ctx.optimizer)
    ctx.scaler.update()
    ctx.optimizer.zero_grad()
    ctx.scheduler.step()

    # Logging
    device_synchronize(ctx.device)

    dt = time.time() - t0
    lr = ctx.scheduler.get_last_lr()[0]
    tokens_per_second = ctx.config.grad_accum_steps * WORLD_SIZE * ctx.config.batch_size * ctx.config.seq_length / dt

    metrics = {
        "train/loss": cum_loss,
        "train/ce_loss": cum_ce_loss,
        "train/kl_loss": cum_kl_loss,
        "train/lr": lr,
        "train/norm": norm,
        "train/tokens_per_second": tokens_per_second,
        "train/mem": device_memory_used(ctx.device),
    }

    print0(
        f"step {step+1:4d}/{ctx.config.max_steps}"
        f" | loss {metrics['train/loss']:.6f}"
        f" | ce_loss {metrics['train/ce_loss']:.6f}"
        f" | kl_loss {metrics['train/kl_loss']:.6f}"
        f" | lr {metrics['train/lr']:.2e}"
        f" | norm {metrics['train/norm']:.4f}"
        f" | tokens_per_second {metrics['train/tokens_per_second']:.0f}"
        f" | dt {(dt)*1000:.2f} ms"
        f" | mem {metrics['train/mem']:2f} GB"
    )

    ctx.tracker.log(metrics, step=step)
    return ctx


@torch.inference_mode()
def validation_step(ctx: TrainContext, step: int):
    """Runs the validation loop with AMP."""
    if (step + 1) % ctx.config.val_interval != 0:
        return ctx
    ctx.model.eval()
    total_loss, total_ce, total_kl = 0.0, 0.0, 0.0
    alpha = ctx.config.alpha
    temperature = ctx.config.temperature

    for batch_idx, (x, y) in enumerate(ctx.valid_loader):
        x, y = x.to(ctx.device), y.to(ctx.device)
        # Forward pass with AMP context for validation
        with torch.autocast(device_type=ctx.device.type, dtype=ctx.dtype, enabled=ctx.config.amp):
            teacher_logits = ctx.teacher(x).logits
            logits = ctx.model(x).logits
            loss, loss_ce, loss_kl = calculate_loss(logits, teacher_logits, y, alpha, temperature)

        total_loss += loss.item()
        total_ce += loss_ce.item()
        total_kl += loss_kl.item()

    if WORLD_SIZE > 1:
        total_losses_tensor = torch.tensor([total_loss, total_ce, total_kl], device=ctx.device)
        dist.all_reduce(total_losses_tensor, op=dist.ReduceOp.SUM)
        total_loss, total_ce, total_kl = total_losses_tensor.tolist()

    avg_loss = total_loss / (len(ctx.valid_loader) * WORLD_SIZE)
    avg_ce = total_ce / (len(ctx.valid_loader) * WORLD_SIZE)
    avg_kl = total_kl / (len(ctx.valid_loader) * WORLD_SIZE)

    ctx.tracker.log({
        "val/loss": avg_loss,
        "val/ce_loss": avg_ce,
        "val/kl_loss": avg_kl,
    }, step=step)
    return ctx


def train(ctx: TrainContext):
    """The main training loop with integrated AMP and resuming."""
    ctx = setup_models(ctx)
    ctx.optimizer = optim.AdamW(ctx.model.parameters(), lr=ctx.config.lr, weight_decay=ctx.config.weight_decay)
    ctx.scheduler = CosineAnnealingLR(ctx.optimizer, T_max=ctx.config.max_steps, eta_min=ctx.config.min_lr)
    ctx = setup_dataloaders(ctx)
    ctx.train_iterator = iter(ctx.train_loader)
    print0("Starting training...")
    ctx.optimizer.zero_grad()
    for step in range(ctx.config.max_steps):
        ctx = training_step(ctx, step)
        ctx = validation_step(ctx, step)
        save_checkpoint(ctx, step)
        gc.collect()
        device_memory_clear(ctx.device)

    if WORLD_SIZE > 1:
        dist.destroy_process_group()
    print0("Training completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a student LLaMA model via full model distillation with advanced features.")
    # Model and Data Paths
    parser.add_argument("--config", type=str, help="Path to YAML config file.")

    parser.add_argument("--model_config_path", type=str, help="Path to the LLaMA model config JSON file.")
    parser.add_argument("--model_weights_path", type=str, help="Path to the pretrained model weights (safetensors).")
    parser.add_argument("--data_path", type=str, help="Path to the training/validation dataset shards.")
    parser.add_argument("--checkpoint_dir", type=str, help="Directory to save checkpoints.")
    parser.add_argument("--val", type=str, help="Dataset split to use for validation.")
    parser.add_argument("--device", type=str, help="Device to train on (cuda, cpu, mps).")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility.")
    
    # Training Hyperparameters
    parser.add_argument("--max_steps", type=int, help="Total number of training steps.")
    parser.add_argument("--batch_size", type=int, help="Batch size per process.")
    parser.add_argument("--seq_length", type=int, help="Sequence length for training.")
    parser.add_argument("--grad_accum_steps", type=int, help="Steps to accumulate gradients before optimizer step.")
    parser.add_argument("--lr", type=float, help="Initial learning rate.")
    parser.add_argument("--min_lr", type=float, help="Minimum learning rate for cosine annealing scheduler.")
    parser.add_argument("--weight_decay", type=float, help="Weight decay for optimizer.")
    parser.add_argument("--warmup_steps", type=int, help="Number of warmup steps for scheduler.")
    parser.add_argument("--grad_clip", type=float, help="Maximum gradient norm for clipping.")

    # Logging and Saving
    parser.add_argument("--save_interval", type=int, help="How often to save checkpoints (in steps).")
    parser.add_argument("--val_interval", type=int, help="How often to run validation (in steps).")

    # Student Model Configuration
    parser.add_argument("--factorization_rank", type=int, help="Factorization rank for approximated attention.")
    parser.add_argument("--layer_sharing", action="store_true", help="Enable layer sharing in student attention.")

    parser.add_argument("--tracker", type=str)
    parser.add_argument("--name", type=str)
    parser.add_argument("--from_checkpoint", type=str, help="Path to a checkpoint file to resume training from.")
    args = parser.parse_args()

    args = parser.parse_args()
    args = {k: v for k, v in vars(args).items() if v is not None}
    if args["config"]:
        base = yaml_load(args.pop("config"))['params']
        args = {**base, **args}  # Command-line args override config file

    train_config = TrainingConfig(**args)
    train(TrainContext(train_config))

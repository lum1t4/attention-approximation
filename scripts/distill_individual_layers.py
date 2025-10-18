import gc
import time
import argparse
from copy import copy, deepcopy
from pathlib import Path
from dataclasses import dataclass
from collections.abc import Callable

import torch
import torch.optim as optim
import torch.nn.init as init
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
# Removed DDP import
from transformers import LlamaConfig
import safetensors
from attention_approximation.modeling_llama import LlamaForCausalLM as TeacherModel
from attention_approximation.pytorch import (
    device_parse,
    intersect_dicts,
    WORLD_SIZE,
    LOCAL_RANK,
    RANK,
    rank_zero_only,
    de_parallel,
    LOCAL_RANK,
    device_memory_clear,
    device_synchronize,
    device_memory_used,
    init_seeds,
)
from attention_approximation.utils import LOGGER, yaml_load
from attention_approximation.data import distributed_dataloader, TokenDataset
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from attention_approximation.trackers import Tracker, AutoTracker
from attention_approximation.modeling_llama_approximated import LlamaAttention as LlamaApproximatedAttention
from attention_approximation.base import BaseTrainConfig, BaseTrainContext, setup_dataloaders, save_checkpoint, print0

TrainingConfig = BaseTrainConfig
TrainContext = BaseTrainContext


class AttentionDistillationWrapper(nn.Module):
    """
    Wrapper to distill teacher attention to student attention using L2 loss.
    This module replaces the standard attention layer during training.
    """
    def __init__(self, student_att: Callable, teacher_att: nn.Module, config: LlamaConfig):
        super().__init__()
        self.student_att = student_att(config=config)
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
    
    def train(self, mode: bool = True):
        self.student_att.train(mode)
        self.teacher_att.eval()  # Ensure teacher is always in eval mode
        return self


def setup_models(ctx: TrainContext):
    """Loads, patches, and prepares the model for training."""
    # model, train, student configs
    model_config = LlamaConfig().from_json_file(ctx.config.model_config_path)
    model_config.use_cache = False
    teacher_config = ctx.config
    student_config = copy(model_config)
    student_config.factorization_rank = ctx.config.factorization_rank
    student_config.layer_sharing = ctx.config.layer_sharing
    student_config.seq_length = ctx.config.seq_length

    # Load teacher model
    model = TeacherModel(model_config)
    checkpoint = safetensors.torch.load_file(teacher_config.model_weights_path)
    csd = intersect_dicts(checkpoint, model.state_dict())
    model.load_state_dict(csd, strict=False)
    print0(f"Transferred {len(csd)}/{len(model.state_dict())} items from pretrained weights")
    model.eval()

    student_params = []
    teacher_params = []
    for i, layer in enumerate(model.model.layers):
        layer.self_attn = AttentionDistillationWrapper(
            student_att=LlamaApproximatedAttention,
            teacher_att=layer.self_attn,
            config=student_config
        )
        student_params.extend(layer.self_attn.student_att.parameters())
        teacher_params.extend(layer.self_attn.teacher_att.parameters())
        print0(f"Replaced attention in layer {i}")

    model = model.to(ctx.device)
    model = torch.compile(model)
    
    if WORLD_SIZE > 1:
        model = DDP(model, device_ids=[LOCAL_RANK])
    ctx.model = model
    ctx.student_params = student_params
    student_num_params = sum(p.numel() for p in student_params)
    teacher_num_params = sum(p.numel() for p in teacher_params)
    print0(f"Total student parameters to train: {student_num_params:,} from teacher params {teacher_num_params:,}")
    ctx.tracker.log("model/trainable_params", student_num_params)
    ctx.tracker.log("model/factorization_rank", teacher_config.factorization_rank)
    return ctx


def training_step(ctx: TrainContext, step: int):
    """Performs a single training step, including forward and backward passes."""
    # Forward pass is performed through the possibly-wrapped model
    ctx.model.train()
    t0 = time.time()
    ctx.optimizer.zero_grad()
    device, amp = ctx.device, ctx.config.amp
    cum_loss = 0.0

    for k in range(ctx.config.grad_accum_steps):
        x, y = next(ctx.train_iterator)
        x, y = x.to(ctx.device, non_blocking=True), y.to(ctx.device, non_blocking=True)
        if WORLD_SIZE > 1:
            # Avoid unnecessary gradient syncs
            ctx.model.require_backward_grad_sync = (k == ctx.config.grad_accum_steps - 1)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            ctx.model(x)
            loss = torch.tensor(0.0, device=device, requires_grad=False)
            num_att_layers = 0
            for layer in de_parallel(ctx.model).model.layers:
                if hasattr(layer.self_attn, 'last_distillation_loss'):
                    num_att_layers += 1
                    loss += layer.self_attn.last_distillation_loss
            loss /= (ctx.config.grad_accum_steps * num_att_layers)
            cum_loss += loss.detach()
            # Backward pass
            ctx.scaler.scale(loss).backward()

    if WORLD_SIZE > 1:
        dist.all_reduce(cum_loss, op=dist.ReduceOp.AVG)
        cum_loss = cum_loss.item()

    # Optimization step
    ctx.scaler.unscale_(ctx.optimizer)
    norm = torch.nn.utils.clip_grad_norm_(ctx.model.parameters(), max_norm=ctx.config.grad_clip)
    ctx.scaler.step(ctx.optimizer)
    ctx.scaler.update()
    ctx.optimizer.zero_grad()
    ctx.scheduler.step()

    device_synchronize(device)


    dt = time.time() - t0
    lr = ctx.scheduler.get_last_lr()[0]
    tokens_per_second = ctx.config.grad_accum_steps * WORLD_SIZE * ctx.config.batch_size * ctx.config.seq_length / dt

    metrics = {
        "train/loss": cum_loss,
        "train/lr": lr,
        "train/norm": norm,
        "train/tokens_per_second": tokens_per_second,
        "train/mem": device_memory_used(ctx.device),
    }

    print0(
        f"step {step+1:4d}/{ctx.config.max_steps}"
        f" | loss {metrics['train/loss']:.6f}"
        f" | lr {metrics['train/lr']:.2e}"
        f" | norm {metrics['train/norm']:.4f}"
        f" | tokens_per_second {metrics['train/tokens_per_second']:.0f}"
        f" | dt {(dt)*1000:.2f} ms"
        f" | mem {metrics['train/mem']:2f} GB"
    )

    ctx.tracker.log(metrics, step=step)
    return ctx


@torch.inference_mode()
def validation_step(ctx: TrainContext, step: int) -> float:
    """Runs the validation loop and returns the average loss."""
    # Ensure model is in a consistent state for validation
    ctx.model.eval()
    if (step + 1) % ctx.config.val_interval != 0:
        return ctx

    loss = torch.zeros(1, device=ctx.device)
    for batch_idx, (x, y) in enumerate(ctx.valid_loader):
        x, y = x.to(ctx.device), y.to(ctx.device)
        _ = ctx.model(x)
        # Collect distillation losses for validation
        step_loss = torch.zeros(1, device=ctx.device)
        num_att_layers = 0
        for layer in de_parallel(ctx.model).model.layers:
            if hasattr(layer.self_attn, 'last_distillation_loss'):
                step_loss += layer.self_attn.last_distillation_loss
                num_att_layers += 1

        if num_att_layers > 0:
            loss += (step_loss / num_att_layers)

    metrics = {"valid/loss": (loss / len(ctx.valid_loader)).item()}
    print0(f"step {step+1:4d}/{ctx.config.max_steps}  | valid/loss {metrics['valid/loss']:.6f}")
    ctx.tracker.log(metrics, step=step)
    return ctx


def train(ctx: TrainContext):
    """The main training loop."""
    ctx = setup_models(ctx)
    print0("Starting training...")
    ctx.optimizer = optim.AdamW(ctx.student_params, lr=ctx.config.lr, weight_decay=ctx.config.weight_decay)
    ctx.scheduler = CosineAnnealingLR(ctx.optimizer, T_max=ctx.config.max_steps, eta_min=ctx.config.min_lr)
    ctx = setup_dataloaders(ctx)
    ctx.train_iterator = iter(ctx.train_loader)

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
    """Parses command-line arguments, creates config and context, then starts training."""
    parser = argparse.ArgumentParser(description="Train replaced attention layers in a LLaMA model via distillation.")
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
    # parser.add_argument("--layer_sharing", action="store_true", help="Enable layer sharing in student attention.")

    parser.add_argument("--tracker", type=str)
    parser.add_argument("--name", type=str)
                        
    args = parser.parse_args()
    args = {k: v for k, v in vars(args).items() if v is not None}
    if args["config"]:
        base = yaml_load(args.pop("config"))['params']
        args = {**base, **args}  # Command-line args override config file

    train_config = TrainingConfig(**args)
    train(TrainContext(train_config))

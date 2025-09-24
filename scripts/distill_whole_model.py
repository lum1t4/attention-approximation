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

from attention_approximation.data import DistributedDataLoader
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
    init_seeds,
    intersect_dicts,
    rank_zero_only,
)
from attention_approximation.utils import LOGGER, yaml_load

DDP_ENABLED = WORLD_SIZE > 1


@dataclass
class TrainingConfig:
    """Stores all hyperparameters and configuration settings."""
    from_checkpoint: str
    # Model and Data Paths
    model_config_path: str = "data/MobileLLM/config.json"
    model_weights_path: str = "data/MobileLLM/model.safetensors"
    data_path: str = "data/edu_fineweb10B"
    checkpoint_dir: str = "checkpoints_full_model"
    val: str = "test"
    device: str = "cuda"  # or "cpu", user can override
    dtype: str = "bfloat16"  # float16 or bfloat16
    seed: int = 0

    # Training Hyperparameters
    max_steps: int = 20000
    batch_size: int = 2
    seq_length: int = 512  # Increased default to show value of chunking
    grad_accum_steps: int = 4
    lr: float = 1e-4
    min_lr: float = 1e-5
    weight_decay: float = 1e-5
    grad_clip: float = 1.0

    # Distillation Hyperparameters
    alpha: float = 0.5
    temperature: float = 2.0
    loss_chunk_size: int = 1024  # Chunk size for memory-efficient loss. 0 to disable.

    # Performance & Mixed Precision
    use_amp: bool = False
    amp_dtype: str = "bfloat16"  # float16 or bfloat16

    # Logging and Saving
    log_interval: int = 10
    save_interval: int = 1000
    val_interval: int = 250
    val_batches: int = 10

    # Student Model Configuration
    factorization_rank: int = 16
    layer_sharing: bool = False

# Enable TF32 for faster matmuls on Ampere+ GPUs
torch.set_float32_matmul_precision('high')


@rank_zero_only
def print0(s: str):
    LOGGER.info(s)


class TrainContext:
    config: TrainingConfig
    teacher_model: nn.Module
    student_model: nn.Module | nn.parallel.DistributedDataParallel
    optimizer: optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    train_loader: DistributedDataLoader
    val_loader: DistributedDataLoader
    scaler: torch.cuda.amp.GradScaler | None
    device: torch.device
    dtype: torch.dtype

    def __init__(self, config: TrainingConfig):
        self.config = config


def setup_models(state: TrainContext):
    """Loads teacher and student models"""
    config = TeacherModel.config_class.from_json_file(state.config.model_config_path)
    teacher = TeacherModel(config)
    ckpt = safetensors.torch.load_file(state.config.model_weights_path)
    csd = intersect_dicts(ckpt, teacher.state_dict())
    teacher.load_state_dict(csd, strict=False)
    teacher = torch.compile(teacher.to(state.device)).eval()
    print0(f"Transferred {len(csd)}/{len(teacher.state_dict())} items from pretrained weights")

    config.factorization_rank = state.config.factorization_rank
    config.layer_sharing = state.config.layer_sharing
    config.seq_length = state.config.seq_length
    model = StudentModel(config)

    ckpt = torch.load(state.config.from_checkpoint, map_location="cpu")['model_state_dict']
    # Only rename the student_att part, keep model. prefix intact
    ckpt = {k.replace(".self_attn.student_att.", ".self_attn."): v for k, v in ckpt.items()}
    csd = intersect_dicts(ckpt, model.state_dict())
    csd = intersect_dicts(ckpt, model.state_dict())
    model.load_state_dict(csd, strict=False)
    model = torch.compile(model.to(state.device))
    LOGGER.info(f"Transferred {len(csd)}/{len(model.state_dict())} items from pretrained weights")

    if DDP_ENABLED: dist.barrier()
    return teacher, model


def setup_dataloaders(state: TrainContext):
    state.train_loader = DistributedDataLoader(path=Path(state.config.data_path), batch_size=state.config.batch_size, seq_len=state.config.seq_length, split="train")
    state.val_loader = DistributedDataLoader(path=Path(state.config.data_path), batch_size=state.config.batch_size, seq_len=state.config.seq_length, split=state.config.val)
    return state

def calculate_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.5,
    temperature: float = 2.0,
):
    """Calculates combined loss"""
    # Cross-Entropy Loss
    loss_ce = F.cross_entropy(student_logits.view(-1, student_logits.size(-1)), targets.view(-1))

    # KL Divergence Distillation Loss
    student_logits_flat = student_logits.view(-1, student_logits.size(-1))
    teacher_logits_flat = teacher_logits.view(-1, teacher_logits.size(-1))

    student_log_probs = F.log_softmax(student_logits_flat / temperature, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits_flat / temperature, dim=-1)
    kl_loss = F.kl_div(student_log_probs, teacher_log_probs, reduction="batchmean", log_target=True)
    loss_kl = kl_loss * (temperature**2)
    return alpha * loss_kl + (1.0 - alpha) * loss_ce, loss_ce, loss_kl


def training_step(state: TrainContext, step: int):
    """Performs a single training step, including forward and backward passes with AMP."""
    state.model.train()

    t0 = time.time()
    state.optimizer.zero_grad()
    alpha = state.config.alpha
    temperature = state.config.temperature

    device, use_amp, dtype = state.device, state.config.use_amp, state.dtype
    cum_loss = 0.0
    cum_ce_loss = 0.0
    cum_kl_loss = 0.0

    for k in range(state.config.grad_accum_steps):
        x, y = state.train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        if DDP_ENABLED:
            # Avoid unnecessary gradient syncs
            state.model.require_backward_grad_sync = (k == state.config.grad_accum_steps - 1)

        # Forward pass with AMP
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
            with torch.inference_mode():
                teacher_logits = state.teacher(x).logits
            logits = state.model(x).logits
            loss, loss_ce, loss_kl = calculate_loss(logits, teacher_logits, y, alpha, temperature)
            loss = loss / state.config.grad_accum_steps
            cum_loss += loss.detach()
            cum_ce_loss += loss_ce.detach() / state.config.grad_accum_steps
            cum_kl_loss += loss_kl.detach() / state.config.grad_accum_steps
            # Backward pass
            state.scaler.scale(loss).backward()

    if DDP_ENABLED:
        dist.all_reduce(cum_loss, op=dist.ReduceOp.AVG)
        dist.all_reduce(cum_ce_loss, op=dist.ReduceOp.AVG)
        dist.all_reduce(cum_kl_loss, op=dist.ReduceOp.AVG)

    cum_loss = cum_loss.item()


    # Optimization step
    # unscale gradients
    state.scaler.unscale_(state.optimizer)
    # clip gradients, optimizer step
    norm = torch.nn.utils.clip_grad_norm_(state.model.parameters(), max_norm=10.0)
    state.scaler.step(state.optimizer)
    state.scaler.update()
    state.optimizer.zero_grad()

    # Logging
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()

    dt = time.time() - t0
    lr = state.scheduler.get_last_lr()[0]
    tokens_per_second = state.config.grad_accum_steps * WORLD_SIZE * state.config.batch_size * state.config.seq_length / dt
    print0(
        f"step {step+1:4d}/{state.config.max_steps} "
        f"| train loss {cum_loss:.6f} | ce loss {cum_ce_loss:.6f} "
        f"| kl loss {cum_kl_loss:.6f} | norm {norm:.4f} | lr {lr:.2e} "
        f"| ({(dt)*1000:.2f} ms | {tokens_per_second:.0f} tok/s)"
        f"| Mem {device_memory_used(device):.2f} GB"
    )
    return state


@torch.inference_mode()
def validation_step(state: TrainContext, step: int):
    """Runs the validation loop with AMP."""
    if (step + 1) % state.config.val_interval != 0:
        return state
    state.model.eval()
    total_loss, total_ce, total_kl = 0.0, 0.0, 0.0

    for _ in range(state.config.val_batches):
        x, y = state.val_loader.next_batch()
        x, y = x.to(state.device), y.to(state.device)

        # Forward pass with AMP context for validation
        with torch.autocast(device_type=state.device.type, dtype=state.dtype, enabled=state.config.use_amp):
            teacher_logits = state.teacher(x).logits
            student_logits = state.model(x).logits
            loss, loss_ce, loss_kl = calculate_loss(state, student_logits, teacher_logits, y)

        total_loss += loss.item()
        total_ce += loss_ce
        total_kl += loss_kl

    state.model.train()

    if DDP_ENABLED:
        total_losses_tensor = torch.tensor([total_loss, total_ce, total_kl], device=state.device)
        dist.all_reduce(total_losses_tensor, op=dist.ReduceOp.SUM)
        total_loss, total_ce, total_kl = total_losses_tensor.tolist()

    avg_loss = total_loss / (state.config.val_batches * WORLD_SIZE)
    avg_ce = total_ce / (state.config.val_batches * WORLD_SIZE)
    avg_kl = total_kl / (state.config.val_batches * WORLD_SIZE)

    return state


@rank_zero_only
def save_checkpoint(state, step):
    """Saves a robust checkpoint for resuming training."""
    if step % state.config.save_interval != 0:
        return
    ckpt_dir = Path(state.config.checkpoint_dir)
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_path = ckpt_dir / f"whole_{step}.pt"

    # Save all necessary states for a full resume
    torch.save(
        {
            "step": step,
            "model_state_dict": de_parallel(state.model).state_dict(),
            "optimizer_state_dict": state.optimizer.state_dict(),
            "scheduler_state_dict": state.scheduler.state_dict(),
            "config": vars(state.config),
        },
        ckpt_path,
    )

    LOGGER.info(f"Saved checkpoint to {ckpt_path}")


def train(state: TrainContext):
    """The main training loop with integrated AMP and resuming."""
    if DDP_ENABLED:
        dist.init_process_group(backend="nccl")
        state.device = torch.device(f"cuda:{LOCAL_RANK}")
        torch.cuda.set_device(state.device)
    else:
        state.device = device_parse(state.config.device)


    # Setup AMP dtype and GradScaler
    config = state.config
    init_seeds(config.seed)


    dtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[config.dtype]
    use_amp = config.use_amp and state.device.type not in {"cpu", "mps"}
    state.amp = use_amp
    state.dtype = dtype
    state.scaler = torch.amp.GradScaler(enabled=use_amp)

    state.teacher, state.model = setup_models(state)

    if DDP_ENABLED:
        state.model = DDP(state.model, device_ids=[LOCAL_RANK])
    state.optimizer = optim.AdamW(state.model.parameters(), lr=state.config.lr, weight_decay=state.config.weight_decay)
    state.scheduler = CosineAnnealingLR(state.optimizer, T_max=state.config.max_steps, eta_min=state.config.min_lr)
    setup_dataloaders(state)

    print0("Starting training...")
    state.optimizer.zero_grad()

    # Loop starts from `start_step`
    for step in range(state.config.max_steps):
        training_step(state, step)
        validation_step(state, step)
        save_checkpoint(state, step + 1)

    print0("Training completed!")
    if DDP_ENABLED:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a student LLaMA model via full model distillation with advanced features.")

    # Model and Data Paths
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    parser.add_argument("--config", type=str, default="config/distill-whole.yml", help="Path to YAML config file.")
    parser.add_argument("--model_config_path", type=str, default="data/MobileLLM/config.json", help="Path to model config JSON.")
    parser.add_argument("--model_weights_path", type=str, default="data/MobileLLM/model.safetensors", help="Path to teacher model weights (.safetensors).")
    parser.add_argument("--data_path", type=str, default="data/minipile", help="Path to training dataset.")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_full_model", help="Directory to save checkpoints.")
    parser.add_argument("--from_checkpoint", type=str, default="checkpoints/checkpoint_last.pt", help="Path to a checkpoint file to resume training from.")
    parser.add_argument("--val", type=str, default="test", help="Dataset split to use for validation.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use: cuda or cpu.")

    # Training Hyperparameters
    parser.add_argument("--max_steps", type=int, default=20000, help="Total number of training steps.")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size per device.")
    parser.add_argument("--seq_length", type=int, default=512, help="Sequence length for training.")
    parser.add_argument("--grad_accum_steps", type=int, default=4, help="Number of gradient accumulation steps.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate.")
    parser.add_argument("--min_lr", type=float, default=1e-5, help="Minimum learning rate for cosine annealing.")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay factor.")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping threshold.")

    # Distillation Hyperparameters
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight for KL loss vs CE loss.")
    parser.add_argument("--temperature", type=float, default=2.0, help="Temperature scaling for distillation.")
    # Performance & Mixed Precision
    parser.add_argument("--use_amp", action="store_true", help="Enable Automatic Mixed Precision (AMP).")
    parser.add_argument("--amp_dtype", type=str, default="bfloat16", choices=["float16", "bfloat16"], help="AMP data type (float16 or bfloat16).")

    # Logging and Saving
    parser.add_argument("--log_interval", type=int, default=10, help="Steps between logging.")
    parser.add_argument("--save_interval", type=int, default=1000, help="Steps between saving checkpoints.")
    parser.add_argument("--val_interval", type=int, default=250, help="Steps between validation runs.")
    parser.add_argument("--val_batches", type=int, default=10, help="Number of batches to evaluate during validation.")

    # Student Model Configuration
    parser.add_argument("--factorization_rank", type=int, default=16, help="Low-rank factorization rank for approximated layers.")
    parser.add_argument("--layer_sharing", action="store_true", help="Enable weight sharing across student layers.")
    args = parser.parse_args()

    args = vars(args)
    if args["config"]:
        args = yaml_load(args.pop("config"))['params']
        # args = {**base, **args}  # Command-line args override config file
    train_config = TrainingConfig(**args)
    state = TrainContext(train_config)
    train(state)



# salloc -A IscrC_LAM-next -p boost_usr_prod --qos=boost_qos_lprod --gres=gpu:1 --mem=0 --time=10:00:00
# srun --pty bash
# python scripts/distill_whole_model.py --config "config/distill-whole.yml"
# torchrun --standalone --nproc_per_node=4 scripts/distill_whole_model.py --config "config/distill-whole.yml"
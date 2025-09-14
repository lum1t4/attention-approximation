# scripts/distill_full_model.py
"""
This script trains a student LLaMA model with approximated attention layers
by distilling knowledge from a larger, frozen teacher model.

It incorporates several advanced features:
1.  Automatic Mixed Precision (AMP) for faster training and reduced memory usage.
2.  Memory-efficient loss chunking to handle long sequences.
3.  Robust checkpointing and resuming for long training runs.

The distillation is performed using a combination of KL Divergence and Cross-Entropy loss.
"""

import time
import argparse
from copy import copy
from pathlib import Path
from dataclasses import dataclass, fields

import torch
import torch.optim as optim
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.parallel import DistributedDataParallel as DDP

import safetensors
from attention_approximation.modeling_llama import LlamaForCausalLM as TeacherModel
from attention_approximation.modeling_llama_approximated import LlamaModel as StudentModel
from attention_approximation.pytorch import (
    device_parse,
    intersect_dicts,
    WORLD_SIZE,
    LOCAL_RANK,
    RANK,
    rank_zero_only,
    de_parallel,
)
from attention_approximation.utils import LOGGER
from distill_individual_layers import DistributedDataLoader


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

    # Training Hyperparameters
    max_steps: int = 20000
    batch_size: int = 2
    seq_length: int = 512  # Increased default to show value of chunking
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    min_learning_rate: float = 1e-5
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
    # (No changes from previous version)
    teacher_config = TeacherModel.config_class.from_json_file(state.config.model_config_path)
    teacher_model = TeacherModel(teacher_config)
    ckpt = safetensors.torch.load_file(state.config.model_weights_path)
    csd = intersect_dicts(ckpt, teacher_model.state_dict())
    teacher_model.load_state_dict(csd, strict=False)
    teacher_model.to(state.device)
    teacher_model.requires_grad_(False)
    teacher_model.eval()
    print0(f"Transferred {len(csd)}/{len(teacher_model.state_dict())} items from pretrained weights")

    student_config = copy(teacher_config)
    student_config.factorization_rank = state.config.factorization_rank
    student_config.layer_sharing = state.config.layer_sharing
    student_config.seq_length = state.config.seq_length
    student_model = StudentModel(student_config)

    ckpt = torch.load(state.config.from_checkpoint, map_location="cpu")['model_state_dict']
    def rename_key(key):
        key = key.replace("model.", "")
        key = key.replace(".self_attn.student_att.", ".self_attn.")
        return key

    ckpt = {rename_key(k) : v for k, v in ckpt.items()}
    csd = intersect_dicts(ckpt, student_model.state_dict())
    student_model.load_state_dict(csd, strict=False)
    student_model.to(state.device)
    student_model.train()
    LOGGER.info(f"Transferred {len(csd)}/{len(student_model.state_dict())} items from pretrained weights")

    if DDP_ENABLED:
        dist.barrier()
        student_model = DDP(student_model, device_ids=[LOCAL_RANK])

    return teacher_model, student_model


def setup_optimizer(state: TrainContext):
    # (No changes from previous version)
    model_to_optimize = de_parallel(state.student_model)
    return optim.AdamW(
        model_to_optimize.parameters(),
        lr=state.config.learning_rate,
        weight_decay=state.config.weight_decay,
    )


def setup_scheduler(state: TrainContext):
    # (No changes from previous version)
    return CosineAnnealingLR(
        state.optimizer, T_max=state.config.max_steps, eta_min=state.config.min_learning_rate
    )


def setup_dataloaders(state: TrainContext):
    # (No changes from previous version)
    train_loader = DistributedDataLoader(
        path=Path(state.config.data_path),
        batch_size=state.config.batch_size,
        seq_len=state.config.seq_length,
        split="train",
    )
    val_loader = DistributedDataLoader(
        path=Path(state.config.data_path),
        batch_size=state.config.batch_size,
        seq_len=state.config.seq_length,
        split=state.config.val,
    )
    return train_loader, val_loader


def calculate_loss(
    state: TrainContext,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    targets: torch.Tensor,
):
    """Calculates combined loss, with optional memory-efficient chunking for KL term."""
    # Cross-Entropy Loss
    loss_ce = F.cross_entropy(student_logits.view(-1, student_logits.size(-1)), targets.view(-1))

    # KL Divergence Distillation Loss
    student_logits_flat = student_logits.view(-1, student_logits.size(-1))
    teacher_logits_flat = teacher_logits.view(-1, teacher_logits.size(-1))

    T = state.config.temperature

    # Check if chunking is enabled and necessary
    if (
        state.config.loss_chunk_size > 0
        and student_logits_flat.size(0) > state.config.loss_chunk_size
    ):
        # Memory-efficient chunked computation
        kl_loss = torch.tensor(0.0, device=state.device, dtype=student_logits.dtype)
        num_tokens = student_logits_flat.size(0)

        for i in range(0, num_tokens, state.config.loss_chunk_size):
            end = i + state.config.loss_chunk_size
            student_chunk = student_logits_flat[i:end]
            teacher_chunk = teacher_logits_flat[i:end]

            student_log_probs = F.log_softmax(student_chunk / T, dim=-1)
            teacher_log_probs = F.log_softmax(teacher_chunk / T, dim=-1)

            # Use log_target=True for efficiency, reduction='sum' to aggregate over chunks
            chunk_kl = F.kl_div(
                student_log_probs, teacher_log_probs, reduction="sum", log_target=True
            )
            kl_loss += chunk_kl

        # Average the loss over all tokens
        kl_loss /= num_tokens

    else:
        # Direct computation for smaller tensors
        student_log_probs = F.log_softmax(student_logits_flat / T, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits_flat / T, dim=-1)
        kl_loss = F.kl_div(
            student_log_probs, teacher_log_probs, reduction="batchmean", log_target=True
        )

    loss_kl = kl_loss * (T**2)
    alpha = state.config.alpha
    combined_loss = alpha * loss_kl + (1.0 - alpha) * loss_ce

    return combined_loss, loss_ce.item(), loss_kl.item()


def training_step(state: TrainContext, batch, step: int):
    """Performs a single training step, including forward and backward passes with AMP."""
    x, y = batch
    x, y = x.to(state.device), y.to(state.device)

    # Forward pass with AMP
    with torch.autocast(
        device_type=state.device.type, dtype=state.dtype, enabled=state.config.use_amp
    ):
        with torch.inference_mode():
            teacher_logits = state.teacher_model(x).logits
        student_logits = state.student_model(x).logits
        loss, loss_ce, loss_kl = calculate_loss(state, student_logits, teacher_logits, y)

    scaled_loss = loss / state.config.gradient_accumulation_steps

    # Backward pass with GradScaler
    if state.scaler:
        state.scaler.scale(scaled_loss).backward()
    else:
        scaled_loss.backward()

    if (step + 1) % state.config.gradient_accumulation_steps == 0:
        model_to_clip = de_parallel(state.student_model)

        # Unscale gradients before clipping
        if state.scaler:
            state.scaler.unscale_(state.optimizer)

        torch.nn.utils.clip_grad_norm_(model_to_clip.parameters(), state.config.grad_clip)

        # Optimizer step with GradScaler
        if state.scaler:
            state.scaler.step(state.optimizer)
            state.scaler.update()
        else:
            state.optimizer.step()

        state.scheduler.step()
        state.optimizer.zero_grad()

    return loss.item(), loss_ce, loss_kl


@torch.inference_mode()
def validate(state: TrainContext):
    """Runs the validation loop with AMP."""
    state.student_model.eval()
    total_loss, total_ce, total_kl = 0.0, 0.0, 0.0

    for _ in range(state.config.val_batches):
        x, y = state.val_loader.next_batch()
        x, y = x.to(state.device), y.to(state.device)

        # Forward pass with AMP context for validation
        with torch.autocast(device_type=state.device.type, dtype=state.dtype, enabled=state.config.use_amp):
            teacher_logits = state.teacher_model(x).logits
            student_logits = state.student_model(x).logits
            loss, loss_ce, loss_kl = calculate_loss(state, student_logits, teacher_logits, y)

        total_loss += loss.item()
        total_ce += loss_ce
        total_kl += loss_kl

    state.student_model.train()

    if DDP_ENABLED:
        total_losses_tensor = torch.tensor([total_loss, total_ce, total_kl], device=state.device)
        dist.all_reduce(total_losses_tensor, op=dist.ReduceOp.SUM)
        total_loss, total_ce, total_kl = total_losses_tensor.tolist()

    avg_loss = total_loss / (state.config.val_batches * WORLD_SIZE)
    avg_ce = total_ce / (state.config.val_batches * WORLD_SIZE)
    avg_kl = total_kl / (state.config.val_batches * WORLD_SIZE)

    return avg_loss, avg_ce, avg_kl


@rank_zero_only
def save_checkpoint(state, step):
    """Saves a robust checkpoint for resuming training."""
    model_to_save = de_parallel(state.student_model)
    ckpt_dir = Path(state.config.checkpoint_dir)
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_path = ckpt_dir / f"checkpoint_step_{step}.pt"

    # Save all necessary states for a full resume
    torch.save(
        {
            "step": step,
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": state.optimizer.state_dict(),
            "scheduler_state_dict": state.scheduler.state_dict(),
            "scaler_state_dict": state.scaler.state_dict() if state.scaler else None,
            "config": state.config,
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
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16}
    state.dtype = dtype_map.get(state.config.amp_dtype, torch.float32)
    use_cuda_for_amp = state.device.type == "cuda" and state.config.use_amp
    state.scaler = torch.amp.GradScaler(enabled=use_cuda_for_amp and state.dtype == torch.float16)
    print0(f"Using device: {state.device}, AMP enabled: {state.config.use_amp}, DType: {state.dtype}")

    state.teacher_model, state.student_model = setup_models(state)
    state.optimizer = setup_optimizer(state)
    state.scheduler = setup_scheduler(state)
    state.train_loader, state.val_loader = setup_dataloaders(state)


    print0("Starting training...")
    state.optimizer.zero_grad()

    running_loss, running_ce, running_kl = 0.0, 0.0, 0.0
    start_time = time.time()

    # Loop starts from `start_step`
    for step in range(state.config.max_steps):
        batch = state.train_loader.next_batch()
        loss, loss_ce, loss_kl = training_step(state, batch, step)
        running_loss += loss
        running_ce += loss_ce
        running_kl += loss_kl

        if (step + 1) % state.config.log_interval == 0:
            if LOCAL_RANK in {-1, 0}:
                # The logic here remains the same, just reporting the results
                # of the more efficient training step.
                avg_loss = running_loss / state.config.log_interval
                avg_ce = running_ce / state.config.log_interval
                avg_kl = running_kl / state.config.log_interval
                elapsed = time.time() - start_time
                tokens_per_sec = (
                    state.config.batch_size
                    * state.config.seq_length
                    * state.config.log_interval
                    * WORLD_SIZE
                ) / elapsed
                print0(
                    f"Step {step + 1}/{state.config.max_steps} | "
                    f"Loss: {avg_loss:.4f} (CE: {avg_ce:.4f}, KL: {avg_kl:.4f}) | "
                    f"LR: {state.scheduler.get_last_lr()[0]:.2e} | "
                    f"Tokens/s: {tokens_per_sec:.0f}"
                )
                running_loss, running_ce, running_kl = 0.0, 0.0, 0.0
                start_time = time.time()

        if (step + 1) % state.config.val_interval == 0:
            val_loss, val_ce, val_kl = validate(state)
            print0(f"Validation Loss: {val_loss:.4f} (CE: {val_ce:.4f}, KL: {val_kl:.4f})")

        if (step + 1) % state.config.save_interval == 0:
            save_checkpoint(state, step + 1)

    print0("Training completed!")
    if DDP_ENABLED:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a student LLaMA model via full model distillation with advanced features.")

    # Model and Data Paths
    parser.add_argument("--model-config-path", type=str, default="data/MobileLLM/config.json", help="Path to model config JSON.")
    parser.add_argument("--model-weights-path", type=str, default="data/MobileLLM/model.safetensors", help="Path to teacher model weights (.safetensors).")
    parser.add_argument("--data-path", type=str, default="data/edu_fineweb10B", help="Path to training dataset.")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints_full_model", help="Directory to save checkpoints.")
    parser.add_argument("--from_checkpoint", type=str, required=True, help="Path to a checkpoint file to resume training from.")
    parser.add_argument("--val", type=str, default="test", help="Dataset split to use for validation.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use: cuda or cpu.")

    # Training Hyperparameters
    parser.add_argument("--max-steps", type=int, default=20000, help="Total number of training steps.")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size per device.")
    parser.add_argument("--seq-length", type=int, default=512, help="Sequence length for training.")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4, help="Number of gradient accumulation steps.")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Initial learning rate.")
    parser.add_argument("--min-learning-rate", type=float, default=1e-5, help="Minimum learning rate for cosine annealing.")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay factor.")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping threshold.")

    # Distillation Hyperparameters
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight for KL loss vs CE loss.")
    parser.add_argument("--temperature", type=float, default=2.0, help="Temperature scaling for distillation.")
    parser.add_argument("--loss-chunk-size", type=int, default=1024, help="Chunk size for KL computation. 0 disables chunking.")

    # Performance & Mixed Precision
    parser.add_argument("--use-amp", action="store_true", help="Enable Automatic Mixed Precision (AMP).")
    parser.add_argument("--amp-dtype", type=str, default="bfloat16", choices=["float16", "bfloat16"], help="AMP data type (float16 or bfloat16).")

    # Logging and Saving
    parser.add_argument("--log-interval", type=int, default=10, help="Steps between logging.")
    parser.add_argument("--save-interval", type=int, default=1000, help="Steps between saving checkpoints.")
    parser.add_argument("--val-interval", type=int, default=250, help="Steps between validation runs.")
    parser.add_argument("--val-batches", type=int, default=10, help="Number of batches to evaluate during validation.")

    # Student Model Configuration
    parser.add_argument("--factorization-rank", type=int, default=16, help="Low-rank factorization rank for approximated layers.")
    parser.add_argument("--layer-sharing", action="store_true", help="Enable weight sharing across student layers.")

    args = parser.parse_args()
    train_config = TrainingConfig(**vars(args))
    state = TrainContext(train_config)
    train(state)

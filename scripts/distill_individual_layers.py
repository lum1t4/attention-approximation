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
from attention_approximation.data import DistributedDataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from attention_approximation.trackers import Tracker, AutoTracker

DDP_ENABLED = WORLD_SIZE > 1

class CP(nn.Module):
    def __init__(self, rank: int, out_units: int = 1):
        super().__init__()
        self.rank = int(rank)
        self.out_units = int(out_units)
        self.register_buffer('weight', torch.ones(self.out_units, self.rank))

    def forward(self, hadamard: torch.Tensor) -> torch.Tensor:
        return hadamard @ self.weight.t()

class CPCircuitLayer(nn.Module):
    def __init__(self, config: LlamaConfig, chunk_size: int = 1_000):
        super().__init__()
        self.out_units = 1
        self.rank = config.factorization_rank
        self.chunk_size = chunk_size

        self.seq_mode_factor = nn.Linear(
            config.hidden_size, config.factorization_rank, bias=config.attention_bias
        )
        # shape: [hidden_size, rank]
        self.hidden_embeddings = nn.Parameter(
            torch.empty(config.hidden_size, config.factorization_rank)
            )
        #initilaize with a small Gaussian like Transformers embeddings
        init.normal_(self.hidden_embeddings, mean=0.0, std=0.02)

        self.cp = CP(rank=config.factorization_rank, out_units=self.out_units)

    def forward(self, hidden_states: torch.Tensor, all_indices: torch.Tensor) -> torch.Tensor:
        batch, seq_len, hidden_size = hidden_states.size()
        device = hidden_states.device

        # Move indices to correct device
        all_indices = all_indices.to(device)

        # seq_embeddings: [batch, seq_len, rank]
        seq_embeddings = self.seq_mode_factor(hidden_states)

        outputs = []
        for start in range(0, all_indices.size(0), self.chunk_size):
            end = min(start + self.chunk_size, all_indices.size(0))
            chunk = all_indices[start:end]  # [chunk_size, 2]

            seq_indices = chunk[:, 0].long()  # [chunk_size]
            hidden_indices = chunk[:, 1].long()  # [chunk_size]

            # Gather sequence embeddings for batch
            seq_emb = seq_embeddings[:, seq_indices]  # [batch, chunk_size, rank]

            # Gather hidden embeddings (shared across batch)
            hidden_emb = self.hidden_embeddings[hidden_indices]  # [chunk_size, rank]
            hidden_emb = hidden_emb.unsqueeze(0).expand(batch, -1, -1)  # [batch, chunk_size, rank]

            # Hadamard product
            hadamard = seq_emb * hidden_emb  # [batch, chunk_size, rank]

            # CP projection
            out_chunk = self.cp(hadamard)  # [batch, chunk_size, out_units]
            outputs.append(out_chunk)

        out = torch.cat(outputs, dim=1)  # [batch, total_size, out_units]

        return out.view(batch, seq_len, hidden_size, self.out_units).squeeze(3)


class LlamaApproximatedAttention(nn.Module):
    def __init__(self, config: LlamaConfig, all_indices: torch.Tensor):
        super().__init__()
        self.all_indices = all_indices
        self.cp_circuit = CPCircuitLayer(config=config, chunk_size=10_000)

    def forward(self, hidden_states: torch.Tensor, **kargs) -> torch.Tensor:
        attn_output = self.cp_circuit(hidden_states, self.all_indices)
        return  attn_output


@dataclass
class TrainingConfig:
    """Stores all hyperparameters and configuration settings."""
    # Model and Data Paths
    model_config_path: str = "data/MobileLLM/config.json"
    model_weights_path: str = "data/MobileLLM/model.safetensors"
    data_path: str = "data/minipile"
    checkpoint_dir: str = "checkpoints"
    val: str = "test"
    device: str = "cuda"
    # Training Hyperparameters
    max_steps: int = 10000
    seed: int = 42
    batch_size: int = 2
    seq_length: int = 128
    grad_accum_steps: int = 4
    lr: float = 1e-3
    min_lr: float = 1e-5
    weight_decay: float = 1e-5
    warmup_steps: int = 100
    grad_clip: float = 1.0
    amp: bool = True

    # Logging and Saving
    save_interval: int = 1000
    val_interval: int = 250
    val_batches: int = 10

    # Student Model Configuration
    factorization_rank: int = 16
    layer_sharing: bool = False

    tracker: str = None
    project: str = "attention-approximation"
    name: str = None


# Enable TF32 for faster matmuls on Ampere+ GPUs
torch.set_float32_matmul_precision('high')


@rank_zero_only
def print0(s: str):
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
    
    def train(self, mode: bool = True):
        self.student_att.train(mode)
        self.teacher_att.eval()  # Ensure teacher is always in eval mode
        return self


class TrainContext:
    config: TrainingConfig
    model: nn.Module | nn.parallel.DistributedDataParallel
    device: torch.device
    tracker: Tracker

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.tracker = AutoTracker(name=config.tracker, config=vars(config))



def setup_model(state):
    """Loads, patches, and prepares the model for training."""
    # model, train, student configs
    m_config = LlamaConfig().from_json_file(state.config.model_config_path)
    m_config.use_cache = False
    t_config = state.config
    s_config = copy(m_config)
    s_config.factorization_rank = state.config.factorization_rank
    s_config.layer_sharing = state.config.layer_sharing
    s_config.seq_length = state.config.seq_length

    # Load teacher model
    model = TeacherModel(m_config)
    checkpoint = safetensors.torch.load_file(t_config.model_weights_path)
    csd = intersect_dicts(checkpoint, model.state_dict())
    model.load_state_dict(csd, strict=False)
    print0(f"Transferred {len(csd)}/{len(model.state_dict())} items from pretrained weights")
    model.eval()

    # Patch attention layers
    grid_y, grid_x = torch.meshgrid(
        torch.arange(state.config.seq_length, dtype=torch.long),
        torch.arange(m_config.hidden_size, dtype=torch.long),
        indexing="ij"
    )
    all_indices = torch.stack([grid_y, grid_x], dim=-1).view(-1, 2)
    s_config.all_indices = all_indices.to(state.device)

    for i, layer in enumerate(model.model.layers):
        layer.self_attn = AttentionDistillationWrapper(
            student_att=LlamaApproximatedAttention,
            teacher_att=layer.self_attn,
            config=s_config
        )
        if LOCAL_RANK in {-1, 0}:
            LOGGER.info(f"Replaced attention in layer {i}")

    model = model.to(state.device)
    model = torch.compile(model)
    
    student_params = []
    for layer in model.model.layers:
        student_params.extend(layer.self_attn.student_att.parameters())

    if DDP_ENABLED:
        model = DDP(model, device_ids=[LOCAL_RANK])
    state.model = model
    state.student_params = student_params
    num_trainable_params = sum(p.numel() for p in student_params)
    print0(f"Total student parameters to train: {num_trainable_params:,}")
    state.tracker.log("model/trainable_params", num_trainable_params)
    state.tracker.log("model/factorization_rank", t_config.factorization_rank)
    return state


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


def training_step(state: TrainContext, step: int):
    """Performs a single training step, including forward and backward passes."""
    # Forward pass is performed through the possibly-wrapped model
    state.model.train()

    t0 = time.time()
    state.optimizer.zero_grad()
    device, amp = state.device, state.config.amp
    cum_loss = 0.0

    for k in range(state.config.grad_accum_steps):
        x, y = state.train_loader.next_batch()
        x, y = x.to(state.device, non_blocking=True), y.to(state.device, non_blocking=True)
        if DDP_ENABLED:
            # Avoid unnecessary gradient syncs
            state.model.require_backward_grad_sync = (k == state.config.grad_accum_steps - 1)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            state.model(x)
            loss = torch.tensor(0.0, device=device, requires_grad=False)
            num_att_layers = 0
            for layer in de_parallel(state.model).model.layers:
                if hasattr(layer.self_attn, 'last_distillation_loss'):
                    num_att_layers += 1
                    loss += layer.self_attn.last_distillation_loss
            loss /= (state.config.grad_accum_steps * num_att_layers)
            cum_loss += loss.detach()
            # Backward pass
            state.scaler.scale(loss).backward()

    if DDP_ENABLED:
        dist.all_reduce(cum_loss, op=dist.ReduceOp.AVG)
        cum_loss = cum_loss.item()

    
    state.scaler.unscale_(state.optimizer)
    torch.nn.utils.clip_grad_norm_(state.student_params, state.config.grad_clip)
    state.scaler.step(state.optimizer)
    state.scaler.update()
    state.optimizer.zero_grad()

    device_synchronize(device)
    dt = time.time() - t0
    lr = state.scheduler.get_last_lr()[0]
    tokens_per_second = (
        state.config.grad_accum_steps
        * WORLD_SIZE
        * state.config.batch_size
        * state.config.seq_length
    ) / dt
    print0(
        f"step {step+1}/{state.config.max_steps} | "
        f"distill Loss: {cum_loss:.4f}  | "
        f"lr: {lr:.2e} | "
        f"tokens/s: {tokens_per_second:.0f} | "
        f"mem {device_memory_used(state.device):.2f} GB | "
        f"bs {state.config.batch_size}x{WORLD_SIZE} | "
        f"factorization_rank {state.config.factorization_rank}" 
    )

    state.tracker.log({
        "train/loss": cum_loss,
        "train/lr": lr,
        "train/tokens_per_sec": tokens_per_second,
        "train/mem_used": device_memory_used(state.device)
    }, step=step)

    gc.collect()
    device_memory_clear(state.device)
    return cum_loss


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

    return (total_loss / state.config.val_batches).item()


@rank_zero_only
def save_checkpoint(state, step):
    """Saves a training checkpoint"""
    if step % state.config.save_interval == 0 or step == state.config.max_steps -1:
        model = de_parallel(state.model)
        ckpt = Path(state.config.checkpoint_dir)
        ckpt.mkdir(exist_ok=True)
        import io
        buffer = io.BytesIO()

        torch.save({
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': state.optimizer.state_dict(),
            'scheduler_state_dict': state.scheduler.state_dict(),
        }, buffer)
        (ckpt / f"checkpoint_last.pt").write_bytes(buffer.getvalue())
        ckpt = ckpt / f"checkpoint_step_{step + 1}.pt"
        (ckpt).write_bytes(buffer.getvalue())
        LOGGER.info(f"Saved checkpoint to {ckpt}")

    return state


def train(state: TrainContext):
    """The main training loop."""
    import os
    os.environ['WANDB_MODE'] = 'offline'
    os.environ['OMP_NUM_THREADS'] = "1"
    if DDP_ENABLED:
        dist.init_process_group(backend="nccl")
        state.device = torch.device(f"cuda:{LOCAL_RANK}")
        torch.cuda.set_device(LOCAL_RANK)
    else:
        state.device = device_parse(state.config.device)
    
    init_seeds(state.config.seed)
    state = setup_model(state)
    state.config.amp = state.device.type == "cuda" and state.config.amp
    print0("Starting training...")
    state.optimizer.zero_grad()
    # num steps to do 100M tokens
    num_steps = int(100 * 10**6 / (
        state.config.seq_length
        * state.config.batch_size
        * WORLD_SIZE
        * state.config.grad_accum_steps
    ))
    state.config.max_steps = min(state.config.max_steps, num_steps)
    state.optimizer = optim.AdamW(state.student_params, lr=state.config.lr, weight_decay=state.config.weight_decay)
    state.scheduler = CosineAnnealingLR(state.optimizer, T_max=state.config.max_steps, eta_min=state.config.min_lr)
    state.scaler = torch.amp.GradScaler(enabled=state.config.amp)
    state.train_loader, state.val_loader = setup_dataloaders(state)

    for step in range(num_steps):
        loss = training_step(state, step)
        if (step + 1) % state.config.val_interval == 0:
            val_loss = validate(state)
            print0(f"Validation Distill Loss: {val_loss:.4f}")
        
        save_checkpoint(state, step + 1)

    if DDP_ENABLED:
        dist.destroy_process_group()

    gc.collect()
    device_memory_clear(state.device)
    print0("Training completed!")



if __name__ == "__main__":
    """Parses command-line arguments, creates config and context, then starts training."""
    parser = argparse.ArgumentParser(description="Train replaced attention layers in a LLaMA model via distillation.")
    # Model and Data Paths
    parser.add_argument("--config", type=str, help="Path to YAML config file.")

    parser.add_argument("--model_config_path", type=str, help="Path to the LLaMA model config JSON file.")
    parser.add_argument("--model_weights_path", type=str, help="Path to the pretrained model weights (safetensors).")
    parser.add_argument("--data_path", type=str, help="Path to the training/validation dataset shards.")
    parser.add_argument("--val", type=str, help="Dataset split to use for validation.")
    parser.add_argument("--checkpoint_dir", type=str, help="Directory to save checkpoints.")
    parser.add_argument("--device", type=str, help="Device to train on (cuda, cpu, mps).")

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
    parser.add_argument("--val_batches", type=int, help="Number of validation batches to average over.")

    # Student Model Configuration
    parser.add_argument("--factorization_rank", type=int, help="Factorization rank for approximated attention.")
    parser.add_argument("--layer_sharing", action="store_true", help="Enable layer sharing in student attention.")

    parser.add_argument("--tracker", type=str)
    parser.add_argument("--name", type=str)
                        
    args = parser.parse_args()
    args = {k: v for k, v in vars(args).items() if v is not None}
    if args["config"]:
        base = yaml_load(args.pop("config"))['params']
        args = {**base, **args}  # Command-line args override config file
    train_config = TrainingConfig(**args)
    state = TrainContext(train_config)
    train(state)

# salloc -A IscrC_LAM-next -p boost_usr_prod --qos=boost_qos_lprod --gres=gpu:4 --mem=0 --time=10:00:00
# srun --pty bash
# cd $FAST/attention-approximation/ && source .venv/bin/activate
# python scripts/distill_individual_layers.py --config "config/distll-layers.yml"
# torchrun --standalone --nproc_per_node=4 scripts/distill_individual_layers.py --config "config/distll-layers.yml"
from typing import Literal
import torch
from dataclasses import dataclass, fields
from torch import nn
from attention_approximation.trackers import Tracker, AutoTracker
from attention_approximation.pytorch import (
    WORLD_SIZE, LOCAL_RANK, device_parse, init_seeds,
    rank_zero_only, de_parallel
)
from attention_approximation.data import distributed_dataloader, TokenDataset
from pathlib import Path
from attention_approximation.utils import LOGGER
import io


@dataclass
class BaseTrainConfig:
    """Base configuration class with shared training attributes."""
    # Model and Data Paths
    model_config_path: str
    model_weights_path: str
    data_path: str
    checkpoint_dir: str
    val: str
    device: str
    dtype: Literal['float32', 'bfloat16', 'float16']
    workers: int

    # Training Hyperparameters
    max_steps: int
    seed: int
    batch_size: int
    seq_length: int
    grad_accum_steps: int
    lr: float
    weight_decay: float
    grad_clip: float
    amp: bool

    # Logging and Saving
    save_interval: int
    val_interval: int

    # Student Model Configuration
    factorization_rank: int
    layer_sharing: bool = False
    
    # Experiment Tracking
    tracker: str = None
    project: str = "attention-approximation"
    name: str = None

    min_lr: float = 1e-5
    warmup_steps: int = 100


class BaseTrainContext:
    model: nn.Module | nn.parallel.DistributedDataParallel
    config: BaseTrainConfig
    device: torch.device
    tracker: Tracker
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    train_loader: torch.utils.data.DataLoader
    valid_loader: torch.utils.data.DataLoader | None
    scaler: torch.cuda.amp.GradScaler
    distributed: bool = WORLD_SIZE > 1

    def __init__(self, config: BaseTrainConfig):
        import os
        os.environ['WANDB_MODE'] = 'offline'
        os.environ['OMP_NUM_THREADS'] = "1"
        self.distributed = WORLD_SIZE > 1
        if self.distributed:
            torch.distributed.init_process_group(backend="nccl")
            self.device = torch.device(f"cuda:{LOCAL_RANK}")
            torch.cuda.set_device(LOCAL_RANK)
        else:
            self.device = device_parse(config.device)

        init_seeds(config.seed)
        config.amp = \
            self.device.type not in {"cpu", "mps"} \
            and config.amp \
            and config.dtype != 'float32'

        self.config = config
        self.dtype = {
            'float32': torch.float32,
            'bfloat16': torch.bfloat16,
            'float16': torch.float16
        }[config.dtype]
        self.scaler = torch.amp.GradScaler(enabled=config.amp and config.dtype == "float16")
        # Enable TF32 for faster matmuls on Ampere+ GPUs
        torch.set_float32_matmul_precision('high')
        self.tracker = AutoTracker(name=config.tracker, config=vars(config))


@rank_zero_only
def print0(s: str):
    LOGGER.info(s)


def setup_dataloaders(ctx: BaseTrainContext):
    """Initializes the data loaders for training and validation."""
    config = ctx.config
    train_dataset = TokenDataset(config.data_path, seq_len=config.seq_length, split='train')
    valid_dataset = TokenDataset(config.data_path, seq_len=config.seq_length, split=config.val)
    ctx.train_loader = distributed_dataloader(train_dataset, workers=config.workers, batch=config.batch_size, mode="train")
    ctx.valid_loader = distributed_dataloader(valid_dataset, workers=config.workers, batch=config.batch_size, mode="valid")
    return ctx


def strip_torch_compile(state_dict, prefix: str = "_orig_mod."):
    """Strip torch.compile `_orig_mod.` prefixes before persisting."""
    return {
        (k.replace(prefix, "", 1) if k.startswith(prefix) else k): v
        for k, v in state_dict.items()
    }


@rank_zero_only
def save_checkpoint(ctx: BaseTrainContext, step: int):
    """Saves a training checkpoint"""
    if (step + 1) % ctx.config.save_interval == 0:
        model_state = strip_torch_compile(de_parallel(ctx.model).state_dict())
        ckpt = Path(ctx.config.checkpoint_dir)
        ckpt.mkdir(exist_ok=True)
        buffer = io.BytesIO()

        torch.save({
            'step': step,
            'model_state_dict': model_state,
            'optimizer_state_dict': ctx.optimizer.state_dict(),
            'scheduler_state_dict': ctx.scheduler.state_dict(),
            "config": vars(ctx.config),
        }, buffer)
        (ckpt / "last.pt").write_bytes(buffer.getvalue())
        (ckpt / f"step_{step + 1}.pt").write_bytes(buffer.getvalue())
        LOGGER.info(f"Saved checkpoint to {ckpt}/step_{step + 1}.pt")
    
    return

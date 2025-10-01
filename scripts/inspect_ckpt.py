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
    device_synchronize,
    init_seeds,
    intersect_dicts,
    rank_zero_only,
)
from attention_approximation.utils import LOGGER, yaml_load


@rank_zero_only
def print0(s: str):
    LOGGER.info(s)

@dataclass
class TrainingConfig:
    """Stores all hyperparameters and configuration settings."""
    from_checkpoint: str = "checkpoints/checkpoint_last.pt"
    # Model and Data Paths
    model_config_path: str = "data/MobileLLM/config.json"
    model_weights_path: str = "data/MobileLLM/model.safetensors"
    data_path: str = "data/edu_fineweb10B"
    checkpoint_dir: str = "checkpoints_full_model"
    # Student Model Configuration
    factorization_rank: int = 128
    layer_sharing: bool = False
    seq_length: int = 512

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


state = TrainContext(TrainingConfig())
state.device = torch.device("cuda:0")


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
for k in list(ckpt.keys()):
    print(k)


for k in model.state_dict().keys():
    print(k)

# Only rename the student_att part, keep model. prefix intact
def match_dict(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace('_orig_mod.', '')
        if 'self_attn.teacher_att.' not in k:
            new_key = new_key.replace(".self_attn.student_att.", ".self_attn.")
            new_state_dict[new_key] = v
    return new_state_dict

ckpt = match_dict(ckpt)
csd = intersect_dicts(ckpt, model.state_dict())
assert len(csd) > 0, "No weights were transferred from the checkpoint. Please check the checkpoint path and model architecture."
model.load_state_dict(csd, strict=False)
model = torch.compile(model.to(state.device))
LOGGER.info(f"Transferred {len(csd)}/{len(model.state_dict())} items from pretrained weights")
"""
This script implements knowledge distillation using final model logits
following the RADLADS Step 1 approach with KL divergence loss.
"""

import time
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import LlamaConfig
import safetensors.torch
from attention_approximation.modeling_llama import LlamaForCausalLM
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
from attention_approximation.data import DistributedDataLoader


DDP_ENABLED = WORLD_SIZE > 1


@dataclass
class LogitsDistillationConfig:
    """Configuration for logits-based knowledge distillation."""

    # Model and Data Paths
    teacher_config_path: str = "data/MobileLLM/config.json"
    teacher_weights_path: str = "data/MobileLLM/model.safetensors"
    student_config_path: str = "data/MobileLLM/config.json"  # Can be different
    student_weights_path: Optional[str] = None  # None for random init
    data_path: str = "data/edu_fineweb10B"
    checkpoint_dir: str = "checkpoints/logits_distillation"
    device: str = "mps"

    # Training Hyperparameters
    max_steps: int = 50000
    batch_size: int = 2
    seq_length: int = 512
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-5
    min_learning_rate: float = 1e-6
    weight_decay: float = 0.0
    warmup_steps: int = 500
    grad_clip: float = 1.0

    # Distillation Parameters
    kl_weight: float = 1.0  # Weight for KL divergence loss
    ce_weight: float = 0.0  # Weight for cross-entropy loss (optional)
    temperature: float = 1.0  # Temperature for softening distributions
    chunk_size: int = 512  # Chunk size for memory-efficient loss calculation

    # Logging and Saving
    log_interval: int = 10
    save_interval: int = 1000
    val_interval: int = 250
    val_batches: int = 10

    # Mixed Precision
    use_amp: bool = False
    dtype: str = "bfloat16"  # float32, float16, bfloat16


class DistillationTrainer:
    """Handles the logits-based knowledge distillation training process."""

    def __init__(self, config: LogitsDistillationConfig):
        self.config = config
        self.device = device_parse(config.device)

        # Setup models
        self.setup_models()

        # Setup data
        self.setup_data()

        # Setup optimization
        self.setup_optimization()

        # Metrics tracking
        self.metrics = {
            "distillation_loss": [],
            "ce_loss": [],
            "total_loss": [],
            "perplexity": [],
            "learning_rate": [],
        }

        # Training state
        self.global_step = 0
        self.best_val_loss = float("inf")

    def setup_models(self):
        """Initialize teacher and student models."""
        LOGGER.info("Loading teacher model...")

        # Load teacher model
        teacher_config = LlamaConfig.from_pretrained(self.config.teacher_config_path)
        self.teacher = LlamaForCausalLM(teacher_config)

        # Load teacher weights
        teacher_state = safetensors.torch.load_file(self.config.teacher_weights_path)
        self.teacher.load_state_dict(teacher_state, strict=True)
        self.teacher.to(self.device)
        self.teacher.eval()  # Teacher always in eval mode

        # Freeze teacher parameters
        for param in self.teacher.parameters():
            param.requires_grad = False

        LOGGER.info("Loading student model...")

        # Load student model
        student_config = LlamaConfig.from_pretrained(self.config.student_config_path)
        self.student = LlamaForCausalLM(student_config)

        # Load student weights if provided
        if self.config.student_weights_path:
            student_state = safetensors.torch.load_file(self.config.student_weights_path)
            self.student.load_state_dict(student_state, strict=True)
        else:
            LOGGER.info("Initializing student model with random weights")

        self.student.to(self.device)
        self.student.train()

        # Setup DDP if needed
        if DDP_ENABLED:
            self.student = DDP(self.student, device_ids=[LOCAL_RANK])
            # Note: Teacher doesn't need DDP as it's not being trained

        # Setup mixed precision
        self.setup_mixed_precision()

    def setup_mixed_precision(self):
        """Configure mixed precision training if enabled."""
        if self.config.use_amp:
            dtype_map = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }
            self.amp_dtype = dtype_map[self.config.dtype]
            self.scaler = torch.cuda.amp.GradScaler(enabled=(self.config.dtype == "float16"))
        else:
            self.amp_dtype = torch.float32
            self.scaler = None

    def setup_data(self):
        """Initialize data loaders."""
        self.train_loader = DistributedDataLoader(
            data_root=self.config.data_path,
            batch_size=self.config.batch_size,
            seq_length=self.config.seq_length,
            split="train",
        )

        self.val_loader = DistributedDataLoader(
            data_root=self.config.data_path,
            batch_size=self.config.batch_size,
            seq_length=self.config.seq_length,
            split="val",
        )

    def setup_optimization(self):
        """Initialize optimizer and scheduler."""
        # Get student parameters
        params = self.student.parameters()

        # Optimizer
        self.optimizer = optim.AdamW(
            params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.95),
            eps=1e-8,
        )

        # Learning rate scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=self.config.max_steps, eta_min=self.config.min_learning_rate
        )

    def compute_distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute the distillation loss using KL divergence.

        Args:
            student_logits: Logits from student model [batch_size, seq_len, vocab_size]
            teacher_logits: Logits from teacher model [batch_size, seq_len, vocab_size]
            labels: Optional labels for CE loss [batch_size, seq_len]

        Returns:
            total_loss: Combined distillation and CE loss
            loss_dict: Dictionary with individual loss components
        """
        # Reshape logits for loss calculation
        batch_size, seq_len, vocab_size = student_logits.shape
        student_logits_flat = student_logits.view(-1, vocab_size)
        teacher_logits_flat = teacher_logits.view(-1, vocab_size)

        # Apply temperature scaling
        student_logits_scaled = student_logits_flat / self.config.temperature
        teacher_logits_scaled = teacher_logits_flat / self.config.temperature

        # Compute KL divergence loss
        if self.config.chunk_size and student_logits_flat.size(0) > self.config.chunk_size:
            # Memory-efficient chunked computation
            kl_loss = torch.tensor(0.0, device=student_logits.device, dtype=student_logits.dtype)
            num_chunks = 0

            for i in range(0, student_logits_flat.size(0), self.config.chunk_size):
                end_idx = min(i + self.config.chunk_size, student_logits_flat.size(0))

                student_chunk = student_logits_scaled[i:end_idx]
                teacher_chunk = teacher_logits_scaled[i:end_idx]

                # KL(P||Q) where P is teacher, Q is student
                student_log_probs = F.log_softmax(student_chunk, dim=-1)
                teacher_log_probs = F.log_softmax(teacher_chunk, dim=-1)

                chunk_kl = F.kl_div(
                    student_log_probs, teacher_log_probs, log_target=True, reduction="sum"
                )

                kl_loss = kl_loss + chunk_kl
                num_chunks += end_idx - i

            kl_loss = kl_loss / num_chunks
        else:
            # Direct computation for smaller batches
            student_log_probs = F.log_softmax(student_logits_scaled, dim=-1)
            teacher_log_probs = F.log_softmax(teacher_logits_scaled, dim=-1)

            kl_loss = F.kl_div(
                student_log_probs, teacher_log_probs, log_target=True, reduction="batchmean"
            )

        # Scale KL loss by temperature squared (as per standard distillation)
        kl_loss = kl_loss * (self.config.temperature**2)

        # Compute optional CE loss
        ce_loss = torch.tensor(0.0, device=student_logits.device)
        if self.config.ce_weight > 0 and labels is not None:
            labels_flat = labels.view(-1)
            # Ignore padding tokens (typically -100)
            ce_loss = F.cross_entropy(
                student_logits_flat, labels_flat, ignore_index=-100, reduction="mean"
            )

        # Combine losses
        total_loss = self.config.kl_weight * kl_loss + self.config.ce_weight * ce_loss

        loss_dict = {
            "kl_loss": kl_loss.item(),
            "ce_loss": ce_loss.item(),
            "total_loss": total_loss.item(),
        }

        return total_loss, loss_dict

    def training_step(self, batch: dict) -> dict:
        """Execute a single training step."""
        input_ids = batch["input_ids"].to(self.device)
        labels = batch.get("labels", input_ids.clone())

        # Forward pass with mixed precision
        with torch.cuda.amp.autocast(enabled=self.config.use_amp, dtype=self.amp_dtype):
            # Get teacher logits (no grad needed)
            with torch.no_grad():
                teacher_outputs = self.teacher(input_ids=input_ids, use_cache=False)
                teacher_logits = teacher_outputs.logits

            # Get student logits
            student_outputs = self.student(input_ids=input_ids, use_cache=False)
            student_logits = student_outputs.logits

            # Compute distillation loss
            loss, loss_dict = self.compute_distillation_loss(
                student_logits, teacher_logits, labels if self.config.ce_weight > 0 else None
            )

        # Scale loss for gradient accumulation
        loss = loss / self.config.gradient_accumulation_steps

        # Backward pass
        if self.config.use_amp and self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        return loss_dict

    @torch.no_grad()
    def validation_step(self, batch: dict) -> dict:
        """Execute a single validation step."""
        input_ids = batch["input_ids"].to(self.device)
        labels = batch.get("labels", input_ids.clone())

        # Forward pass
        with torch.cuda.amp.autocast(enabled=self.config.use_amp, dtype=self.amp_dtype):
            # Get teacher logits
            teacher_outputs = self.teacher(input_ids=input_ids, use_cache=False)
            teacher_logits = teacher_outputs.logits

            # Get student logits
            student_outputs = self.student(input_ids=input_ids, use_cache=False)
            student_logits = student_outputs.logits

            # Compute distillation loss
            loss, loss_dict = self.compute_distillation_loss(
                student_logits, teacher_logits, labels if self.config.ce_weight > 0 else None
            )

        # Calculate perplexity
        if self.config.ce_weight > 0:
            perplexity = torch.exp(torch.tensor(loss_dict["ce_loss"]))
            loss_dict["perplexity"] = perplexity.item()
        else:
            # Estimate perplexity from KL loss
            loss_dict["perplexity"] = 0.0

        return loss_dict

    def train(self):
        """Main training loop."""
        LOGGER.info(f"Starting training for {self.config.max_steps} steps")

        train_losses = []
        grad_accum_counter = 0

        for step in range(self.config.max_steps):
            self.global_step = step

            # Training step
            batch = next(self.train_loader)
            loss_dict = self.training_step(batch)
            train_losses.append(loss_dict)

            grad_accum_counter += 1

            # Optimizer step
            if grad_accum_counter == self.config.gradient_accumulation_steps:
                # Gradient clipping
                if self.config.grad_clip > 0:
                    if self.config.use_amp and self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.student.parameters(), self.config.grad_clip
                    )

                # Optimizer step
                if self.config.use_amp and self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.scheduler.step()
                self.optimizer.zero_grad()
                grad_accum_counter = 0

            # Logging
            if step % self.config.log_interval == 0 and step > 0:
                avg_losses = {
                    k: sum(d[k] for d in train_losses) / len(train_losses)
                    for k in train_losses[0].keys()
                }

                lr = self.scheduler.get_last_lr()[0]

                LOGGER.info(
                    f"Step {step}/{self.config.max_steps} | "
                    f"KL Loss: {avg_losses['kl_loss']:.4f} | "
                    f"CE Loss: {avg_losses['ce_loss']:.4f} | "
                    f"Total Loss: {avg_losses['total_loss']:.4f} | "
                    f"LR: {lr:.2e}"
                )

                train_losses = []

            # Validation
            if step % self.config.val_interval == 0 and step > 0:
                self.validate()

            # Checkpoint saving
            if step % self.config.save_interval == 0 and step > 0:
                self.save_checkpoint(step)

        LOGGER.info("Training completed!")

    @torch.no_grad()
    def validate(self):
        """Run validation loop."""
        self.student.eval()

        val_losses = []
        for i in range(self.config.val_batches):
            batch = next(self.val_loader)
            loss_dict = self.validation_step(batch)
            val_losses.append(loss_dict)

        # Average validation metrics
        avg_losses = {
            k: sum(d[k] for d in val_losses) / len(val_losses) for k in val_losses[0].keys()
        }

        LOGGER.info(
            f"Validation | "
            f"KL Loss: {avg_losses['kl_loss']:.4f} | "
            f"CE Loss: {avg_losses['ce_loss']:.4f} | "
            f"Total Loss: {avg_losses['total_loss']:.4f} | "
            f"Perplexity: {avg_losses.get('perplexity', 0):.2f}"
        )

        # Save best model
        if avg_losses["total_loss"] < self.best_val_loss:
            self.best_val_loss = avg_losses["total_loss"]
            self.save_checkpoint("best")

        self.student.train()

    @rank_zero_only
    def save_checkpoint(self, tag: str):
        """Save model checkpoint."""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Get model state dict
        student_state = de_parallel(self.student).state_dict()

        # Save checkpoint
        checkpoint = {
            "model_state_dict": student_state,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "config": self.config,
            "best_val_loss": self.best_val_loss,
        }

        checkpoint_path = checkpoint_dir / f"checkpoint_{tag}.pt"
        torch.save(checkpoint, checkpoint_path)
        LOGGER.info(f"Saved checkpoint to {checkpoint_path}")

        # Also save model weights in safetensors format
        safetensors_path = checkpoint_dir / f"model_{tag}.safetensors"
        safetensors.torch.save_file(student_state, safetensors_path)
        LOGGER.info(f"Saved model weights to {safetensors_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        de_parallel(self.student).load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))

        LOGGER.info(f"Loaded checkpoint from {checkpoint_path}")
        LOGGER.info(f"Resuming from step {self.global_step}")


def main():
    parser = argparse.ArgumentParser(description="Logits-based Knowledge Distillation")

    # Model paths
    parser.add_argument(
        "--teacher_config",
        type=str,
        default="data/MobileLLM/config.json",
        help="Path to teacher model config",
    )
    parser.add_argument(
        "--teacher_weights",
        type=str,
        default="data/MobileLLM/model.safetensors",
        help="Path to teacher model weights",
    )
    parser.add_argument(
        "--student_config",
        type=str,
        default="data/MobileLLM/config.json",
        help="Path to student model config",
    )
    parser.add_argument(
        "--student_weights",
        type=str,
        default=None,
        help="Path to student model weights (optional)",
    )

    # Training parameters
    parser.add_argument("--max_steps", type=int, default=50000, help="Maximum training steps")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size per device")
    parser.add_argument("--seq_length", type=int, default=512, help="Sequence length")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Initial learning rate")
    parser.add_argument(
        "--grad_accum_steps", type=int, default=4, help="Gradient accumulation steps"
    )

    # Distillation parameters
    parser.add_argument(
        "--kl_weight", type=float, default=1.0, help="Weight for KL divergence loss"
    )
    parser.add_argument(
        "--ce_weight", type=float, default=0.0, help="Weight for cross-entropy loss"
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Temperature for distillation"
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=512,
        help="Chunk size for memory-efficient loss calculation",
    )

    # Other parameters
    parser.add_argument(
        "--data_path", type=str, default="data/edu_fineweb10B", help="Path to training data"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints/logits_distillation",
        help="Directory for saving checkpoints",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for training")
    parser.add_argument("--use_amp", action="store_true", help="Use automatic mixed precision")
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Data type for mixed precision",
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )

    args = parser.parse_args()

    # Create config from arguments
    config = LogitsDistillationConfig(
        teacher_config_path=args.teacher_config,
        teacher_weights_path=args.teacher_weights,
        student_config_path=args.student_config,
        student_weights_path=args.student_weights,
        data_path=args.data_path,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        gradient_accumulation_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate,
        kl_weight=args.kl_weight,
        ce_weight=args.ce_weight,
        temperature=args.temperature,
        chunk_size=args.chunk_size,
        use_amp=args.use_amp,
        dtype=args.dtype,
    )

    # Initialize trainer
    trainer = DistillationTrainer(config)

    # Load checkpoint if resuming
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Start training
    trainer.train()


if __name__ == "__main__":
    main()

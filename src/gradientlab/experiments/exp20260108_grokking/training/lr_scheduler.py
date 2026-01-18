"""
Custom learning rate scheduler for grokking experiments.

The scheduler has 4 phases:
1. Warmup: Linear ramp from min_lr to max_lr
2. Cosine decay: Decay from max_lr to min_lr
3. Triggered ramp-up: When train_accuracy >= 99%, cosine ramp from min_lr to max_lr for one epoch
4. Constant: Stay at max_lr for remaining steps
"""

import math
from typing import Optional


class GrokScheduler:
    """
    Learning rate scheduler designed to trigger grokking behavior.

    The key feature is a one-time LR ramp-up when training accuracy reaches 99%.
    This sudden increase in learning rate can trigger the grokking transition
    where the model suddenly generalizes after overfitting.
    """

    def __init__(
        self,
        optimizer,
        total_steps: int,
        warmup_steps: int,
        steps_per_epoch: int,
        min_lr: float = 1e-4,
        max_lr: float = 1e-3,
    ):
        """
        Initialize the Grok scheduler.

        Args:
            optimizer: PyTorch optimizer
            total_steps: Total number of training steps
            warmup_steps: Number of warmup steps
            steps_per_epoch: Steps per epoch (for ramp-up duration)
            min_lr: Minimum learning rate
            max_lr: Maximum learning rate
        """
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.steps_per_epoch = steps_per_epoch
        self.min_lr = min_lr
        self.max_lr = max_lr

        self.current_step = 0
        self.rampup_triggered = False
        self.rampup_start_step: Optional[int] = None

        # Pre-compute decay steps (excluding rampup)
        self.decay_steps = total_steps - warmup_steps - steps_per_epoch

    def step(self):
        """Update learning rate. Call after each optimizer step."""
        self.current_step += 1
        self._update_lr()

    def trigger_rampup(self, train_accuracy: float):
        """
        Check if ramp-up should be triggered based on train accuracy.

        Call this at the end of each epoch with the epoch's train accuracy.

        Args:
            train_accuracy: Training accuracy for the epoch (0.0 to 1.0)
        """
        if not self.rampup_triggered and train_accuracy >= 0.99:
            self.rampup_triggered = True
            self.rampup_start_step = self.current_step
            print(
                f"[GrokScheduler] RAMP-UP TRIGGERED at step {self.current_step} "
                f"(train_acc={train_accuracy:.4f})"
            )

    def _update_lr(self):
        """Internal method to compute and set the current learning rate."""
        if self.rampup_triggered:
            steps_since_trigger = self.current_step - self.rampup_start_step  # type: ignore
            if steps_since_trigger < self.steps_per_epoch:
                # Phase 3: Ramp-up (cosine from min_lr to max_lr)
                progress = steps_since_trigger / self.steps_per_epoch
                # Cosine schedule: smooth ramp from min to max
                lr = self.min_lr + (self.max_lr - self.min_lr) * (
                    1 - math.cos(math.pi * progress)
                ) / 2
            else:
                # Phase 4: Constant at max_lr
                lr = self.max_lr
        elif self.current_step < self.warmup_steps:
            # Phase 1: Warmup (linear)
            lr = self.min_lr + (self.max_lr - self.min_lr) * (
                self.current_step / self.warmup_steps
            )
        else:
            # Phase 2: Cosine decay
            steps_since_warmup = self.current_step - self.warmup_steps
            progress = min(steps_since_warmup / self.decay_steps, 1.0)
            # Cosine annealing: decay from max to min
            lr = self.min_lr + (self.max_lr - self.min_lr) * (
                1 + math.cos(math.pi * progress)
            ) / 2

        # Update all parameter groups
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def state_dict(self) -> dict:
        """Return scheduler state for checkpointing."""
        return {
            "current_step": self.current_step,
            "rampup_triggered": self.rampup_triggered,
            "rampup_start_step": self.rampup_start_step,
        }

    def load_state_dict(self, state_dict: dict):
        """Load scheduler state from checkpoint."""
        self.current_step = state_dict["current_step"]
        self.rampup_triggered = state_dict["rampup_triggered"]
        self.rampup_start_step = state_dict["rampup_start_step"]

    def get_last_lr(self) -> list[float]:
        """Get current learning rates (compatible with PyTorch schedulers)."""
        return [group["lr"] for group in self.optimizer.param_groups]

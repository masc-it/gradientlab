"""
Loss evaluation for loss landscape visualization.
"""

from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .directions import DirectionGenerator


class LossEvaluator:
    """
    Evaluate model loss efficiently across landscape grid.

    Supports:
    - Mixed precision evaluation
    - Batch limiting for efficiency
    - Custom loss functions or model's built-in loss
    """

    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        loss_fn: Callable | None = None,
        device: str = "cuda",
        use_mixed_precision: bool = True,
        num_batches: int | None = None,
    ):
        """
        Initialize loss evaluator.

        Args:
            model: The PyTorch model to evaluate.
            dataloader: DataLoader providing evaluation data.
            loss_fn: Optional custom loss function. If None, uses model's built-in loss.
                     Expected signature: loss_fn(model_output, batch) -> scalar tensor.
            device: Device for computation.
            use_mixed_precision: Use mixed precision for efficiency.
            num_batches: Limit evaluation to this many batches (None=all).
        """
        self.model = model
        self.dataloader = dataloader
        self.loss_fn = loss_fn
        self.device = torch.device(device)
        self.use_mixed_precision = use_mixed_precision
        self.num_batches = num_batches

        # Determine dtype for autocast
        if use_mixed_precision and self.device.type == "cuda":
            self._dtype = torch.bfloat16
        else:
            self._dtype = None

    def _load_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Load weights into model (only trainable parameters)."""
        current_state = self.model.state_dict()

        # Only update parameters that are in our state_dict
        for name, param in state_dict.items():
            if name in current_state:
                current_state[name].copy_(param)

    @torch.no_grad()
    def evaluate_at_weights(
        self,
        state_dict: dict[str, torch.Tensor],
    ) -> float:
        """
        Load given weights into model and compute average loss.

        Memory-efficient: processes batches sequentially without gradient tracking.

        Args:
            state_dict: Weights to load into the model.

        Returns:
            Average loss over the evaluation data.
        """
        # Load weights
        self._load_weights(state_dict)
        self.model.eval()

        total_loss = 0.0
        num_samples = 0

        for batch_idx, batch in enumerate(self.dataloader):
            if self.num_batches is not None and batch_idx >= self.num_batches:
                break

            # Move batch to device
            if isinstance(batch, dict):
                batch = {k: v.to(self.device) for k, v in batch.items()}
            elif isinstance(batch, (list, tuple)):
                batch = [
                    v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for v in batch
                ]

            # Get batch size
            if isinstance(batch, dict):
                first_tensor = next(iter(batch.values()))
                batch_size = first_tensor.size(0)
            else:
                batch_size = batch[0].size(0)

            # Compute loss
            with torch.autocast(
                device_type=self.device.type,
                dtype=self._dtype,
                enabled=self.use_mixed_precision and self._dtype is not None,
            ):
                if self.loss_fn is not None:
                    # Custom loss function
                    output = self.model(**batch) if isinstance(batch, dict) else self.model(*batch)
                    loss = self.loss_fn(output, batch)
                else:
                    # Use model's built-in loss
                    if isinstance(batch, dict):
                        # Add labels for loss computation if model expects them
                        output = self.model(**batch)
                    else:
                        output = self.model(*batch)

                    # Extract loss from output
                    if isinstance(output, dict):
                        loss = output.get("loss")
                        if loss is None:
                            raise ValueError(
                                "Model output dict has no 'loss' key. "
                                "Provide a custom loss_fn or ensure model returns {'loss': ...}"
                            )
                    elif hasattr(output, "loss"):
                        loss = output.loss
                    else:
                        raise ValueError(
                            "Cannot extract loss from model output. "
                            "Provide a custom loss_fn."
                        )

            total_loss += loss.item() * batch_size
            num_samples += batch_size

        if num_samples == 0:
            raise ValueError("No samples evaluated. Check your dataloader.")

        return total_loss / num_samples

    def evaluate_grid(
        self,
        direction_generator: DirectionGenerator,
        direction1: dict[str, torch.Tensor],
        direction2: dict[str, torch.Tensor],
        alphas: torch.Tensor,
        betas: torch.Tensor,
        verbose: bool = True,
    ) -> torch.Tensor:
        """
        Evaluate loss at all grid points.

        Args:
            direction_generator: For computing perturbed weights.
            direction1: First direction from direction_generator.
            direction2: Second direction from direction_generator.
            alphas: 1D tensor of alpha values.
            betas: 1D tensor of beta values.
            verbose: Show progress bar.

        Returns:
            2D tensor of shape (len(alphas), len(betas)) with loss values.
        """
        n_alpha = len(alphas)
        n_beta = len(betas)
        total_points = n_alpha * n_beta

        loss_grid = torch.zeros(n_alpha, n_beta)

        iterator = direction_generator.iter_grid_weights(
            alphas, betas, direction1, direction2
        )

        if verbose:
            iterator = tqdm(iterator, total=total_points, desc="Evaluating landscape")

        for i, j, perturbed_weights in iterator:
            loss = self.evaluate_at_weights(perturbed_weights)
            loss_grid[i, j] = loss

        return loss_grid

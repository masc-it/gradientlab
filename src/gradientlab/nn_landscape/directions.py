"""
Direction generation for loss landscape visualization.

Implements filter-normalized random directions from:
"Visualizing the Loss Landscape of Neural Nets" (Li et al., 2018)
"""

from typing import Iterator

import torch
import torch.nn as nn


class DirectionGenerator:
    """
    Generate filter-normalized random directions for loss landscape visualization.

    Filter-wise normalization ensures that random directions have the same scale
    as the corresponding weights, making visualizations comparable across architectures.
    """

    def __init__(
        self,
        model: nn.Module,
        normalize_filter_wise: bool = True,
        seed: int | None = 42,
    ):
        """
        Initialize direction generator.

        Args:
            model: The PyTorch model to generate directions for.
            normalize_filter_wise: If True, apply filter-wise normalization.
            seed: Random seed for reproducibility.
        """
        self.model = model
        self.normalize_filter_wise = normalize_filter_wise
        self.seed = seed

        # Capture reference weights
        self._reference_state = self._capture_reference()

    def _capture_reference(self) -> dict[str, torch.Tensor]:
        """Capture current model weights as reference point."""
        return {
            name: param.detach().clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

    def _generate_random_direction(
        self, generator: torch.Generator
    ) -> dict[str, torch.Tensor]:
        """Generate a random direction tensor for each parameter."""
        direction = {}
        for name, ref_param in self._reference_state.items():
            direction[name] = torch.randn(
                ref_param.shape,
                dtype=ref_param.dtype,
                device=ref_param.device,
                generator=generator,
            )
        return direction

    def _normalize_direction_filter_wise(
        self,
        direction: dict[str, torch.Tensor],
        reference: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Apply filter-wise normalization to direction.

        For each parameter tensor:
        - Conv layers (4D): normalize per output filter (dim 0)
        - Linear layers (2D): normalize per output unit (dim 0)
        - 1D parameters (biases, norms): normalize entire tensor

        This ensures: ||d[i]|| = ||w[i]|| for each filter/unit i.
        """
        normalized = {}

        for name, d in direction.items():
            ref = reference[name]

            if d.ndim >= 2:
                # Multi-dimensional: normalize per "filter" (first dimension)
                # Flatten all dims except the first for norm computation
                d_flat = d.view(d.size(0), -1)
                ref_flat = ref.view(ref.size(0), -1)

                # Compute norms per filter
                d_norms = d_flat.norm(dim=1, keepdim=True) + 1e-10
                ref_norms = ref_flat.norm(dim=1, keepdim=True)

                # Scale factors
                scale = ref_norms / d_norms

                # Apply scaling and reshape back
                d_normalized = (d_flat * scale).view_as(d)
            else:
                # 1D parameters: normalize entire tensor
                d_norm = d.norm() + 1e-10
                ref_norm = ref.norm()
                d_normalized = d * (ref_norm / d_norm)

            normalized[name] = d_normalized

        return normalized

    def _get_device(self) -> torch.device:
        """Get the device of the model parameters."""
        first_param = next(iter(self._reference_state.values()))
        return first_param.device

    def generate_directions(
        self,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """
        Generate two filter-normalized random directions.

        Returns:
            Tuple of (direction1, direction2) as state_dict-like dicts.
        """
        # Get device from model parameters
        device = self._get_device()

        # Create generator on the correct device with seed for reproducibility
        generator = torch.Generator(device=device)
        if self.seed is not None:
            generator.manual_seed(self.seed)

        # Generate two random directions
        d1 = self._generate_random_direction(generator)
        d2 = self._generate_random_direction(generator)

        # Apply filter-wise normalization if enabled
        if self.normalize_filter_wise:
            d1 = self._normalize_direction_filter_wise(d1, self._reference_state)
            d2 = self._normalize_direction_filter_wise(d2, self._reference_state)

        return d1, d2

    def get_perturbed_weights(
        self,
        alpha: float,
        beta: float,
        direction1: dict[str, torch.Tensor],
        direction2: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Compute W_ref + alpha * d1 + beta * d2.

        Args:
            alpha: Coefficient for first direction.
            beta: Coefficient for second direction.
            direction1: First direction (from generate_directions).
            direction2: Second direction (from generate_directions).

        Returns:
            State dict with perturbed weights.
        """
        perturbed = {}
        for name, ref in self._reference_state.items():
            perturbed[name] = ref + alpha * direction1[name] + beta * direction2[name]
        return perturbed

    def iter_grid_weights(
        self,
        alphas: torch.Tensor,
        betas: torch.Tensor,
        direction1: dict[str, torch.Tensor],
        direction2: dict[str, torch.Tensor],
    ) -> Iterator[tuple[int, int, dict[str, torch.Tensor]]]:
        """
        Iterate over all grid points, yielding perturbed weights.

        Args:
            alphas: 1D tensor of alpha values.
            betas: 1D tensor of beta values.
            direction1: First direction.
            direction2: Second direction.

        Yields:
            Tuples of (i, j, perturbed_weights) for each grid point.
        """
        for i, alpha in enumerate(alphas.tolist()):
            for j, beta in enumerate(betas.tolist()):
                yield i, j, self.get_perturbed_weights(alpha, beta, direction1, direction2)

    @property
    def reference_state(self) -> dict[str, torch.Tensor]:
        """Get the reference weights (read-only copy)."""
        return {k: v.clone() for k, v in self._reference_state.items()}

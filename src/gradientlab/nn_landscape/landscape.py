"""
Main orchestration class for loss landscape visualization.
"""

from pathlib import Path
from typing import Callable

import plotly.graph_objects as go
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import LandscapeConfig, VisualizationConfig
from .directions import DirectionGenerator
from .evaluator import LossEvaluator
from .visualization import LandscapeVisualizer


class LossLandscape:
    """
    Main class for computing and visualizing loss landscapes.

    Based on "Visualizing the Loss Landscape of Neural Nets" (Li et al., 2018).

    Example:
        ```python
        from gradientlab.nn_landscape import LossLandscape, LandscapeConfig

        config = LandscapeConfig(grid_size=31, num_batches=50)
        landscape = LossLandscape(model, dataloader, config)
        landscape.compute()
        fig = landscape.plot()
        landscape.save("landscape.html")
        ```
    """

    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        config: LandscapeConfig | None = None,
        loss_fn: Callable | None = None,
        device: str | None = None,
    ):
        """
        Initialize LossLandscape.

        Args:
            model: The PyTorch model to analyze.
            dataloader: DataLoader providing evaluation data.
            config: Landscape configuration. Uses defaults if None.
            loss_fn: Optional custom loss function. If None, uses model's built-in loss.
            device: Device for computation. If None, uses config.device.
        """
        self.model = model
        self.dataloader = dataloader
        self.config = config or LandscapeConfig()
        self.loss_fn = loss_fn
        self.device = device or self.config.device

        # Move model to device
        self.model.to(self.device)

        # Computed results (populated by compute())
        self._loss_grid: torch.Tensor | None = None
        self._alphas: torch.Tensor | None = None
        self._betas: torch.Tensor | None = None
        self._direction1: dict[str, torch.Tensor] | None = None
        self._direction2: dict[str, torch.Tensor] | None = None
        self._reference_weights: dict[str, torch.Tensor] | None = None

    def compute(self, verbose: bool = True) -> "LossLandscape":
        """
        Compute the loss landscape grid.

        Args:
            verbose: Show progress bar.

        Returns:
            Self for method chaining.
        """
        # Generate directions
        direction_generator = DirectionGenerator(
            model=self.model,
            normalize_filter_wise=self.config.normalize_filter_wise,
            seed=self.config.random_seed,
        )

        self._direction1, self._direction2 = direction_generator.generate_directions()
        self._reference_weights = direction_generator.reference_state

        # Create grid
        self._alphas = torch.linspace(
            self.config.alpha_range[0],
            self.config.alpha_range[1],
            self.config.grid_size,
        )
        self._betas = torch.linspace(
            self.config.beta_range[0],
            self.config.beta_range[1],
            self.config.grid_size,
        )

        # Create evaluator
        evaluator = LossEvaluator(
            model=self.model,
            dataloader=self.dataloader,
            loss_fn=self.loss_fn,
            device=self.device,
            use_mixed_precision=self.config.use_mixed_precision,
            num_batches=self.config.num_batches,
        )

        # Evaluate grid
        self._loss_grid = evaluator.evaluate_grid(
            direction_generator=direction_generator,
            direction1=self._direction1,
            direction2=self._direction2,
            alphas=self._alphas,
            betas=self._betas,
            verbose=verbose,
        )

        # Restore reference weights after evaluation
        self._restore_reference_weights()

        return self

    def _restore_reference_weights(self) -> None:
        """Restore model to reference weights after landscape computation."""
        if self._reference_weights is not None:
            current_state = self.model.state_dict()
            for name, param in self._reference_weights.items():
                if name in current_state:
                    current_state[name].copy_(param)

    def _check_computed(self) -> None:
        """Raise error if compute() hasn't been called."""
        if self._loss_grid is None:
            raise RuntimeError("Landscape not computed. Call compute() first.")

    def plot(
        self,
        vis_config: VisualizationConfig | None = None,
    ) -> go.Figure:
        """
        Generate visualization of computed landscape.

        Args:
            vis_config: Visualization configuration. Uses defaults if None.

        Returns:
            Plotly Figure object.
        """
        self._check_computed()

        config = vis_config or VisualizationConfig()
        visualizer = LandscapeVisualizer(config)

        if config.plot_type == "surface":
            return visualizer.plot_surface_3d(
                self._loss_grid, self._alphas, self._betas
            )
        elif config.plot_type == "contour":
            return visualizer.plot_contour(
                self._loss_grid, self._alphas, self._betas
            )
        elif config.plot_type == "1d":
            return visualizer.plot_1d_slice(
                self._loss_grid, self._alphas, self._betas
            )
        else:  # "both"
            return visualizer.plot_combined(
                self._loss_grid, self._alphas, self._betas
            )

    def save(
        self,
        path: str | Path,
        vis_config: VisualizationConfig | None = None,
        include_data: bool = True,
    ) -> None:
        """
        Save visualization and optionally the raw data.

        Args:
            path: Output path. If ends with .html, saves interactive plot.
                  Otherwise, saves as image (requires kaleido).
            vis_config: Visualization configuration for the plot.
            include_data: If True, also saves raw data as .pt file.
        """
        self._check_computed()

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Generate and save plot
        fig = self.plot(vis_config)
        config = vis_config or VisualizationConfig()
        visualizer = LandscapeVisualizer(config)

        if path.suffix == ".html":
            visualizer.save_html(fig, path)
        else:
            visualizer.save_image(fig, path)

        # Save raw data
        if include_data:
            data_path = path.with_suffix(".pt")
            self.save_data(data_path)

    def save_data(self, path: str | Path) -> None:
        """
        Save raw landscape data for later analysis.

        Args:
            path: Output .pt file path.
        """
        self._check_computed()

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "loss_grid": self._loss_grid,
                "alphas": self._alphas,
                "betas": self._betas,
                "direction1": self._direction1,
                "direction2": self._direction2,
                "reference_weights": self._reference_weights,
                "config": self.config.model_dump(),
            },
            path,
        )

    @classmethod
    def load_data(cls, path: str | Path) -> dict:
        """
        Load previously saved landscape data.

        Args:
            path: Path to .pt file.

        Returns:
            Dictionary with loss_grid, alphas, betas, directions, etc.
        """
        return torch.load(path, weights_only=False)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        model: nn.Module,
        dataloader: DataLoader,
        config: LandscapeConfig | None = None,
        loss_fn: Callable | None = None,
        device: str | None = None,
        strict: bool = True,
    ) -> "LossLandscape":
        """
        Create LossLandscape from a saved checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file (.pt or .pth).
            model: Model instance (architecture must match checkpoint).
            dataloader: DataLoader for evaluation.
            config: Landscape configuration.
            loss_fn: Optional custom loss function.
            device: Device for computation.
            strict: Whether to strictly enforce state_dict key matching.

        Returns:
            LossLandscape instance ready for compute().
        """
        checkpoint = torch.load(checkpoint_path, weights_only=True)

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif "model" in checkpoint:
                state_dict = checkpoint["model"]
            else:
                # Assume it's a raw state dict
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict, strict=strict)

        return cls(
            model=model,
            dataloader=dataloader,
            config=config,
            loss_fn=loss_fn,
            device=device,
        )

    @property
    def loss_grid(self) -> torch.Tensor:
        """Get computed loss grid."""
        self._check_computed()
        return self._loss_grid

    @property
    def alphas(self) -> torch.Tensor:
        """Get alpha values."""
        self._check_computed()
        return self._alphas

    @property
    def betas(self) -> torch.Tensor:
        """Get beta values."""
        self._check_computed()
        return self._betas

    @property
    def center_loss(self) -> float:
        """Get loss at the reference point (alpha=0, beta=0)."""
        self._check_computed()
        center_idx = self.config.grid_size // 2
        return self._loss_grid[center_idx, center_idx].item()

    @property
    def min_loss(self) -> float:
        """Get minimum loss in the landscape."""
        self._check_computed()
        return self._loss_grid.min().item()

    @property
    def max_loss(self) -> float:
        """Get maximum loss in the landscape."""
        self._check_computed()
        return self._loss_grid.max().item()

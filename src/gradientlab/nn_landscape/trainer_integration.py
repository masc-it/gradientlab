"""
Training loop integration for capturing landscape snapshots during training.
"""

from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import LandscapeConfig, VisualizationConfig
from .landscape import LossLandscape


class LandscapeCallback:
    """
    Callback for capturing landscape snapshots during training.

    Can be integrated into any training loop.

    Example:
        ```python
        callback = LandscapeCallback(
            dataloader=val_loader,
            capture_every_n_epochs=100,
            output_dir=exp_dir / "landscapes"
        )

        for epoch in range(num_epochs):
            train_one_epoch(...)
            landscape = callback.on_epoch_end(model, epoch)
            if landscape:
                print(f"Captured landscape at epoch {epoch}")

        # Create animation
        callback.create_animation("training_evolution.gif")
        ```
    """

    def __init__(
        self,
        dataloader: DataLoader,
        config: LandscapeConfig | None = None,
        capture_every_n_epochs: int = 10,
        output_dir: Path | str | None = None,
        loss_fn: Callable | None = None,
        device: str = "cuda",
        auto_save: bool = True,
    ):
        """
        Initialize landscape callback.

        Args:
            dataloader: DataLoader for landscape evaluation.
            config: Landscape configuration. Uses defaults if None.
            capture_every_n_epochs: Capture landscape every N epochs.
            output_dir: Directory to save landscape plots and data.
            loss_fn: Optional custom loss function.
            device: Device for computation.
            auto_save: Automatically save plots when capturing.
        """
        self.dataloader = dataloader
        self.config = config or LandscapeConfig()
        self.capture_every_n_epochs = capture_every_n_epochs
        self.output_dir = Path(output_dir) if output_dir else None
        self.loss_fn = loss_fn
        self.device = device
        self.auto_save = auto_save

        # Create output directory
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        # Store snapshots metadata
        self._snapshots: list[dict] = []

    def on_epoch_end(
        self,
        model: nn.Module,
        epoch: int,
        step: int | None = None,
        verbose: bool = False,
        **metadata,
    ) -> LossLandscape | None:
        """
        Call at end of each epoch to potentially capture landscape.

        Args:
            model: Current model state.
            epoch: Current epoch number.
            step: Optional global step number.
            verbose: Show progress during landscape computation.
            **metadata: Additional metadata to store with snapshot.

        Returns:
            LossLandscape if captured at this epoch, None otherwise.
        """
        if epoch % self.capture_every_n_epochs != 0:
            return None

        return self.capture_snapshot(
            model=model,
            epoch=epoch,
            step=step,
            verbose=verbose,
            **metadata,
        )

    def capture_snapshot(
        self,
        model: nn.Module,
        epoch: int,
        step: int | None = None,
        verbose: bool = True,
        **metadata,
    ) -> LossLandscape:
        """
        Manually capture a landscape snapshot.

        Args:
            model: Current model state.
            epoch: Current epoch number.
            step: Optional global step number.
            verbose: Show progress during computation.
            **metadata: Additional metadata to store.

        Returns:
            Computed LossLandscape.
        """
        # Compute landscape
        landscape = LossLandscape(
            model=model,
            dataloader=self.dataloader,
            config=self.config,
            loss_fn=self.loss_fn,
            device=self.device,
        )
        landscape.compute(verbose=verbose)

        # Store metadata
        snapshot_info = {
            "epoch": epoch,
            "step": step,
            "center_loss": landscape.center_loss,
            "min_loss": landscape.min_loss,
            "max_loss": landscape.max_loss,
            **metadata,
        }
        self._snapshots.append(snapshot_info)

        # Auto-save if configured
        if self.auto_save and self.output_dir:
            filename = f"landscape_epoch_{epoch:06d}"
            landscape.save(
                self.output_dir / f"{filename}.html",
                include_data=True,
            )

        return landscape

    def get_snapshots(self) -> list[dict]:
        """Get metadata for all captured snapshots."""
        return self._snapshots.copy()

    def create_animation(
        self,
        output_path: Path | str,
        fps: int = 2,
        vis_config: VisualizationConfig | None = None,
    ) -> None:
        """
        Create animation showing landscape evolution during training.

        This creates an HTML file with a slider to view landscapes at different epochs.

        Args:
            output_path: Output file path (.html).
            fps: Frames per second (used if exporting to video format).
            vis_config: Visualization configuration.
        """
        if not self.output_dir:
            raise ValueError("output_dir must be set to create animation")

        if not self._snapshots:
            raise ValueError("No snapshots captured. Call capture_snapshot() first.")

        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import numpy as np

        config = vis_config or VisualizationConfig()

        # Load all landscape data
        frames = []
        epochs = []

        for snapshot in self._snapshots:
            epoch = snapshot["epoch"]
            data_path = self.output_dir / f"landscape_epoch_{epoch:06d}.pt"

            if not data_path.exists():
                continue

            data = LossLandscape.load_data(data_path)
            loss_grid = data["loss_grid"].numpy()
            alphas = data["alphas"].numpy()
            betas = data["betas"].numpy()

            if config.log_scale:
                loss_grid = np.log10(loss_grid + 1e-10)

            frames.append({
                "loss_grid": loss_grid,
                "alphas": alphas,
                "betas": betas,
                "epoch": epoch,
            })
            epochs.append(epoch)

        if not frames:
            raise ValueError("No landscape data files found in output_dir")

        # Create figure with slider
        alpha_mesh, beta_mesh = np.meshgrid(
            frames[0]["alphas"], frames[0]["betas"], indexing="ij"
        )

        fig = go.Figure()

        # Add first frame
        fig.add_trace(
            go.Surface(
                x=alpha_mesh,
                y=beta_mesh,
                z=frames[0]["loss_grid"],
                colorscale=config.colorscale,
                showscale=True,
                colorbar=dict(
                    title="Log₁₀(Loss)" if config.log_scale else "Loss",
                ),
            )
        )

        # Create frames for animation
        plotly_frames = []
        for frame_data in frames:
            plotly_frames.append(
                go.Frame(
                    data=[
                        go.Surface(
                            x=alpha_mesh,
                            y=beta_mesh,
                            z=frame_data["loss_grid"],
                            colorscale=config.colorscale,
                        )
                    ],
                    name=str(frame_data["epoch"]),
                )
            )

        fig.frames = plotly_frames

        # Create slider steps
        steps = []
        for epoch in epochs:
            steps.append(
                dict(
                    args=[
                        [str(epoch)],
                        dict(
                            frame=dict(duration=1000 // fps, redraw=True),
                            mode="immediate",
                            transition=dict(duration=300),
                        ),
                    ],
                    label=str(epoch),
                    method="animate",
                )
            )

        sliders = [
            dict(
                active=0,
                currentvalue=dict(
                    prefix="Epoch: ",
                    visible=True,
                    xanchor="center",
                ),
                pad=dict(t=50, b=10),
                steps=steps,
            )
        ]

        z_label = "Log₁₀(Loss)" if config.log_scale else "Loss"

        fig.update_layout(
            title=dict(text="Loss Landscape Evolution", x=0.5, xanchor="center"),
            scene=dict(
                xaxis_title="α (Direction 1)",
                yaxis_title="β (Direction 2)",
                zaxis_title=z_label,
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
            ),
            width=config.width,
            height=config.height,
            sliders=sliders,
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    y=0,
                    x=0.1,
                    xanchor="right",
                    yanchor="top",
                    pad=dict(t=0, r=10),
                    buttons=[
                        dict(
                            label="▶ Play",
                            method="animate",
                            args=[
                                None,
                                dict(
                                    frame=dict(duration=1000 // fps, redraw=True),
                                    fromcurrent=True,
                                    transition=dict(duration=300),
                                ),
                            ],
                        ),
                        dict(
                            label="⏸ Pause",
                            method="animate",
                            args=[
                                [None],
                                dict(
                                    frame=dict(duration=0, redraw=False),
                                    mode="immediate",
                                    transition=dict(duration=0),
                                ),
                            ],
                        ),
                    ],
                )
            ],
        )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path), include_plotlyjs=True)

"""
Visualization utilities for loss landscape using Plotly.

Provides interactive 3D surface plots, contour plots, and 1D slices.
"""

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch

from .config import VisualizationConfig


class LandscapeVisualizer:
    """Create interactive visualizations of loss landscapes using Plotly."""

    def __init__(self, config: VisualizationConfig | None = None):
        """
        Initialize visualizer.

        Args:
            config: Visualization configuration. Uses defaults if None.
        """
        self.config = config or VisualizationConfig()

    def _prepare_data(
        self,
        loss_grid: torch.Tensor,
        alphas: torch.Tensor,
        betas: torch.Tensor,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for plotting, applying log scale if configured."""
        alphas_np = alphas.numpy()
        betas_np = betas.numpy()
        loss_np = loss_grid.numpy()

        if self.config.log_scale:
            # Apply log scale, handling zeros
            loss_np = np.log10(loss_np + 1e-10)

        return alphas_np, betas_np, loss_np

    def plot_surface_3d(
        self,
        loss_grid: torch.Tensor,
        alphas: torch.Tensor,
        betas: torch.Tensor,
        title: str = "Loss Landscape",
    ) -> go.Figure:
        """
        Create interactive 3D surface plot.

        Args:
            loss_grid: 2D tensor of loss values (shape: n_alpha x n_beta).
            alphas: 1D tensor of alpha values.
            betas: 1D tensor of beta values.
            title: Plot title.

        Returns:
            Plotly Figure object.
        """
        alphas_np, betas_np, loss_np = self._prepare_data(loss_grid, alphas, betas)

        # Create meshgrid for surface
        alpha_mesh, beta_mesh = np.meshgrid(alphas_np, betas_np, indexing="ij")

        fig = go.Figure(
            data=[
                go.Surface(
                    x=alpha_mesh,
                    y=beta_mesh,
                    z=loss_np,
                    colorscale=self.config.colorscale,
                    colorbar=dict(
                        title=dict(
                            text="Log₁₀(Loss)" if self.config.log_scale else "Loss",
                            side="right",
                        ),
                    ),
                )
            ]
        )

        z_label = "Log₁₀(Loss)" if self.config.log_scale else "Loss"

        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor="center"),
            scene=dict(
                xaxis_title="α (Direction 1)",
                yaxis_title="β (Direction 2)",
                zaxis_title=z_label,
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2),
                ),
            ),
            width=self.config.width,
            height=self.config.height,
            margin=dict(l=0, r=0, t=40, b=0),
        )

        return fig

    def plot_contour(
        self,
        loss_grid: torch.Tensor,
        alphas: torch.Tensor,
        betas: torch.Tensor,
        title: str = "Loss Landscape Contours",
    ) -> go.Figure:
        """
        Create 2D contour plot.

        Args:
            loss_grid: 2D tensor of loss values.
            alphas: 1D tensor of alpha values.
            betas: 1D tensor of beta values.
            title: Plot title.

        Returns:
            Plotly Figure object.
        """
        alphas_np, betas_np, loss_np = self._prepare_data(loss_grid, alphas, betas)

        fig = go.Figure(
            data=[
                go.Contour(
                    x=alphas_np,
                    y=betas_np,
                    z=loss_np.T,  # Transpose for correct orientation
                    colorscale=self.config.colorscale,
                    ncontours=self.config.num_contours,
                    colorbar=dict(
                        title=dict(
                            text="Log₁₀(Loss)" if self.config.log_scale else "Loss",
                            side="right",
                        ),
                    ),
                    contours=dict(
                        showlabels=True,
                        labelfont=dict(size=10, color="white"),
                    ),
                )
            ]
        )

        # Mark the center (reference weights)
        fig.add_trace(
            go.Scatter(
                x=[0],
                y=[0],
                mode="markers",
                marker=dict(size=12, color="red", symbol="x"),
                name="Reference",
                showlegend=True,
            )
        )

        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor="center"),
            xaxis_title="α (Direction 1)",
            yaxis_title="β (Direction 2)",
            width=self.config.width,
            height=self.config.height,
        )

        return fig

    def plot_1d_slice(
        self,
        loss_grid: torch.Tensor,
        alphas: torch.Tensor,
        betas: torch.Tensor,
        direction: str = "alpha",
        title: str = "1D Loss Slice",
    ) -> go.Figure:
        """
        Create 1D slice through the landscape.

        Args:
            loss_grid: 2D tensor of loss values.
            alphas: 1D tensor of alpha values.
            betas: 1D tensor of beta values.
            direction: Which slice to take: "alpha", "beta", or "diagonal".
            title: Plot title.

        Returns:
            Plotly Figure object.
        """
        alphas_np, betas_np, loss_np = self._prepare_data(loss_grid, alphas, betas)

        # Find center indices
        center_alpha = len(alphas_np) // 2
        center_beta = len(betas_np) // 2

        if direction == "alpha":
            # Slice along alpha at beta=0
            x_values = alphas_np
            y_values = loss_np[:, center_beta]
            x_label = "α (Direction 1)"
        elif direction == "beta":
            # Slice along beta at alpha=0
            x_values = betas_np
            y_values = loss_np[center_alpha, :]
            x_label = "β (Direction 2)"
        elif direction == "diagonal":
            # Diagonal slice (alpha = beta)
            min_len = min(len(alphas_np), len(betas_np))
            x_values = alphas_np[:min_len]
            y_values = np.array([loss_np[i, i] for i in range(min_len)])
            x_label = "α = β (Diagonal)"
        else:
            raise ValueError(f"Unknown direction: {direction}. Use 'alpha', 'beta', or 'diagonal'.")

        fig = go.Figure(
            data=[
                go.Scatter(
                    x=x_values,
                    y=y_values,
                    mode="lines+markers",
                    line=dict(width=2),
                    marker=dict(size=4),
                )
            ]
        )

        # Mark the center point
        if direction == "alpha":
            ref_x, ref_y = 0, loss_np[center_alpha, center_beta]
        elif direction == "beta":
            ref_x, ref_y = 0, loss_np[center_alpha, center_beta]
        else:
            ref_x, ref_y = 0, loss_np[center_alpha, center_beta]

        fig.add_trace(
            go.Scatter(
                x=[ref_x],
                y=[ref_y],
                mode="markers",
                marker=dict(size=12, color="red", symbol="x"),
                name="Reference",
            )
        )

        y_label = "Log₁₀(Loss)" if self.config.log_scale else "Loss"

        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor="center"),
            xaxis_title=x_label,
            yaxis_title=y_label,
            width=self.config.width,
            height=self.config.height // 2,
            showlegend=True,
        )

        return fig

    def plot_combined(
        self,
        loss_grid: torch.Tensor,
        alphas: torch.Tensor,
        betas: torch.Tensor,
        title: str = "Loss Landscape",
    ) -> go.Figure:
        """
        Create combined figure with 3D surface and contour side-by-side.

        Args:
            loss_grid: 2D tensor of loss values.
            alphas: 1D tensor of alpha values.
            betas: 1D tensor of beta values.
            title: Overall plot title.

        Returns:
            Plotly Figure with subplots.
        """
        alphas_np, betas_np, loss_np = self._prepare_data(loss_grid, alphas, betas)

        # Create meshgrid
        alpha_mesh, beta_mesh = np.meshgrid(alphas_np, betas_np, indexing="ij")

        fig = make_subplots(
            rows=1,
            cols=2,
            specs=[[{"type": "surface"}, {"type": "xy"}]],
            subplot_titles=["3D Surface", "Contour View"],
            horizontal_spacing=0.1,
        )

        # 3D Surface
        fig.add_trace(
            go.Surface(
                x=alpha_mesh,
                y=beta_mesh,
                z=loss_np,
                colorscale=self.config.colorscale,
                showscale=False,
            ),
            row=1,
            col=1,
        )

        # Contour
        fig.add_trace(
            go.Contour(
                x=alphas_np,
                y=betas_np,
                z=loss_np.T,
                colorscale=self.config.colorscale,
                ncontours=self.config.num_contours,
                colorbar=dict(
                    title="Log₁₀(Loss)" if self.config.log_scale else "Loss",
                    x=1.02,
                ),
            ),
            row=1,
            col=2,
        )

        # Reference point marker on contour
        fig.add_trace(
            go.Scatter(
                x=[0],
                y=[0],
                mode="markers",
                marker=dict(size=12, color="red", symbol="x"),
                name="Reference",
                showlegend=False,
            ),
            row=1,
            col=2,
        )

        z_label = "Log₁₀(Loss)" if self.config.log_scale else "Loss"

        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor="center"),
            width=self.config.width * 2,
            height=self.config.height,
            scene=dict(
                xaxis_title="α",
                yaxis_title="β",
                zaxis_title=z_label,
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
            ),
        )

        fig.update_xaxes(title_text="α (Direction 1)", row=1, col=2)
        fig.update_yaxes(title_text="β (Direction 2)", row=1, col=2)

        return fig

    def save_html(self, fig: go.Figure, path: str | Path) -> None:
        """
        Save figure as standalone HTML file.

        Args:
            fig: Plotly Figure to save.
            path: Output file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(path), include_plotlyjs=True)

    def save_image(self, fig: go.Figure, path: str | Path, scale: int = 2) -> None:
        """
        Save figure as static image (PNG, SVG, etc.).

        Requires kaleido package: pip install kaleido

        Args:
            fig: Plotly Figure to save.
            path: Output file path (extension determines format).
            scale: Resolution multiplier.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_image(str(path), scale=scale)

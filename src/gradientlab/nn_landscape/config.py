"""
Configuration classes for loss landscape visualization.
"""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class LandscapeConfig(BaseModel):
    """Configuration for loss landscape computation."""

    # Grid parameters
    grid_size: int = Field(default=51, description="Number of points along each direction")
    alpha_range: tuple[float, float] = Field(
        default=(-1.0, 1.0), description="Range for first direction"
    )
    beta_range: tuple[float, float] = Field(
        default=(-1.0, 1.0), description="Range for second direction"
    )

    # Direction generation
    normalize_filter_wise: bool = Field(
        default=True, description="Apply filter-wise normalization to directions"
    )
    random_seed: int | None = Field(default=42, description="Seed for reproducibility")

    # Evaluation
    num_batches: int | None = Field(
        default=None, description="Limit batches for efficiency (None=all)"
    )
    use_mixed_precision: bool = Field(
        default=True, description="Use mixed precision during evaluation"
    )
    device: str = Field(default="cuda", description="Device for computation")

    # Output
    output_dir: Path | None = Field(default=None, description="Directory to save results")
    save_directions: bool = Field(
        default=True, description="Save direction tensors for reproducibility"
    )

    class Config:
        arbitrary_types_allowed = True


class VisualizationConfig(BaseModel):
    """Configuration for loss landscape visualization."""

    plot_type: Literal["surface", "contour", "both", "1d"] = Field(
        default="both", description="Type of plot to generate"
    )
    colorscale: str = Field(default="Viridis", description="Plotly colorscale name")
    log_scale: bool = Field(default=True, description="Use log scale for loss values")

    # 3D surface options
    width: int = Field(default=800, description="Plot width in pixels")
    height: int = Field(default=600, description="Plot height in pixels")

    # Contour options
    num_contours: int = Field(default=30, description="Number of contour levels")

    # Output
    save_path: Path | None = Field(default=None, description="Path to save the plot")

    class Config:
        arbitrary_types_allowed = True

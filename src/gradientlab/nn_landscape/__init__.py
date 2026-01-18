"""
Loss Landscape Visualization for PyTorch Models.

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

For training integration:
    ```python
    from gradientlab.nn_landscape import LandscapeCallback

    callback = LandscapeCallback(
        dataloader=val_loader,
        capture_every_n_epochs=100,
        output_dir=exp_dir / "landscapes"
    )

    for epoch in range(num_epochs):
        train_one_epoch(...)
        callback.on_epoch_end(model, epoch)
    ```
"""

from .config import LandscapeConfig, VisualizationConfig
from .directions import DirectionGenerator
from .evaluator import LossEvaluator
from .landscape import LossLandscape
from .trainer_integration import LandscapeCallback
from .visualization import LandscapeVisualizer

__all__ = [
    "LossLandscape",
    "LandscapeConfig",
    "VisualizationConfig",
    "DirectionGenerator",
    "LossEvaluator",
    "LandscapeVisualizer",
    "LandscapeCallback",
]

"""
Experiment configuration for grokking experiments.
"""

import sys
from pathlib import Path
from typing import Optional

from pydantic import BaseModel


# Platform-specific dataset path
if sys.platform == "darwin":
    ds_name = str(Path(__file__).parent / "data" / "min_coin_change")
else:
    ds_name = "/media/mascit/Lexar/datasets/min_coin_change/"

# Create experiment directory
exp_dir = Path(__file__).parent / "data"
exp_dir.mkdir(exist_ok=True, parents=True)


class ExpConfig(BaseModel):
    """
    Configuration for grokking experiment.

    Key parameters:
    - eval_every_n_epochs: Controls evaluation frequency.
        * 1 = evaluate every epoch (slower but more detailed monitoring)
        * 10 = evaluate every 10th epoch (faster training, less overhead)
        * 50 = evaluate every 50th epoch (recommended for long training runs)
        Note: Evaluation always runs on the last epoch regardless of this setting.
    """

    # Data
    ds_name: str = ds_name
    batch_size: int = 128
    num_workers: int = 4

    # Training
    num_epochs: int = 20000
    device: str = "cuda"

    # Optimizer
    min_lr: float = 1e-4
    max_lr: float = 1e-3
    weight_decay: float = 1.0
    warmup_ratio: float = 0.05

    # Logging & Checkpointing
    project_name: str = "grokking"
    exp_name: str = Path(__file__).parent.stem  # "exp20260108_grokking"
    exp_dir: Path = exp_dir
    log_steps: int = 10  # Log training metrics every N steps
    save_steps: int = 500  # Periodic sampling (currently unused)
    eval_every_n_epochs: int = 200  # Run full eval every N epochs (includes generation + autoregressive accuracy)

    # Resume
    resume_from: Optional[str] = None

    # Stopping criterion
    target_eval_accuracy: float = 0.995  # Stop at 99.5%

    class Config:
        arbitrary_types_allowed = True

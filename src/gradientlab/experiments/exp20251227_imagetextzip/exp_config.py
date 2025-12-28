from pathlib import Path
import sys
from typing import Optional
from pydantic import BaseModel

ds_name = (
    "/Volumes/Lexar/datasets/imagetextzip/"
    if sys.platform == "darwin"
    else "/media/mascit/Lexar/datasets/imagetextzip/"
)

exp_dir = Path(__file__).parent / "data"
exp_dir.mkdir(exist_ok=True)


class ExpConfig(BaseModel):
    batch_size: int = 32
    device: str = "cuda"

    ds_name: str = ds_name
    project_name: str = "imagetextzip"
    exp_name: str = Path(__file__).parent.stem

    exp_dir: Path = exp_dir
    num_epochs: int = 1500
    min_lr: float = 5e-5
    max_lr: float = 2e-4
    warmup_ratio: float = 0.1
    num_workers: int = 4
    weight_decay: float = 1e-2
    resume_from: Optional[str] = "src/gradientlab/experiments/exp20251227_imagetextzip/data/model/"
    log_steps: int = 10
    save_steps: int = 100

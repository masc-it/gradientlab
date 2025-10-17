from pathlib import Path
import sys
from typing import Optional
from pydantic import BaseModel

ds_name = (
    "/Volumes/Lexar/datasets/fineweb-2-ita/processed_tok"
    if sys.platform == "darwin"
    else "/media/mascit/Lexar/datasets/fineweb-2-ita/processed_tok"
)

exp_dir = Path(__file__).parent / "data"
exp_dir.mkdir(exist_ok=True)


class ExpConfig(BaseModel):
    batch_size: int = 32
    ds_name: str = ds_name
    exp_name: str = Path(__file__).parent.stem
    exp_dir: Path = exp_dir
    num_epochs: int = 1
    min_lr: float = 4e-5
    max_lr: float = 6e-4
    warmup_ratio: float = 0.03
    num_workers: int = 4
    device: str = "mps"
    weight_decay = 1e-3
    resume_from: Optional[str] = None

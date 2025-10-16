from pathlib import Path
from pydantic import BaseModel


class ExpConfig(BaseModel):
    batch_size: int = 32
    ds_name: str = ""
    exp_name: str = Path(__file__).parent.stem
    num_epochs: int = 1
    min_lr: float = 4e-5
    max_lr: float = 6e-4
    warmup_ratio: float = 0.03

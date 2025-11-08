from pathlib import Path
import sys
from typing import Optional
from pydantic import BaseModel

ds_name = "InstaDeepAI/nucleotide_transformer_downstream_tasks"

exp_dir = Path(__file__).parent / "data"
exp_dir.mkdir(exist_ok=True)


class ExpConfig(BaseModel):
    batch_size: int = 128
    device: str = "cuda"

    ds_name: str = ds_name
    project_name: str = "vlm_pretraining_kda_in1k"
    exp_name: str = Path(__file__).parent.stem
    
    exp_dir: Path = exp_dir
    num_epochs: int = 50
    min_lr: float = 4e-5
    max_lr: float = 6e-4
    warmup_ratio: float = 0.1
    num_workers: int = 4
    weight_decay: float = 1e-2
    resume_from: Optional[str] = None
    log_steps: int = 10
    save_steps: int = 1000

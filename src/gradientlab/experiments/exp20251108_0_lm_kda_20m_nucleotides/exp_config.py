from pathlib import Path
from typing import Literal, Optional
from pydantic import BaseModel

ds_name = "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised"

exp_dir = Path(__file__).parent / "data"
exp_dir.mkdir(exist_ok=True)


class ExpConfig(BaseModel):
    batch_size: int = 32
    device: str = "cuda"

    task: Literal["pretraining", "600_tasks", "all_tasks"] = "pretraining"
    ds_name: str = ds_name
    project_name: str = "lm_pretraining_kda_nucleotides"
    exp_name: str = Path(__file__).parent.stem
    
    exp_dir: Path = exp_dir
    num_epochs: int = 10
    min_lr: float = 4e-5
    max_lr: float = 6e-4
    warmup_ratio: float = 0.1
    num_workers: int = 4
    weight_decay: float = 1e-2
    resume_from: Optional[str] = None
    log_steps: int = 10
    save_steps: int = 1000
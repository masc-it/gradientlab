from pathlib import Path
from typing import Literal, Optional
from pydantic import BaseModel

ds_name = "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised"

exp_dir = Path(__file__).parent / "data"
exp_dir.mkdir(exist_ok=True)


class ExpConfig(BaseModel):
    batch_size: int = 32
    device: str = "cuda"

    task: Literal["pretraining", "600_tasks", "all_tasks"] = "all_tasks"
    ds_name: str = ds_name
    project_name: str = "lm_kda_nucleotides_600_tasks"
    exp_name: str = Path(__file__).parent.stem
    
    exp_dir: Path = exp_dir
    num_epochs: int = 2
    min_lr: float = 1e-5
    max_lr: float = 1e-4
    warmup_ratio: float = 0.05
    num_workers: int = 4
    weight_decay: float = 1e-2
    resume_from: Optional[str] = "/media/mascit/data/Projects/python/gradientlab/src/gradientlab/experiments/exp20251108_0_lm_kda_20m_nucleotides/data/model"
    log_steps: int = 10
    save_steps: int = 1000

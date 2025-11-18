from pathlib import Path
from typing import Literal, Optional
from pydantic import BaseModel

ds_name = "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised"

exp_dir = Path(__file__).parent / "data"
exp_dir.mkdir(exist_ok=True)


class ExpConfig(BaseModel):
    batch_size: int = 64
    device: str = "cuda"

    task: Literal["pretraining", "600_tasks", "all_tasks"] = "pretraining"
    ds_name: str = ds_name
    project_name: str = "lm_pretraining_vanilla_kda_nucleotides"
    exp_name: str = Path(__file__).parent.stem
    
    exp_dir: Path = exp_dir
    num_epochs: int = 165
    min_lr: float = 1e-5
    max_lr: float = 2e-5
    warmup_ratio: float = 0.1
    num_workers: int = 4
    weight_decay: float = 1e-2
    resume_from: Optional[str] = "/media/mascit/data/Projects/python/gradientlab/src/gradientlab/experiments/exp20251113_0_lm_vanilla_kda_20m_nucleotides/data/model_150"
    log_steps: int = 10
    save_steps: int = 1000
from pathlib import Path
from typing import Literal, Optional
from pydantic import BaseModel

#ds_name = "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised"
ds_name = "mascIT/InstaDeepAI_human_reference_genome"

exp_dir = Path(__file__).parent / "data"
exp_dir.mkdir(exist_ok=True)


class ExpConfig(BaseModel):
    batch_size: int = 64
    device: str = "cuda"

    task: Literal["pretraining", "600_tasks", "all_tasks", "pretraining_human"] = "pretraining_human"
    ds_name: str = ds_name
    project_name: str = "lm_pretraining_vanilla_nucleotides"
    exp_name: str = Path(__file__).parent.stem
    
    exp_dir: Path = exp_dir
    num_epochs: int = 12
    min_lr: float = 5e-5
    max_lr: float = 2e-4
    warmup_ratio: float = 0.1
    num_workers: int = 5
    weight_decay: float = 1e-2
    resume_from: Optional[str] = None
    log_steps: int = 10
    save_steps: int = 1000
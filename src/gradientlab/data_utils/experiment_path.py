from pathlib import Path


def get_ckpt_path_by_exp_name(exp_name: str):
    return Path(__file__).parent.parent / "experiments" / exp_name / "data" / "model"
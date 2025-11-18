import json
from pathlib import Path
import inspect


def hf_add_custom_model_metadata(hf_save_dir: Path, exp_name: str, model_cls, config_cls):

    exp_name = Path(inspect.getfile(model_cls)).parent.parent.stem
    model_name = model_cls.__name__
    custom_model_str = f"""
from gradientlab.experiments.{exp_name}.modeling.model import {model_name}
__all__ = ["{model_name}"]
""".strip()
    (hf_save_dir / "modeling_custom.py").write_text(custom_model_str)
    
    config_name = config_cls.__name__
    custom_cfg_str = f"""
from gradientlab.experiments.{exp_name}.modeling.model_cfg import {config_name}
__all__ = ["{config_name}"]
""".strip()
    (hf_save_dir / "configuration_custom.py").write_text(custom_cfg_str)

    
    hf_cfg = hf_save_dir / "config.json"

    obj = json.loads(hf_cfg.read_text())

    obj["auto_map"] = {
        "AutoConfig": f"configuration_custom.{config_name}",
        "AutoModelForCausalLM": f"modeling_custom.{model_name}"
    }

    hf_cfg.write_text(json.dumps(obj, indent=2, ensure_ascii=False))

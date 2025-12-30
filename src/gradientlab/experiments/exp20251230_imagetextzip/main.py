import json

from gradientlab.experiments.exp20251230_imagetextzip.exp_config import (
    ExpConfig,
)
from gradientlab.experiments.exp20251230_imagetextzip.modeling.factory import (
    ModelFactory,
)
from gradientlab.experiments.exp20251230_imagetextzip.trainer import (
    Trainer,
)
from gradientlab.logging_utils.log_model_params import pretty_print_model


def main():
    print("=== START TRAINING ===")
    exp_cfg = ExpConfig()
    model, tokenizer, model_cfg = ModelFactory.build_vit_5m(exp_cfg.resume_from)

    print(json.dumps(model_cfg.model_dump(), indent=2) + "\n")
    pretty_print_model(model)

    print("\n" + exp_cfg.model_dump_json(indent=2))

    trainer = Trainer(model, tokenizer, model_cfg, exp_cfg)
    try:
        trainer.train()
    except KeyboardInterrupt as e:
        # trainer._save_state()
        pass


if __name__ == "__main__":
    main()

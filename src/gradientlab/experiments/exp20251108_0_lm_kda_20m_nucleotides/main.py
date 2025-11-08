import json
import sys

import torch
from transformers import GenerationConfig
from gradientlab.experiments.exp20251108_0_lm_kda_20m_nucleotides.exp_config import (
    ExpConfig,
)
from gradientlab.experiments.exp20251108_0_lm_kda_20m_nucleotides.modeling.factory import (
    GPTFactory,
)
from gradientlab.experiments.exp20251108_0_lm_kda_20m_nucleotides.trainer import (
    Trainer,
)
from gradientlab.logging_utils.log_model_params import pretty_print_model

DEBUG_GEN = False

def main():
    print("=== START TRAINING ===")
    model, tokenizer, model_cfg = GPTFactory.build_20m()

    print(json.dumps(model_cfg.to_dict(), indent=2) + "\n")
    pretty_print_model(model)

    exp_cfg = ExpConfig()
    print("\n" + exp_cfg.model_dump_json(indent=2))

    if DEBUG_GEN:
        inputs = tokenizer(["<|im_start|>ciaooo come va"], return_tensors="pt", add_special_tokens=False)

        print(inputs)
        model.eval()
        preds = model.generate(
            **inputs, # type: ignore
            generation_config=GenerationConfig(),
            max_length=20,
            do_sample=False,
        )
        print(preds)
        print(tokenizer.decode(preds[0]))
        sys.exit(0)

    trainer = Trainer(model, tokenizer, model_cfg, exp_cfg)
    try:
        trainer.train()
    except KeyboardInterrupt as e:
        trainer._save_state()

if __name__ == "__main__":
    main()

from gradientlab.experiments.exp20251016_0_lm_20m_polyrelu_lm_vanilla_fineweb_ita.exp_config import (
    ExpConfig,
)
from gradientlab.experiments.exp20251016_0_lm_20m_polyrelu_lm_vanilla_fineweb_ita.modeling.factory import (
    GPTFactory,
)
from gradientlab.experiments.exp20251016_0_lm_20m_polyrelu_lm_vanilla_fineweb_ita.trainer import (
    train,
)
from gradientlab.logging_utils.log_model_params import pretty_print_model


if __name__ == "__main__":
    model, tokenizer, model_cfg = GPTFactory.build_20m()

    print(model_cfg.model_dump_json(indent=2) + "\n")
    pretty_print_model(model)

    exp_cfg = ExpConfig()
    print("\n" + exp_cfg.model_dump_json(indent=2))

    train(model, tokenizer, model_cfg, exp_cfg)

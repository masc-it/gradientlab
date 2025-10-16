from gradientlab.experiments.exp20251016_0_lm_20m_polyrelu_lm_vanilla_fineweb_ita.modeling.factory import (
    GPTFactory,
)
from gradientlab.logging_utils.log_model_params import pretty_print_model


if __name__ == "__main__":
    model, tokenizer, cfg = GPTFactory.build_20m()
    pretty_print_model(model)

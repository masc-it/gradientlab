from gradientlab.experiments.exp20251016_0_lm_20m_polyrelu_lm_vanilla_fineweb_ita.modeling.model_cfg import (
    ModelConfig,
)
from gradientlab.experiments.exp20251016_0_lm_20m_polyrelu_lm_vanilla_fineweb_ita.modeling.model import (
    GPT,
)
from gradientlab.tokenizers.qwen_tokenizer import qwen3_tokenizer


class GPTFactory:
    @staticmethod
    def build_20m():
        tokenizer = qwen3_tokenizer()
        cfg = ModelConfig(
            enc_dropout=0.05,
            vocab_size=tokenizer.vocab_size + 1,
            pad_token_id=tokenizer.pad_token_id,  # type: ignore
            bos_token_id=tokenizer.convert_tokens_to_ids("<|im_start|>"),  # type: ignore
            eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>"),  # type: ignore
            embed_dim=64,
            enc_num_layers=22,
            hidden_dim=256,
            num_heads=4,
        )
        model = GPT(cfg)
        return model, tokenizer, cfg

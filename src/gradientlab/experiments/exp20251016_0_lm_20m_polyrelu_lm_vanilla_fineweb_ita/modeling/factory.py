from gradientlab.experiments.exp20251016_0_lm_20m_polyrelu_lm_vanilla_fineweb_ita.modeling.model_cfg import (
    ModelConfig,
)
from gradientlab.experiments.exp20251016_0_lm_20m_polyrelu_lm_vanilla_fineweb_ita.modeling.model import (
    GPTForCausalLM,
)
from gradientlab.tokenizers.qwen_tokenizer import qwen3_tokenizer
from gradientlab.tokenizers.byte_tokenizer import byte_tokenizer


class GPTFactory:
    @staticmethod
    def build_20m():
        tokenizer = byte_tokenizer()
        cfg = ModelConfig(
            dropout=0.05,
            attn_dropout=0.0,
            vocab_size=tokenizer.total_vocab_size,
            pad_token_id=tokenizer.pad_token_id,  # type: ignore
            bos_token_id=tokenizer.convert_tokens_to_ids("<|im_start|>"),  # type: ignore
            eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>"),  # type: ignore
            embed_dim=32,
            num_layers=22,
            hidden_dim=256,
            ffn_mult=4.0,
            hidden_squeeze_ratio=0.5,
            num_heads=4,
            num_kv_groups=2,
            use_bias=True,
            max_length=4096,
            tie_word_embeddings=False,
        )

        model = GPTForCausalLM(cfg)
        return model, tokenizer, cfg

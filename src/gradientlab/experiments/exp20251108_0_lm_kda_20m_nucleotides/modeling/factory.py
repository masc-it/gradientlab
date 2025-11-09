from gradientlab.experiments.exp20251108_0_lm_kda_20m_nucleotides.modeling.model_cfg import (
    ModelConfig,
)
from gradientlab.experiments.exp20251108_0_lm_kda_20m_nucleotides.modeling.model import (
    GPTNucleotidesForCausalLM,
)
from gradientlab.tokenizers.byte_tokenizer import byte_tokenizer


class GPTFactory:
    @staticmethod
    def build_20m(resume_from: str | None = None):
        tokenizer = byte_tokenizer()
        cfg = ModelConfig(
            dropout=0.01,
            attn_dropout=0.0,
            vocab_size=tokenizer.total_vocab_size,
            pad_token_id=tokenizer.pad_token_id,  # type: ignore
            bos_token_id=tokenizer.convert_tokens_to_ids("<|im_start|>"),  # type: ignore
            eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>"),  # type: ignore
            embed_dim=256,
            num_layers=22,
            hidden_dim=256,
            patch_size=8,
            ffn_mult=4.0,
            hidden_squeeze_ratio=0.5,
            num_heads=4,
            num_kv_groups=2,
            use_bias=True,
            max_length=4096,
            tie_word_embeddings=False,
        )

        if resume_from is None:
            model = GPTNucleotidesForCausalLM(cfg)
        else:
            print(" === LOAD WEIGHTS FROM CKPT ===")
            model = GPTNucleotidesForCausalLM.from_pretrained(resume_from)
        return model, tokenizer, cfg

from gradientlab.tokenizers.byte_tokenizer import byte_tokenizer

from gradientlab.experiments.exp20251118_0_lm_30m.modeling.model import GPTForCausalLM
from gradientlab.experiments.exp20251118_0_lm_30m.modeling.model_cfg import ModelConfig


class GPTFactory:
    @staticmethod
    def build_30m(resume_from: str | None = None):
        tokenizer = byte_tokenizer()
        cfg = ModelConfig(
            dropout=0.1,
            attn_dropout=0.05,
            vocab_size=tokenizer.total_vocab_size,
            pad_token_id=tokenizer.pad_token_id,  # type: ignore
            bos_token_id=tokenizer.convert_tokens_to_ids("<|im_start|>"),  # type: ignore
            eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>"),  # type: ignore
            num_layers=24,
            hidden_dim=256,
            ffn_mult=4.0,
            num_heads=4,
            num_kv_groups=4,
            use_bias=True,
            max_length=2048,
            max_position_embeddings=2048,
            tie_word_embeddings=True,
        )

        if resume_from is None:
            model = GPTForCausalLM(cfg)
        else:
            print(" === LOAD WEIGHTS FROM CKPT ===")
            model = GPTForCausalLM.from_pretrained(resume_from, config=cfg)
        return model, tokenizer, cfg

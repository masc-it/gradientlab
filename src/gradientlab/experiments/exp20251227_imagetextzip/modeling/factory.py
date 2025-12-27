import torch
from gradientlab.experiments.exp20251227_imagetextzip.modeling.model import (
    DecoderConfig,
    ModelConfig,
    SwinEncoderConfig,
    SwinImageToText,
)
from gradientlab.tokenizers.byte_tokenizer import byte_tokenizer


class GPTFactory:
    @staticmethod
    def build_8m(resume_from: str | None = None):
        tokenizer = byte_tokenizer()

        cfg = ModelConfig(
            vocab_size=tokenizer.total_vocab_size,
            pad_token_id=tokenizer.pad_token_id,  # type: ignore
            encoder=SwinEncoderConfig(
                patch_size=4,
                embed_dim=48,
                depths=(2, 2, 2, 2),
                num_heads=(2, 2, 4, 8),
                drop=0.05,
                attn_drop=0.05,
                drop_path=0.05,
                window_size=8,
                force_flash=False,
            ),
            decoder=DecoderConfig(
                vocab_size=tokenizer.total_vocab_size,
                d_model=128,
                n_layers=4,
                n_heads=8,
                max_seq_len=4096,
                force_flash=False,
                dropout=0.05,
                drop_path=0.05,
            ),
        )

        model = SwinImageToText(cfg)
        if resume_from is not None:
            model.load_state_dict(torch.load(resume_from, map_location="cpu"))

        return model, tokenizer, cfg

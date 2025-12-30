from pathlib import Path
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
    def build_5m(resume_from: str | None = None, use_alibi: bool = True):
        tokenizer = byte_tokenizer()

        cfg = ModelConfig(
            vocab_size=tokenizer.total_vocab_size,
            pad_token_id=tokenizer.pad_token_id,  # type: ignore
            encoder=SwinEncoderConfig(
                patch_size=4,
                embed_dim=32,
                depths=(2, 2, 2, 4),
                num_heads=(2, 2, 4, 4),
                drop=0.1,
                attn_drop=0.1,
                drop_path=0.1,
                window_size=8,
                force_flash=False,
            ),
            decoder=DecoderConfig(
                vocab_size=tokenizer.total_vocab_size,
                d_model=128,
                n_layers=1,
                n_heads=4,
                max_seq_len=4096,
                force_flash=False,
                dropout=0.1,
                drop_path=0.0,
                use_alibi=use_alibi,
                alibi_max_positions=8192,  # 2x max_seq_len for safety
            ),
        )

        model = SwinImageToText(cfg)
        if resume_from is not None:
            model.load_state_dict(torch.load(Path(resume_from) / "model.pt", map_location="cpu"))

        return model, tokenizer, cfg
    
    @staticmethod
    def build_8m(resume_from: str | None = None, use_alibi: bool = True):
        tokenizer = byte_tokenizer()

        cfg = ModelConfig(
            vocab_size=tokenizer.total_vocab_size,
            pad_token_id=tokenizer.pad_token_id,  # type: ignore
            encoder=SwinEncoderConfig(
                patch_size=8,
                embed_dim=32,
                depths=(2, 2, 2, 2),
                num_heads=(2, 2, 4, 8),
                drop=0.0,
                attn_drop=0.0,
                drop_path=0.1,
                window_size=8,
                force_flash=False,
            ),
            decoder=DecoderConfig(
                vocab_size=tokenizer.total_vocab_size,
                d_model=256,
                n_layers=1,
                n_heads=4,
                max_seq_len=4096,
                force_flash=False,
                dropout=0.0,
                drop_path=0.0,
                use_alibi=use_alibi,
                alibi_max_positions=8192,  # 2x max_seq_len for safety
            ),
        )

        model = SwinImageToText(cfg)
        if resume_from is not None:
            model.load_state_dict(torch.load(Path(resume_from) / "model.pt", map_location="cpu"))

        return model, tokenizer, cfg

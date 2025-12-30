from pathlib import Path
import torch

from gradientlab.tokenizers.byte_tokenizer import byte_tokenizer
from .model import (
    ModelConfig,
    CNNEncoderConfig,
    SlotAttentionConfig,
    SequenceCounterConfig,
    ImageTextSlotModel,
)


class ModelFactory:
    @staticmethod
    def build_5m(resume_from: str | None = None):
        """
        Build 5M parameter ImageTextSlot model.

        Args:
            resume_from: Optional path to checkpoint directory to resume from

        Returns:
            Tuple of (model, tokenizer, cfg)
        """
        # Create tokenizer
        tokenizer = byte_tokenizer()
        pad_token_id = tokenizer.pad_token_id
        assert isinstance(pad_token_id, int)

        # Build encoder config
        # Using smaller dims and depths to hit 5M param target
        encoder_cfg = CNNEncoderConfig(
            in_channels=1,
            stem_channels=64,
            depths=(2, 2, 6, 2),  # Increased stage 3 depth since no spatial downsampling
            dims=(64, 256, 256, 256),  # Stage 2 expands to 256, stages 3-4 maintain
            drop_path_rate=0.1,
            layer_scale_init=1e-6,
            mlp_ratio=2.0,
        )

        # Build slot attention config
        slot_attn_cfg = SlotAttentionConfig(
            num_slots=1024,
            d_model=320,  # Reduced from 384 to save params
            num_heads=8,
            mlp_ratio=2,
            dropout=0.1,
            drop_path=0.1,
        )

        # Build sequence counter config
        counter_cfg = SequenceCounterConfig(
            d_model=320,  # Match slot attention d_model
            hidden_dim=160,  # Reduced proportionally
            dropout=0.1,
        )

        # Build main model config
        model_cfg = ModelConfig(
            encoder=encoder_cfg,
            slot_attention=slot_attn_cfg,
            counter=counter_cfg,
            vocab_size=512,
            pad_token_id=pad_token_id,
            num_slots=1024,
            label_smoothing=0.01,
            counter_loss_weight=0.5,  # Reduced from 1.0 to stabilize gradients
        )

        # Instantiate model
        model = ImageTextSlotModel(model_cfg)

        # Load checkpoint if resuming
        if resume_from is not None:
            ckpt_path = Path(resume_from) / "model.pt"
            if ckpt_path.exists():
                state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
                model.load_state_dict(state_dict)
                print(f"Loaded checkpoint from {ckpt_path}")
            else:
                print(f"Warning: Checkpoint not found at {ckpt_path}, starting from scratch")

        return model, tokenizer, model_cfg

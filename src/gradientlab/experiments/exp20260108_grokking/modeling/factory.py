from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from gradientlab.tokenizers.byte_tokenizer import byte_tokenizer
from gradientlab.experiments.exp20260108_grokking.modeling.config import ModelConfig
from gradientlab.experiments.exp20260108_grokking.modeling.model import DecoderOnlyTransformer


def init_weights(module: nn.Module, std: float = 0.02):
    """Initialize weights with GPT-style initialization."""
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=std)


class ModelFactory:
    @staticmethod
    def build_grokking_model(
        resume_from: Optional[str] = None
    ) -> tuple[DecoderOnlyTransformer, object, ModelConfig]:
        """
        Build the grokking decoder-only transformer model.

        Args:
            resume_from: Optional path to checkpoint directory

        Returns:
            Tuple of (model, tokenizer, config)
        """
        tokenizer = byte_tokenizer()

        config = ModelConfig(
            vocab_size=tokenizer.total_vocab_size,
            pad_token_id=tokenizer.pad_token_id, # type: ignore
            d_model=256,
            n_layers=3,
            num_heads=4,
            dropout=0.1,
            ffn_mult=4.0,
            use_alibi=True,
            alibi_max_positions=4096,
        )

        model = DecoderOnlyTransformer(config)

        # Weight initialization (GPT-style)
        model.apply(lambda m: init_weights(m, std=0.02))

        # Zero out padding token embedding
        with torch.no_grad():
            model.embeddings.weight[config.pad_token_id].fill_(0.0)

        if resume_from is not None:
            checkpoint_path = Path(resume_from) / "model.pt"
            if checkpoint_path.exists():
                state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

                # Handle compiled model checkpoint (has "_orig_mod." prefix)
                if any(key.startswith("_orig_mod.") for key in state_dict.keys()):
                    state_dict = {key.replace("_orig_mod.", ""): value for key, value in state_dict.items()}
                    print("Stripped '_orig_mod.' prefix from compiled checkpoint")

                model.load_state_dict(state_dict)
                print(f"Loaded model from {checkpoint_path}")
            else:
                print(f"Warning: Checkpoint {checkpoint_path} not found, starting from scratch")

        return model, tokenizer, config

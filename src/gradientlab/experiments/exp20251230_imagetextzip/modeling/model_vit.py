import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel

from gradientlab.neuralblocks.norm_layers.rmsnorm import RMSNorm
from gradientlab.neuralblocks.ffn.swiglu import SwiGLUFeedForward
from .attention import FlashMHA


# ============================================================================
# Config Classes
# ============================================================================


class ViTEncoderConfig(BaseModel):
    """Configuration for Vision Transformer encoder"""
    in_channels: int = 1
    image_size: int = 128
    patch_size: int = 8
    d_model: int = 256
    num_heads: int = 8
    num_layers: int = 6
    mlp_ratio: float = 2.5
    dropout: float = 0.1


class SequenceCounterConfig(BaseModel):
    """Configuration for sequence counter head"""
    d_model: int = 256
    hidden_dim: int = 128
    dropout: float = 0.1


class ModelViTConfig(BaseModel):
    """Main ViT model configuration"""
    encoder: ViTEncoderConfig
    counter: SequenceCounterConfig
    vocab_size: int = 512
    pad_token_id: int
    num_slots: int = 1024
    chars_per_patch: int = 4  # 1024 slots / 256 patches = 4
    label_smoothing: float = 0.1
    counter_loss_weight: float = 0.1


# ============================================================================
# ViT Encoder Components
# ============================================================================


class PatchEmbedding(nn.Module):
    """
    Convert image to patch embeddings using Conv2d.

    Input: (B, C, H, W) image
    Output: (B, num_patches, d_model) patch sequence
    """

    def __init__(
        self,
        in_channels: int = 1,
        d_model: int = 256,
        patch_size: int = 8,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model

        # Conv2d with kernel_size=patch_size, stride=patch_size
        self.proj = nn.Conv2d(
            in_channels, d_model,
            kernel_size=patch_size, stride=patch_size
        )
        self.norm = RMSNorm(d_model, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) image tensor
        Returns:
            (B, num_patches, d_model) patch embeddings
        """
        # (B, C, H, W) -> (B, d_model, H/P, W/P)
        x = self.proj(x)
        # Flatten spatial: (B, d_model, H', W') -> (B, d_model, H'*W')
        x = x.flatten(2)
        # Transpose: (B, d_model, num_patches) -> (B, num_patches, d_model)
        x = x.transpose(1, 2)
        # Normalize
        x = self.norm(x)
        return x


class TransformerBlock(nn.Module):
    """
    Standard ViT transformer block with pre-norm.

    Architecture:
        RMSNorm → Self-Attention → Dropout → Residual
        RMSNorm → SwiGLU FFN → Dropout → Residual
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        mlp_ratio: float = 2.5,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Self-attention
        self.norm1 = RMSNorm(d_model)
        self.attn = FlashMHA(
            dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            use_alibi=False
        )
        self.dropout1 = nn.Dropout(dropout)

        # FFN
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLUFeedForward(
            d_model=d_model,
            mult=mlp_ratio,
            dropout=dropout,
            use_bias=True,
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, d_model) input sequence
        Returns:
            (B, L, d_model) output sequence
        """
        # Self-attention with pre-norm
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, is_causal=False)
        x = x + self.dropout1(attn_out)

        # FFN with pre-norm
        normed = self.norm2(x)
        ffn_out = self.ffn(normed)
        x = x + self.dropout2(ffn_out)

        return x


class ViTEncoder(nn.Module):
    """
    Vision Transformer encoder for 128x128 grayscale images.

    Architecture:
        PatchEmbedding → + Positional Embedding → N × TransformerBlock → LayerNorm

    Output: (B, num_patches, d_model) - sequence of patch features
    """

    def __init__(self, cfg: ViTEncoderConfig):
        super().__init__()
        self.cfg = cfg

        # Calculate number of patches
        self.num_patches = (cfg.image_size // cfg.patch_size) ** 2

        # Patch embedding
        self.patch_embed = PatchEmbedding(
            in_channels=cfg.in_channels,
            d_model=cfg.d_model,
            patch_size=cfg.patch_size,
        )

        # Learnable positional embeddings
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, cfg.d_model)
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=cfg.d_model,
                num_heads=cfg.num_heads,
                mlp_ratio=cfg.mlp_ratio,
                dropout=cfg.dropout,
            )
            for _ in range(cfg.num_layers)
        ])

        # Final normalization
        self.norm = RMSNorm(cfg.d_model, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, 128, 128) grayscale images
        Returns:
            (B, num_patches, d_model) sequence of patch features
        """
        # Patch embedding: (B, 1, 128, 128) -> (B, 256, d_model)
        x = self.patch_embed(x)

        # Add positional embedding
        x = x + self.pos_embed

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final norm
        x = self.norm(x)

        return x


# ============================================================================
# Sequence Counter
# ============================================================================


class SequenceCounter(nn.Module):
    """
    Predicts the actual character count in the image.

    Architecture:
        Global Average Pool → MLP(d_model → hidden_dim → 1) → Sigmoid

    Note: Predicts normalized count in [0, 1] range for training stability.
    """

    def __init__(self, cfg: SequenceCounterConfig):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, 1),
            nn.Sigmoid(),  # Normalize to [0, 1] range
        )

    def forward(self, slot_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            slot_features: (B, num_slots, d_model)

        Returns:
            (B, 1) predicted character count (normalized to [0, 1])
        """
        # Global average pooling over slots
        pooled = slot_features.mean(dim=1)  # (B, d_model)
        count_normalized = self.mlp(pooled)  # (B, 1) in [0, 1]
        return count_normalized


# ============================================================================
# Main Model
# ============================================================================


class ImageTextSlotViTModel(nn.Module):
    """
    ImageTextSlot with ViT encoder: direct projection from encoder features to character slots.

    Each patch predicts multiple characters (chars_per_patch).
    Non-autoregressive text recognition from images.
    """

    def __init__(self, cfg: ModelViTConfig):
        super().__init__()
        self.cfg = cfg

        # ViT Encoder
        self.encoder = ViTEncoder(cfg.encoder)
        encoder_dim = cfg.encoder.d_model

        # Each patch predicts chars_per_patch characters
        # Classifier: d_model -> chars_per_patch * vocab_size
        self.classifier = nn.Linear(encoder_dim, cfg.chars_per_patch * cfg.vocab_size, bias=True)

        # Sequence counter head
        self.counter = SequenceCounter(cfg.counter)

        # Apply weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights following modern best practices"""
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            # Scale initialization for large output layers
            if isinstance(module, nn.Linear) and module.out_features >= 256:
                std = 0.02 / math.sqrt(module.out_features)
                nn.init.trunc_normal_(module.weight, std=std)
            else:
                nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.zeros_(module.bias)
            if hasattr(module, 'weight') and module.weight is not None:
                nn.init.ones_(module.weight)

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Forward pass for training and inference.

        Args:
            pixel_values: (B, 1, 128, 128) normalized grayscale images
            input_ids: (B, seq_len) tokenized text (not used during forward, only for loss)
            labels: (B, seq_len) target tokens
            attention_mask: (B, seq_len) mask for real tokens

        Returns:
            dict with keys:
                - "loss": total loss (if labels provided)
                - "loss_ce": classification loss component
                - "loss_count": counter loss component
                - "logits": (B, num_slots, vocab_size)
                - "count_pred": (B, 1)
        """
        B = pixel_values.shape[0]
        num_patches = self.encoder.num_patches

        # 1. ViT Encoding
        vit_features = self.encoder(pixel_values)  # (B, num_patches, d_model)

        # 2. Classify each patch into chars_per_patch characters
        patch_logits = self.classifier(vit_features)  # (B, num_patches, chars_per_patch * vocab_size)

        # 3. Reshape to (B, num_slots, vocab_size)
        logits = patch_logits.view(B, num_patches * self.cfg.chars_per_patch, self.cfg.vocab_size)

        # 4. Sequence counter (uses encoder features)
        count_pred = self.counter(vit_features)  # (B, 1)

        output = {
            "logits": logits,
            "count_pred": count_pred,
        }

        # Compute loss if labels provided
        if labels is not None:
            # Prepare targets: pad/truncate to num_slots
            targets = self._prepare_targets(labels)  # (B, num_slots)

            # Classification loss
            loss_ce = F.cross_entropy(
                logits.view(-1, self.cfg.vocab_size),
                targets.view(-1),
                ignore_index=self.cfg.pad_token_id,
                label_smoothing=self.cfg.label_smoothing,
            )

            # Counting loss (with normalized targets for stability)
            if attention_mask is not None:
                actual_counts = attention_mask.sum(dim=1).float()  # (B,)
            else:
                # If no mask, count non-pad tokens in labels
                actual_counts = (labels != self.cfg.pad_token_id).sum(dim=1).float()

            # Normalize counts to [0, 1] range for stable training
            actual_counts_normalized = actual_counts / self.cfg.num_slots

            # MSE loss for normalized values (both in [0, 1] range)
            loss_count = F.mse_loss(
                count_pred.squeeze(-1),
                actual_counts_normalized,
            )

            # Total loss
            total_loss = loss_ce + self.cfg.counter_loss_weight * loss_count

            output.update({
                "loss": total_loss,
                "loss_ce": loss_ce,
                "loss_count": loss_count,
            })

        return output

    def _prepare_targets(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Pad or truncate labels to exactly num_slots length.

        Args:
            labels: (B, seq_len) with variable seq_len

        Returns:
            (B, num_slots) padded/truncated targets
        """
        B, seq_len = labels.shape

        if seq_len < self.cfg.num_slots:
            # Pad with pad_token_id
            padding = torch.full(
                (B, self.cfg.num_slots - seq_len),
                self.cfg.pad_token_id,
                device=labels.device,
                dtype=labels.dtype,
            )
            targets = torch.cat([labels, padding], dim=1)
        elif seq_len > self.cfg.num_slots:
            # Truncate (rare case)
            targets = labels[:, : self.cfg.num_slots]
        else:
            targets = labels

        return targets

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.Tensor,
        bos_token_id: int,
        eos_token_id: int,
        max_new_tokens: int = 256,
    ) -> list:
        """
        Generate text from image (non-autoregressive).

        All slots predicted in parallel, then truncated based on counter prediction.

        Args:
            pixel_values: (B, 1, 128, 128) normalized grayscale images
            bos_token_id: Beginning of sequence token (not used in non-AR generation)
            eos_token_id: End of sequence token (not used in non-AR generation)
            max_new_tokens: Maximum tokens to generate (not used, kept for API compatibility)

        Returns:
            List of (seq_len,) tensors, one per batch item
        """
        self.eval()
        B = pixel_values.shape[0]
        num_patches = self.encoder.num_patches

        # 1. Encode image
        vit_features = self.encoder(pixel_values)  # (B, num_patches, d_model)

        # 2. Classify each patch
        patch_logits = self.classifier(vit_features)  # (B, num_patches, chars_per_patch * vocab_size)
        logits = patch_logits.view(B, num_patches * self.cfg.chars_per_patch, self.cfg.vocab_size)

        # 3. Counter prediction
        count_pred = self.counter(vit_features)  # (B, 1)

        # 4. Greedy decode: argmax over vocab
        tokens = torch.argmax(logits, dim=-1)  # (B, num_slots)

        # 5. Truncate to predicted sequence length (denormalize from [0,1] to [1, num_slots])
        pred_lengths = (count_pred.squeeze(-1) * self.cfg.num_slots).round().long().clamp(1, self.cfg.num_slots)

        # 6. Return variable-length sequences
        output = []
        for i in range(B):
            seq_len = pred_lengths[i].item()
            output.append(tokens[i, :seq_len])

        return output

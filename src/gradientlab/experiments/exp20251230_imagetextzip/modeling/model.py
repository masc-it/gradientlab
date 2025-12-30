import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from pydantic import BaseModel

from gradientlab.neuralblocks.norm_layers.rmsnorm import RMSNorm
from gradientlab.neuralblocks.ffn.swiglu import SwiGLUFeedForward


# ============================================================================
# Config Classes
# ============================================================================


class CNNEncoderConfig(BaseModel):
    """Configuration for ConvNeXt-inspired CNN encoder"""
    in_channels: int = 1
    stem_channels: int = 80
    depths: Tuple[int, int, int, int] = (2, 2, 6, 2)
    dims: Tuple[int, int, int, int] = (80, 160, 320, 640)
    drop_path_rate: float = 0.1
    layer_scale_init: float = 1e-6
    mlp_ratio: float = 4.0


class SlotAttentionConfig(BaseModel):
    """Configuration for slot attention block"""
    num_slots: int = 1024
    d_model: int = 384
    num_heads: int = 8
    mlp_ratio: float = 2.5
    dropout: float = 0.1
    drop_path: float = 0.1


class SequenceCounterConfig(BaseModel):
    """Configuration for sequence counter head"""
    d_model: int = 384
    hidden_dim: int = 192
    dropout: float = 0.1


class ModelConfig(BaseModel):
    """Main model configuration"""
    encoder: CNNEncoderConfig
    slot_attention: SlotAttentionConfig
    counter: SequenceCounterConfig
    vocab_size: int = 512
    pad_token_id: int
    num_slots: int = 1024
    label_smoothing: float = 0.1
    counter_loss_weight: float = 0.1


# ============================================================================
# Utility Modules
# ============================================================================


class DropPath(nn.Module):
    """Stochastic depth (DropPath) for residual branches."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


# ============================================================================
# ConvNeXt Encoder Components
# ============================================================================


class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt Block with inverted bottleneck design.

    Architecture:
        Depthwise Conv 7x7 → LayerNorm → Expand (4x) → GELU → Compress → DropPath
    """

    def __init__(
        self,
        dim: int,
        drop_path: float = 0.0,
        layer_scale_init: float = 1e-6,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=7, padding=3, groups=dim
        )  # Depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        hidden_dim = int(dim * mlp_ratio)
        self.pwconv1 = nn.Linear(dim, hidden_dim)  # Expand
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(hidden_dim, dim)  # Compress

        # Layer scale parameter (initialized small for stable deep networks)
        self.layer_scale = nn.Parameter(
            layer_scale_init * torch.ones(dim), requires_grad=True
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            (B, C, H, W)
        """
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.layer_scale * x
        x = x.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
        x = residual + self.drop_path(x)
        return x


class Downsample(nn.Module):
    """
    Downsampling layer: 2x spatial reduction, 2x channel increase.
    LayerNorm → Conv2d(kernel=2, stride=2)
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(in_channels, eps=1e-6)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            (B, C_out, H//2, W//2)
        """
        x = x.permute(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
        x = self.conv(x)
        return x


class ConvNeXtStage(nn.Module):
    """
    One stage of ConvNeXt encoder: optional downsample + list of blocks.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int,
        drop_path_rates: list,
        layer_scale_init: float,
        mlp_ratio: float,
        downsample: bool = True,
    ):
        super().__init__()

        # Downsample if needed (between stages)
        if downsample:
            self.downsample = Downsample(in_channels, out_channels)
        else:
            self.downsample = nn.Identity()
            assert in_channels == out_channels, "If no downsample, in/out channels must match"

        # Stack of ConvNeXt blocks
        self.blocks = nn.ModuleList([
            ConvNeXtBlock(
                dim=out_channels,
                drop_path=drop_path_rates[i],
                layer_scale_init=layer_scale_init,
                mlp_ratio=mlp_ratio,
            )
            for i in range(depth)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C_in, H, W)
        Returns:
            (B, C_out, H', W')
        """
        x = self.downsample(x)
        for block in self.blocks:
            x = block(x)
        return x


class CNNEncoder(nn.Module):
    """
    ConvNeXt-inspired CNN encoder for 128x128 grayscale images.

    Architecture:
        Stem (Conv4x4/4) → Stage1 (80d) → Stage2 (160d) → Stage3 (320d) → Stage4 (640d)

    Output: (B, 16, 640) - flattened 4x4 spatial with 640-dim features
    """

    def __init__(self, cfg: CNNEncoderConfig):
        super().__init__()
        self.cfg = cfg

        # Stem: Aggressive downsampling (4x4 kernel, stride=4)
        # Input: (B, 1, 128, 128) → Output: (B, 80, 32, 32)
        self.stem = nn.Conv2d(
            cfg.in_channels, cfg.stem_channels, kernel_size=4, stride=4
        )
        self.stem_norm = nn.LayerNorm(cfg.stem_channels, eps=1e-6)

        # Build drop_path rates (linearly increasing)
        total_depth = sum(cfg.depths)
        dpr = [x.item() for x in torch.linspace(0, cfg.drop_path_rate, total_depth)]

        # Build 4 stages
        self.stages = nn.ModuleList()
        in_channels = cfg.stem_channels
        cur_depth = 0

        for stage_idx, (depth, out_channels) in enumerate(zip(cfg.depths, cfg.dims)):
            stage_dpr = dpr[cur_depth : cur_depth + depth]

            # Only stage 2 (idx=1) downsamples for 16x16 output
            self.stages.append(
                ConvNeXtStage(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    depth=depth,
                    drop_path_rates=stage_dpr,
                    layer_scale_init=cfg.layer_scale_init,
                    mlp_ratio=cfg.mlp_ratio,
                    downsample=(stage_idx == 1),
                )
            )
            in_channels = out_channels
            cur_depth += depth

        # Final normalization
        self.norm = nn.LayerNorm(cfg.dims[-1], eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, 128, 128) grayscale images
        Returns:
            (B, 256, 256) sequence of spatial features (16x16 grid)
        """
        # Stem
        x = self.stem(x)  # (B, 64, 32, 32)
        x = x.permute(0, 2, 3, 1)  # (B, 32, 32, 64)
        x = self.stem_norm(x)
        x = x.permute(0, 3, 1, 2)  # (B, 64, 32, 32)

        # Stages
        # Stage 1: (B, 64, 32, 32) - no downsample
        # Stage 2: (B, 256, 16, 16) - downsample (only spatial downsampling stage)
        # Stage 3: (B, 256, 16, 16) - no downsample (deeper processing at 16×16)
        # Stage 4: (B, 256, 16, 16) - no downsample (deeper processing at 16×16)
        for stage in self.stages:
            x = stage(x)

        # Flatten spatial dimensions: (B, 256, 16, 16) → (B, 256, 256)
        x = x.flatten(2).transpose(1, 2)  # (B, 256, 256)

        # Final norm
        x = self.norm(x)

        return x


# ============================================================================
# Attention Components
# ============================================================================


class FlashMHA(nn.Module):
    """Multi-Head Attention using SDPA (FlashAttention when eligible)."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float = 0.0,
        force_flash: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"
        self.force_flash = bool(force_flash)

        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.k_proj = nn.Linear(dim, dim, bias=True)
        self.v_proj = nn.Linear(dim, dim, bias=True)
        self.out_proj = nn.Linear(dim, dim, bias=True)
        self.dropout = float(dropout)

    def forward(
        self,
        x_q: torch.Tensor,
        x_kv: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x_q: Query input (B, L_q, D)
            x_kv: Key/Value input (B, L_kv, D). If None, uses x_q (self-attention)
            attn_mask: Optional attention mask
            is_causal: Whether to apply causal masking

        Returns:
            Output tensor (B, L_q, D)
        """
        if x_kv is None:
            x_kv = x_q

        B, L_q, D = x_q.shape
        L_kv = x_kv.shape[1]

        # Project Q, K, V
        q = self.q_proj(x_q).view(B, L_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_kv).view(B, L_kv, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_kv).view(B, L_kv, self.num_heads, self.head_dim).transpose(1, 2)
        # Shape: (B, num_heads, L, head_dim)

        dropout_p = self.dropout if self.training else 0.0

        # SDPA with optional FlashAttention
        ctx = (
            sdpa_kernel(SDPBackend.FLASH_ATTENTION)
            if self.force_flash
            else sdpa_kernel([
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION,
                SDPBackend.MATH,
            ])
        )
        with ctx:
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal
            )

        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(B, L_q, D)
        return self.out_proj(out)


class SlotAttentionBlock(nn.Module):
    """
    Slot attention block: cross-attention from slot queries to CNN features.

    Architecture:
        Cross-Attention (slots attend to CNN) → RMSNorm → SwiGLU FFN → RMSNorm
    """

    def __init__(self, cfg: SlotAttentionConfig, cnn_dim: int):
        super().__init__()
        self.cfg = cfg

        # Project CNN features to slot dimension if needed
        self.memory_proj = (
            nn.Linear(cnn_dim, cfg.d_model)
            if cnn_dim != cfg.d_model
            else nn.Identity()
        )

        # Cross-attention
        self.cross_attn = FlashMHA(
            dim=cfg.d_model,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
        )
        self.norm1 = RMSNorm(cfg.d_model)

        # FFN
        self.ffn = SwiGLUFeedForward(
            d_model=cfg.d_model,
            mult=cfg.mlp_ratio,
            dropout=cfg.dropout,
            use_bias=True,
        )
        self.norm2 = RMSNorm(cfg.d_model)

        # DropPath for residual connections
        self.drop_path1 = DropPath(cfg.drop_path) if cfg.drop_path > 0.0 else nn.Identity()
        self.drop_path2 = DropPath(cfg.drop_path) if cfg.drop_path > 0.0 else nn.Identity()

    def forward(
        self,
        slot_queries: torch.Tensor,
        memory: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            slot_queries: (B, num_slots, d_model) - learnable queries
            memory: (B, S, cnn_dim) - CNN feature sequence

        Returns:
            (B, num_slots, d_model) - contextualized slot features
        """
        # Project memory if needed
        memory = self.memory_proj(memory)  # (B, S, d_model)

        # Cross-attention with pre-norm
        normed = self.norm1(slot_queries)
        attn_out = self.cross_attn(
            x_q=normed,
            x_kv=memory,
            is_causal=False,
        )
        slot_queries = slot_queries + self.drop_path1(attn_out)

        # FFN with pre-norm
        normed = self.norm2(slot_queries)
        ffn_out = self.ffn(normed)
        slot_queries = slot_queries + self.drop_path2(ffn_out)

        return slot_queries


# ============================================================================
# Sequence Counter
# ============================================================================


class SequenceCounter(nn.Module):
    """
    Predicts the actual character count in the image.

    Architecture:
        Global Average Pool → MLP(384 → 192 → 1) → Sigmoid

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


class ImageTextSlotModel(nn.Module):
    """
    ImageTextSlot: CNN encoder + slot attention + dual classification/counting heads.

    Non-autoregressive text recognition from images using learnable slot queries.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        # CNN Encoder
        self.encoder = CNNEncoder(cfg.encoder)

        # Projection: CNN output dim → slot attention dim
        cnn_output_dim = cfg.encoder.dims[-1]
        slot_dim = cfg.slot_attention.d_model
        self.projection = nn.Linear(cnn_output_dim, slot_dim)

        # Learnable slot queries
        self.slot_queries = nn.Parameter(
            torch.zeros(1, cfg.num_slots, slot_dim)
        )
        nn.init.trunc_normal_(self.slot_queries, std=0.02)

        # Add 1D positional embeddings for slot ordering
        self.slot_pos_embed = nn.Parameter(
            torch.zeros(1, cfg.num_slots, slot_dim)
        )
        # Initialize positional embeddings directly (apply() doesn't catch module-level params)
        nn.init.trunc_normal_(self.slot_pos_embed, std=0.02)

        # Slot attention block
        self.slot_attention = SlotAttentionBlock(cfg.slot_attention, cnn_dim=slot_dim)

        # Pre-classifier normalization for stability
        self.pre_classifier_norm = nn.LayerNorm(slot_dim)

        # Classification head: slot → vocab
        self.classifier = nn.Linear(slot_dim, cfg.vocab_size, bias=True)

        # Sequence counter head
        self.counter = SequenceCounter(cfg.counter)

        # Apply weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights following modern best practices"""
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            # Scale initialization for large output layers (e.g., classifier)
            if isinstance(module, nn.Linear) and module.out_features >= 256:
                # Use output dimension scaling for better gradient stability
                std = 0.02 / math.sqrt(module.out_features)
                nn.init.trunc_normal_(module.weight, std=std)
            else:
                nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d)):
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.zeros_(module.bias)
            if hasattr(module, 'weight') and module.weight is not None:
                nn.init.ones_(module.weight)
        elif isinstance(module, ConvNeXtBlock):
            # Layer scale already initialized in constructor
            pass

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

        # 1. CNN Encoding
        cnn_features = self.encoder(pixel_values)  # (B, 256, 256)

        # 2. Project to slot dimension
        memory = self.projection(cnn_features)  # (B, 256, 320)

        # 3. Broadcast slot queries with positional embeddings and apply cross-attention
        slot_queries = (self.slot_queries + self.slot_pos_embed).expand(B, -1, -1)  # (B, 1024, 320)
        slot_features = self.slot_attention(slot_queries, memory)  # (B, 1024, 320)

        # 4. Classification head (with pre-normalization for stability)
        slot_features_normed = self.pre_classifier_norm(slot_features)
        logits = self.classifier(slot_features_normed)  # (B, 1024, 512)

        # 5. Sequence counter
        count_pred = self.counter(slot_features)  # (B, 1)

        output = {
            "logits": logits,
            "count_pred": count_pred,
        }

        # Compute loss if labels provided
        if labels is not None:
            # Prepare targets: pad/truncate to num_slots
            targets = self._prepare_targets(labels)  # (B, 1024)

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

        All 1024 slots predicted in parallel, then truncated based on counter prediction.

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

        # 1. Encode image through all components
        cnn_features = self.encoder(pixel_values)
        memory = self.projection(cnn_features)
        slot_queries = (self.slot_queries + self.slot_pos_embed).expand(B, -1, -1)
        slot_features = self.slot_attention(slot_queries, memory)
        slot_features_normed = self.pre_classifier_norm(slot_features)
        logits = self.classifier(slot_features_normed)  # (B, 1024, 512)
        count_pred = self.counter(slot_features)  # (B, 1)

        # 2. Greedy decode: argmax over vocab
        tokens = torch.argmax(logits, dim=-1)  # (B, 1024)

        # 3. Truncate to predicted sequence length (denormalize from [0,1] to [1, num_slots])
        pred_lengths = (count_pred.squeeze(-1) * self.cfg.num_slots).round().long().clamp(1, self.cfg.num_slots)

        # 4. Return variable-length sequences
        output = []
        for i in range(B):
            seq_len = pred_lengths[i].item()
            output.append(tokens[i, :seq_len])

        return output

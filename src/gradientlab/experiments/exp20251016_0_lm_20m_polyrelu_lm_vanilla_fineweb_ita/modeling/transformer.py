from typing import Optional
import torch
from torch import nn
from transformers import Cache

from gradientlab.experiments.exp20251016_0_lm_20m_polyrelu_lm_vanilla_fineweb_ita.modeling.attention import (
    Attention,
)
from gradientlab.experiments.exp20251016_0_lm_20m_polyrelu_lm_vanilla_fineweb_ita.modeling.model_cfg import (
    ModelConfig,
)
from gradientlab.neuralblocks.ffn.polynorm import PolyNormFeedForward
from gradientlab.neuralblocks.model_types import AttnMask
from gradientlab.neuralblocks.norm_layers.rmsnorm import RMSNorm


class TransformerEncoderBlock(nn.Module):
    def __init__(self, cfg: ModelConfig, hidden_dim: int, ffn_mult: int) -> None:
        super().__init__()
        self.cfg = cfg

        hidden_dim = hidden_dim
        self.norm = RMSNorm(cfg.hidden_dim)

        self.self_attn = Attention(
            in_hidden_dim=cfg.hidden_dim,
            hidden_dim=hidden_dim,
            dropout=cfg.dropout,
            num_heads=cfg.num_heads,
            use_bias=cfg.use_bias,
        )
        self.ffn = PolyNormFeedForward(
            cfg.hidden_dim,
            dropout=cfg.dropout,
            mult=ffn_mult,
            use_bias=cfg.use_bias,
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: AttnMask,
        kv_cache: Optional[Cache],
        layer_idx: int,
        is_causal: bool,
        use_cache: bool,
    ):
        u = self.norm(x)

        # parallel branches
        x_attn = self.self_attn(
            u,
            u,
            u,
            attn_mask=attn_mask,
            is_causal=is_causal,
            kv_cache=kv_cache,
            layer_idx=layer_idx,
            use_cache=use_cache,
        )
        x_ffn = self.ffn(u)

        # single residual add (PaLM style)
        x = x + (x_attn + x_ffn)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        #
        self.blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    cfg,
                    hidden_dim=int(cfg.hidden_dim * cfg.hidden_squeeze_ratio)
                    if (i >= cfg.num_layers // 3 and i % 2 == 0)
                    and (i != cfg.num_layers - 1)
                    else cfg.hidden_dim,
                    ffn_mult=cfg.ffn_mult,
                )
                for i in range(cfg.num_layers)
            ]
        )
        self.norm_out = RMSNorm(cfg.hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Cache] = None,
        attn_mask: AttnMask = None,
        use_cache: bool = False,
        is_causal: bool = False,
    ):
        for i, block in enumerate(self.blocks):
            x = block(
                x,
                attn_mask=attn_mask,
                kv_cache=kv_cache,
                layer_idx=i,
                is_causal=is_causal,
                use_cache=use_cache,
            )
        x = self.norm_out(x)
        return x, kv_cache

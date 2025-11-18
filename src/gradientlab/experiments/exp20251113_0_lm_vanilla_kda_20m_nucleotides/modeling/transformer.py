from typing import Optional
import torch
from torch import nn
from transformers import Cache

from gradientlab.experiments.exp20251113_0_lm_vanilla_kda_20m_nucleotides.modeling.attention import Attention
from gradientlab.experiments.exp20251113_0_lm_vanilla_kda_20m_nucleotides.modeling.model_cfg import (
    ModelConfig,
)
from gradientlab.neuralblocks.ffn.polynorm import PolyNormFeedForward

from gradientlab.neuralblocks.ffn.swiglu import SwiGLUFeedForward
from gradientlab.neuralblocks.model_types import AttnMask
from gradientlab.neuralblocks.norm_layers.rmsnorm import RMSNorm
from fla.layers import KimiDeltaAttention


class TransformerEncoderBlock(nn.Module):
    def __init__(self, cfg: ModelConfig, hidden_dim: int, ffn_mult: float, layer_idx: int) -> None:
        super().__init__()
        self.cfg = cfg

        hidden_dim = hidden_dim
        self.norm1 = RMSNorm(cfg.hidden_dim)
        self.norm2 = RMSNorm(cfg.hidden_dim)
        
        """ self.attn = KimiDeltaAttention(
            hidden_size=hidden_dim,
            num_heads=cfg.num_heads,
            head_dim=hidden_dim//cfg.num_heads,
            layer_idx=layer_idx,
        ) """

        self.attn = Attention(
            in_hidden_dim=hidden_dim,
            hidden_dim=hidden_dim,
            dropout=cfg.attn_dropout,
            num_heads=cfg.num_heads,
        )

        self.ffn = SwiGLUFeedForward(
            cfg.hidden_dim,
            dropout=cfg.dropout,
            mult=ffn_mult,
            use_bias=cfg.use_bias,
        )

    #@torch.compiler.disable()
    def _attn(self, **kwargs):
        return self.attn(
            **kwargs
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
        
        attn_input = self.norm1(x)
        
        """ x_attn, _, kv_cache = self._attn(
            hidden_states=attn_input,
            attn_mask=attn_mask,
            kv_cache=kv_cache,
            use_cache=use_cache,
        ) """

        x_attn = self._attn(
            in_q=attn_input,
            in_k=attn_input,
            in_v=attn_input,
            attn_mask=attn_mask,
            is_causal=is_causal,
            kv_cache=kv_cache,
            layer_idx=layer_idx,
            use_cache=use_cache,
        )

        x = x + x_attn

        ffn_out = self.ffn(self.norm2(x))
        x = x + ffn_out
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
                    layer_idx=i,
                    hidden_dim=cfg.hidden_dim,
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

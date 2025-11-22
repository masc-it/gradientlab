from typing import Optional
import torch
from torch import nn
import torch.nn.functional as F
from transformers import Cache

from gradientlab.neuralblocks.model_types import AttnMask


class Attention(nn.Module):
    def __init__(
        self,
        in_hidden_dim: int,
        hidden_dim: int,
        dropout: float,
        num_heads: int,
        use_bias: bool = True,
        qk_ratio: float = 2.0,  # <1.0 squeezes Q/K head dim
        v_ratio: float = 2.0,  # can be 1.0–2.0 to expand V head dim
        num_kv_groups: Optional[
            int
        ] = 2,  # =1 -> MQA, < num_heads -> GQA, =num_heads -> classic MHA
        qk_init_scale: float = 0.5,  # downscale Q/K init std to stabilize squeezed logits
    ) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.in_hidden_dim = in_hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout

        assert hidden_dim % num_heads == 0, (
            "`hidden_dim` must be divisible by `num_heads`."
        )
        self.head_dim = hidden_dim // num_heads

        # ---- derived per-head dims after squeeze/expand ----
        # Q/K head dim (squeezed) and V head dim (possibly expanded)
        self.qk_head_dim = max(1, int(round(self.head_dim * qk_ratio)))
        self.v_head_dim = max(1, int(round(self.head_dim * v_ratio)))

        # Grouped-Query / Multi-Query: how many distinct K/V groups to compute
        if num_kv_groups is None:
            num_kv_groups = num_heads  # default: classic MHA
        assert 1 <= num_kv_groups <= num_heads and (num_heads % num_kv_groups == 0), (
            "`num_kv_groups` must divide `num_heads`."
        )
        self.num_kv_groups = num_kv_groups
        self.heads_per_kv_group = num_heads // num_kv_groups

        # Projections:
        # Q: [B, L, in_hidden] -> [B, L, num_heads * qk_head_dim]
        # K: [B, L, in_hidden] -> [B, L, num_kv_groups * qk_head_dim]
        # V: [B, L, in_hidden] -> [B, L, num_kv_groups * v_head_dim]
        self.q = nn.Linear(
            self.in_hidden_dim, self.num_heads * self.qk_head_dim, bias=use_bias
        )
        self.k = nn.Linear(
            self.in_hidden_dim, self.num_kv_groups * self.qk_head_dim, bias=use_bias
        )
        self.v = nn.Linear(
            self.in_hidden_dim, self.num_kv_groups * self.v_head_dim, bias=use_bias
        )

        self.gate_proj = nn.Linear(
            self.in_hidden_dim, 
            self.num_heads * self.v_head_dim, 
            bias=False
        )
        # Output projection expects concatenated V of size (num_heads * v_head_dim)
        self.out_proj = nn.Linear(
            self.num_heads * self.v_head_dim, self.in_hidden_dim, bias=use_bias
        )

        self._init_weights(qk_init_scale=qk_init_scale)

    def _init_weights(self, qk_init_scale: float = 0.5):
        # Xavier uniform for all, then gently downscale Q/K to stabilize squeezed logits.
        for m in [self.q, self.k, self.v, self.out_proj]:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        with torch.no_grad():
            self.q.weight.mul_(qk_init_scale)
            self.k.weight.mul_(qk_init_scale)

    def forward(
        self,
        in_q: torch.Tensor,
        in_k: torch.Tensor,
        in_v: torch.Tensor,
        attn_mask: AttnMask,
        is_causal: bool,
        kv_cache: Cache,
        layer_idx: int,
        use_cache: bool = False,
    ):
        B, L_Q, D = in_q.size()
        _, L_KV, D = in_k.size()

        q = self.q(in_q)
        q = self._reshape_q(q, B, L_Q)  # [B, H, L_Q, qk_head_dim]

        k = self._reshape_k(self.k(in_k), B, L_KV)  # [B, G, L_KV, qk_head_dim]
        v = self._reshape_v(self.v(in_v), B, L_KV)  # [B, G, L_KV, v_head_dim]

        if use_cache:
            k, v = kv_cache.update(k, v, layer_idx)

        # 3) Expand K/V groups to per-head tensors (repeat per group)
        #    q: [B, H, L_Q, qk_d], k_g/v_g: [B, G, L_KV, *], expand -> [B, H, L_KV, *]
        if self.num_kv_groups != self.num_heads:
            k = k.repeat_interleave(self.heads_per_kv_group, dim=1)
            v = v.repeat_interleave(self.heads_per_kv_group, dim=1)

        # print(f"{q.shape=} {k.shape=} {v.shape=}")
        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            is_causal=is_causal,
            dropout_p=0.0,
        )  # -> [B, H, L_Q, v_head_dim]

        # Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free
        # https://arxiv.org/abs/2505.06708
        gate_scores = torch.sigmoid(
            self.gate_proj(in_q)
        )  # [B, L_Q, num_heads * v_head_dim]
        
        # Reshape gate scores to match SDPA output: [B, num_heads, L_Q, v_head_dim]
        gate_scores = gate_scores.view(
            B, L_Q, self.num_heads, self.v_head_dim
        ).transpose(1, 2)
        
        # Apply multiplicative gating element-wise
        x = x * gate_scores  # [B, H, L_Q, v_head_dim]
        
        # 5) Merge heads and project out
        x = x.transpose(1, 2).reshape(B, L_Q, self.num_heads * self.v_head_dim)
        x = self.out_proj(x)
        return x

    # ---------- shape helpers ----------
    def _reshape_q(self, x: torch.Tensor, B: int, L: int):
        # x: [B, L, H*qk_d] -> [B, H, L, qk_d]
        return x.view(B, L, self.num_heads, self.qk_head_dim).transpose(1, 2)

    def _reshape_k(self, x: torch.Tensor, B: int, L: int):
        # x: [B, L, G*qk_d] -> [B, G, L, qk_d]
        return x.view(B, L, self.num_kv_groups, self.qk_head_dim).transpose(1, 2)

    def _reshape_v(self, x: torch.Tensor, B: int, L: int):
        # x: [B, L, G*v_d] -> [B, G, L, v_d]
        return x.view(B, L, self.num_kv_groups, self.v_head_dim).transpose(1, 2)

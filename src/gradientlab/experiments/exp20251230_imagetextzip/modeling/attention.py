from typing import Dict, Optional, Tuple
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

class FlashMHA(nn.Module):
    """MHA implemented via SDPA (FlashAttention when eligible)."""

    def __init__(
        self, dim: int, num_heads: int, dropout: float = 0.0, force_flash: bool = False,
        use_alibi: bool = False, alibi_max_positions: int = 4096
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim
        self.force_flash = bool(force_flash)

        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.k_proj = nn.Linear(dim, dim, bias=True)
        self.v_proj = nn.Linear(dim, dim, bias=True)
        self.out_proj = nn.Linear(dim, dim, bias=True)
        self.dropout = float(dropout)

        self.use_alibi = use_alibi
        if self.use_alibi:
            self._init_alibi_biases(alibi_max_positions)

    def _init_alibi_biases(self, max_positions: int):
        """
        Pre-compute ALiBi slopes and bias matrix.

        ALiBi uses slopes: m_h = 2^(-8h/H) for head h (0-indexed).
        Bias for position (i,j) = -m_h * |i - j|
        For causal attention, we only need lower triangular part.
        """
        # Compute per-head slopes: m_h = 2^(-8h/H)
        # Shape: (num_heads,)
        slopes = torch.pow(
            2.0,
            -8.0 * torch.arange(1, self.num_heads + 1, dtype=torch.float32) / self.num_heads
        )

        # Pre-compute relative position matrix: |i - j|
        # For causal attention, we need positions [0, max_positions)
        positions = torch.arange(max_positions, dtype=torch.float32)
        # Create matrix where element [i, j] = |i - j|
        # Shape: (max_positions, max_positions)
        relative_positions = torch.abs(positions[:, None] - positions[None, :])

        # Apply slopes to get biases per head
        # Shape: (num_heads, max_positions, max_positions)
        # bias[h, i, j] = -slopes[h] * |i - j|
        alibi_biases = -slopes[:, None, None] * relative_positions[None, :, :]

        # For causal attention, mask out future positions (set to -inf)
        # This creates a lower-triangular pattern
        causal_mask = torch.triu(torch.ones(max_positions, max_positions, dtype=torch.bool), diagonal=1)
        alibi_biases_causal = alibi_biases.clone()
        alibi_biases_causal[:, causal_mask] = float('-inf')

        # Register as buffers (moved to device automatically, not trained)
        # Note: persistent=False means they won't be saved in state_dict
        self.register_buffer("alibi_slopes", slopes, persistent=False)
        self.register_buffer("alibi_biases_causal", alibi_biases_causal, persistent=False)
        self.register_buffer("alibi_biases_non_causal", alibi_biases, persistent=False)
        self.alibi_max_positions = max_positions

    def _get_alibi_bias(
        self,
        seq_len_q: int,
        seq_len_k: int,
        is_causal: bool,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """
        Get ALiBi bias for given sequence lengths.

        Args:
            seq_len_q: Query sequence length
            seq_len_k: Key sequence length (may differ in cross-attention)
            is_causal: Whether this is causal attention
            device: Target device

        Returns:
            Bias tensor of shape (1, num_heads, seq_len_q, seq_len_k) or None
        """
        if not self.use_alibi:
            return None

        # For self-attention, seq_len_q == seq_len_k
        # For cross-attention, ALiBi should NOT be used
        if seq_len_q != seq_len_k:
            # This is cross-attention, no ALiBi
            return None

        # Check if we need to recompute (sequence too long)
        if seq_len_q > self.alibi_max_positions:
            # Fall back to on-the-fly computation for very long sequences
            return self._compute_alibi_bias_dynamic(seq_len_q, is_causal, device)

        # Slice pre-computed biases
        if is_causal:
            bias = self.alibi_biases_causal[:, :seq_len_q, :seq_len_k]
        else:
            bias = self.alibi_biases_non_causal[:, :seq_len_q, :seq_len_k]

        # Add batch dimension: (num_heads, L, L) -> (1, num_heads, L, L)
        return bias.unsqueeze(0)

    def _compute_alibi_bias_dynamic(
        self,
        seq_len: int,
        is_causal: bool,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Dynamically compute ALiBi bias for sequences longer than cached length.
        This is a fallback for very long sequences.
        """
        positions = torch.arange(seq_len, dtype=torch.float32, device=device)
        relative_positions = torch.abs(positions[:, None] - positions[None, :])

        slopes = self.alibi_slopes.to(device)
        biases = -slopes[:, None, None] * relative_positions[None, :, :]

        if is_causal:
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)
            biases[:, causal_mask] = float('-inf')

        return biases.unsqueeze(0)  # (1, num_heads, L, L)

    def _sdpa(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        attn_mask: Optional[torch.Tensor],
        is_causal: bool,
        alibi_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Scaled dot-product attention with optional ALiBi biases.

        Args:
            q, k, v: Query, key, value tensors (B, num_heads, L, head_dim)
            attn_mask: Optional boolean or float attention mask
            is_causal: Whether to apply causal masking
            alibi_bias: Optional ALiBi bias (1, num_heads, L_q, L_k) or (B, num_heads, L_q, L_k)
        """
        dropout_p = self.dropout if self.training else 0.0

        # Merge ALiBi bias with existing attention mask
        if alibi_bias is not None:
            if attn_mask is not None:
                # Combine masks: both must be broadcastable
                # attn_mask is typically (B, 1, L, L) boolean or float
                # alibi_bias is (1, num_heads, L, L)

                # Convert boolean mask to float if needed
                if attn_mask.dtype == torch.bool:
                    # True = can attend, False = cannot attend
                    # Convert to additive mask: False -> -inf, True -> 0.0
                    float_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
                    float_mask.masked_fill_(~attn_mask, float('-inf'))
                    combined_mask = float_mask + alibi_bias
                else:
                    # Already float, just add
                    combined_mask = attn_mask + alibi_bias
            else:
                combined_mask = alibi_bias

            # When using explicit mask, disable is_causal flag
            # (the mask already encodes causality through -inf values)
            use_causal = False
        else:
            combined_mask = attn_mask
            use_causal = is_causal

        ctx = (
            sdpa_kernel(SDPBackend.FLASH_ATTENTION)
            if self.force_flash
            else sdpa_kernel(
                [
                    SDPBackend.FLASH_ATTENTION,
                    SDPBackend.EFFICIENT_ATTENTION,
                    SDPBackend.MATH,
                ]
            )
        )
        with ctx:
            return F.scaled_dot_product_attention(
                q, k, v, attn_mask=combined_mask, dropout_p=dropout_p, is_causal=use_causal
            )

    def forward(
        self,
        x_q: torch.Tensor,
        x_kv: Optional[torch.Tensor] = None,
        *,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        kv_cache: Optional[Dict[str, torch.Tensor]] = None,
        cache_prefix: str = "self",
        precomputed_k: Optional[torch.Tensor] = None,
        precomputed_v: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        if x_kv is None:
            x_kv = x_q

        B, L, D = x_q.shape
        S = x_kv.shape[1]

        q = (
            self.q_proj(x_q).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        )  # (B,h,L,hd)

        if precomputed_k is not None and precomputed_v is not None:
            k = precomputed_k
            v = precomputed_v
        else:
            # Standard logic (Self-Attn or non-optimized Cross-Attn)
            if x_kv is None:
                x_kv = x_q
            S = x_kv.shape[1]
            k = self.k_proj(x_kv).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(x_kv).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Handle KV caching
        actual_is_causal = is_causal
        if kv_cache is not None:
            k_key, v_key = f"{cache_prefix}_k", f"{cache_prefix}_v"
            if k_key in kv_cache and v_key in kv_cache:
                k = torch.cat([kv_cache[k_key], k], dim=2)
                v = torch.cat([kv_cache[v_key], v], dim=2)
            kv_cache[k_key] = k
            kv_cache[v_key] = v
            # With incremental decoding and no future keys, causal masking is unnecessary.
            actual_is_causal = False

        # Generate ALiBi bias
        # Note: k.shape[2] is the actual key sequence length after caching
        alibi_bias = self._get_alibi_bias(
            seq_len_q=q.shape[2],  # Current query length (usually 1 during generation)
            seq_len_k=k.shape[2],  # Key length (grows with cache)
            is_causal=actual_is_causal,
            device=q.device,
        )

        out = self._sdpa(q, k, v, attn_mask=attn_mask, is_causal=actual_is_causal, alibi_bias=alibi_bias)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(out), kv_cache


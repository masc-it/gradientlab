import torch


def make_bool_causal_mask(
    target_ids: torch.Tensor,
    pad_id: int,
) -> torch.Tensor:
    """
    Build a boolean attention mask for SDPA (PyTorch 2.6+):
      - True  => allowed (takes part in attention)
      - False => masked (disallowed)

    The result is broadcastable to [B, H, T, T].

    Args:
        target_ids: [B, T] token ids
        pad_id: int, padding token id

    Returns:
        attn_mask: bool [B, 1, T, T] with True=allow, False=mask
    """
    B, T = target_ids.shape
    device = target_ids.device

    # Allow attending only to non-PAD keys
    key_is_valid = target_ids != pad_id  # [B, T], True=valid key
    key_allow = key_is_valid.view(B, 1, 1, T)  # [B,1,1,T]
    key_allow = key_allow.expand(B, 1, T, T)  # [B,1,T,T]

    # Causal: allow only j <= i
    causal_allow = torch.tril(
        torch.ones(T, T, dtype=torch.bool, device=device)
    )  # [T,T]
    causal_allow = causal_allow.view(1, 1, T, T)  # [B,1,T,T]

    # Final boolean mask: True=allow, False=mask
    attn_mask = key_allow & causal_allow  # [B,1,T,T]
    return attn_mask


def build_causal_mask_4d(q_len: int, past_len: int, device):
    # Boolean mask where True = *masked* for SDPA
    # Keys are indexed [0 .. past_len + q_len - 1]
    k_len = past_len + q_len
    i = torch.arange(q_len, device=device).unsqueeze(-1)  # [q,1]
    j = torch.arange(k_len, device=device).unsqueeze(0)  # [1,k]
    # allow keys j <= past_len + i
    causal_mask = j > (past_len + i)  # [q,k], True means block
    # SDPA expects broadcastable [B or 1, H or 1, q, k]
    return causal_mask.view(1, 1, q_len, k_len)

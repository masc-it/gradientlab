import math
from typing import Dict, Optional, Tuple
import torch
from torch import nn

from gradientlab.experiments.exp20251016_0_lm_20m_polyrelu_lm_vanilla_fineweb_ita.config import (
    ModelConfig,
)

from gradientlab.experiments.exp20251016_0_lm_20m_polyrelu_lm_vanilla_fineweb_ita.modeling.transformer import (
    TransformerEncoder,
)
from gradientlab.neuralblocks.model_types import Model_KVCache

import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Default positional encoding."""

    def __init__(self, d_model: int, max_len: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(pos * div)
        pe[0, :, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor, start_pos: int) -> torch.Tensor:
        return self.dropout(x + self.pe[:, start_pos : start_pos + x.size(1)])  # type: ignore


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.num_layers = cfg.enc_num_layers
        self.vocab_size = cfg.vocab_size
        self.pad_token_id = cfg.pad_token_id
        self.txt_embed = nn.Embedding(
            cfg.vocab_size, cfg.embed_dim, padding_idx=cfg.pad_token_id
        )
        self.pos_encoding = PositionalEncoding(
            cfg.embed_dim, max_len=cfg.max_len, dropout=0.0
        )

        self.dec_adapter = nn.Linear(cfg.embed_dim, cfg.hidden_dim, bias=False)
        self.decoder = TransformerEncoder(cfg)

        self.lm_head = nn.Linear(cfg.embed_dim, cfg.vocab_size, bias=False)

        self.lm_head.weight = self.txt_embed.weight
        self.init_net_weights()

    def forward(
        self,
        input_ids: torch.Tensor,
        kv_cache: Model_KVCache,
        target_ids: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ):
        Q_LEN = input_ids.size(1)
        first_kv_block = kv_cache[0]
        if use_cache and self.num_layers > 0 and first_kv_block is not None:
            CACHE_LEN: int = int(first_kv_block["k"].size(2))
            attn_mask = None
        else:
            CACHE_LEN = 0
            attn_mask = None

        embeds = self.txt_embed(input_ids)
        embeds = self.pos_encoding(embeds, CACHE_LEN)

        hidden_states = self.dec_adapter(embeds)
        hidden_states, kv_cache = self.decoder(
            hidden_states,
            attn_mask=attn_mask,
            is_causal=True if Q_LEN > 1 and attn_mask is None else False,
            kv_cache=kv_cache,
            use_cache=use_cache,
        )

        # hidden_states = self.lm_adapter(hidden_states)
        hidden_states = F.linear(hidden_states, self.dec_adapter.weight.T)

        logits = self.lm_head(hidden_states)

        loss: Optional[Dict[str, torch.Tensor]] = None
        if target_ids is not None:
            ce = torch.nn.functional.cross_entropy(
                logits[:, :-1].contiguous().reshape(-1, self.vocab_size),
                target_ids[:, 1:].reshape(-1),
                ignore_index=self.pad_token_id,
            )
            loss = {"loss": ce, "ce_loss": ce}
        return logits, loss, kv_cache

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        use_cache: bool,
        max_len: int,
    ):
        kv_cache: Model_KVCache = [None] * self.cfg.enc_num_layers

        B = input_ids.size(0)
        decoder_ids = input_ids

        finished = torch.zeros(B, dtype=torch.bool, device=input_ids.device)
        device = input_ids.device

        autocast_dtype = (
            torch.bfloat16 if device.type in ["cpu", "cuda"] else torch.float16
        )
        for i in range(max_len):
            if i > 0:
                decoder_input = (
                    decoder_ids if not use_cache else decoder_ids[:, -1].unsqueeze(-1)
                )
            else:
                decoder_input = decoder_ids

            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
            ):
                logits, _, kv_cache = self.forward(
                    decoder_input,
                    kv_cache=kv_cache,  #  type: ignore
                    use_cache=use_cache,
                )

            # greedy decoding
            next_token = torch.argmax(logits[:, -1, :], dim=1, keepdim=True)

            new_finished = next_token.squeeze(1) == self.cfg.eos_token_id
            finished = finished | new_finished
            decoder_ids = torch.concat([decoder_ids, next_token], dim=1)
            if finished.all():
                break

        return decoder_ids

    def _init_weights(self, m):
        std = 0.02
        if isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=1 / math.sqrt(self.cfg.embed_dim))
            m.weight.data[self.cfg.pad_token_id].zero_()
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        # Residual branch scaling: apply to attention out-proj and MLP second proj
        scale = 1.0 / math.sqrt(
            2 * self.cfg.enc_num_layers
        )  # Pre-LN, two residuals per block
        for name, p in m.named_parameters():
            # Adjust these name checks to your actual module names:
            if name.endswith(
                ("self_attn.out_proj.weight", "ffn.w2.weight", "ffn.w3.weight")
            ):
                with torch.no_grad():
                    p.mul_(scale)

    def init_net_weights(self):
        self.apply(self._init_weights)

    @staticmethod
    def get_loss_keys():
        return ["loss", "ce_loss"]

    @torch.jit.export
    def generate_step(
        self,
        input_ids: torch.Tensor,  # [B, 1] (last token)
        kv_cache: Model_KVCache,
        use_cache: bool = True,
        # sampling controls:
        temperature: float = 1.0,  # > 0 (1.0 = logits as-is)
        top_k: int = 0,  # 0 = disabled; else keep top_k tokens
        min_p: float = 0.0,  # 0..1; keep tokens with p >= min_p * max_p
        repetition_penalty: float = 1.0,  # >=1.0; 1.0 = disabled
        # repetition context (optional)
        prev_tokens: Optional[
            torch.Tensor
        ] = None,  # [B, T_prev] full or windowed history
        rep_window: int = 128,  # apply penalty over last N tokens
    ) -> Tuple[torch.Tensor, Model_KVCache]:
        # 1) forward one step
        logits, _, kv_cache = self.forward(
            input_ids, kv_cache=kv_cache, target_ids=None, use_cache=use_cache
        )

        # work on final-step logits, ensure float32 for numerics
        last_logits = logits[:, -1, :].to(dtype=torch.float32)  # [B, V]
        B, V = last_logits.size()

        # 2) temperature
        t = temperature if temperature > 1e-6 else 1e-6
        last_logits = last_logits / t

        # 3) repetition penalty (presence-style, GPT-2)
        #    apply to any token that appears in the recent history per batch
        if (repetition_penalty != 1.0) and (prev_tokens is not None):
            rp = repetition_penalty if repetition_penalty > 0.0 else 1.0
            for b in range(B):
                # select last rep_window tokens for this batch
                Tprev = int(prev_tokens.size(1))
                start = int(Tprev - rep_window) if Tprev > rep_window else 0
                hist = prev_tokens[b, start:Tprev].contiguous()
                if hist.numel() == 0:
                    continue
                uniq = torch.unique(hist)  # 1D Long
                row = last_logits[b]
                sel = row.index_select(0, uniq)
                # if logit > 0: divide; else multiply
                sel = torch.where(sel > 0, sel / rp, sel * rp)
                row = row.scatter(0, uniq, sel)
                last_logits[b] = row

        # 4) top-k (on logits)
        if top_k > 0 and top_k < V:
            k = int(top_k)
            topk_vals = torch.topk(last_logits, k, dim=1).values  # [B, k]
            kth = topk_vals[:, -1].unsqueeze(1)  # [B, 1]
            mask = last_logits < kth
            last_logits = last_logits.masked_fill(mask, -1e9)

        # 5) convert to probabilities
        probs = F.softmax(last_logits, dim=1)  # [B, V]

        # 6) min_p filter (keep p >= min_p * p_max)
        if min_p > 0.0:
            mp = min_p if min_p < 1.0 else 1.0
            pmax = probs.max(dim=1, keepdim=True).values  # [B, 1]
            thresh = pmax * mp
            keep = probs >= thresh
            probs = torch.where(keep, probs, torch.zeros_like(probs))

        # 7) renormalize; if a row sums to zero, fall back to argmax
        sums = probs.sum(dim=1, keepdim=True)  # [B, 1]
        # avoid divide-by-zero
        probs = torch.where(sums > 0, probs / sums, probs)

        # 8) sample (row-wise); fallback to argmax when needed
        next_token = torch.empty(B, 1, dtype=torch.long, device=probs.device)
        for b in range(B):
            if float(sums[b, 0]) > 0.0:
                # multinomial expects 1D probs
                idx = torch.multinomial(probs[b], 1)  # [1]
                next_token[b, 0] = idx[0]
            else:
                next_token[b, 0] = int(torch.argmax(last_logits[b], dim=0))

        return next_token, kv_cache

import math
from typing import Optional
import torch
from torch import nn


from gradientlab.experiments.exp20251108_0_lm_kda_20m_nucleotides.modeling.model_cfg import (
    ModelConfig,
)
from gradientlab.experiments.exp20251108_0_lm_kda_20m_nucleotides.modeling.transformer import (
    TransformerEncoder,
)

import torch.nn.functional as F

from transformers import GenerationMixin, PreTrainedModel, AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.cache_utils import Cache, DynamicCache

from gradientlab.neuralblocks.attention.causal_mask import make_causal_mask_from_attn_mask

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


class GPTVForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = ModelConfig
    supports_gradient_checkpointing = False

    def __init__(self, config: ModelConfig):
        super().__init__(config)

        self.cfg = config
        self.num_layers = config.num_layers
        self.vocab_size = config.vocab_size
        self.pad_token_id = config.pad_token_id

        self.txt_embed = nn.Embedding(
            config.vocab_size, config.embed_dim, padding_idx=config.pad_token_id
        )
        self.pos_encoding = PositionalEncoding(
            config.embed_dim, max_len=config.max_length, dropout=0.0
        )

        self.dec_adapter = nn.Linear(config.embed_dim, config.hidden_dim, bias=False)
        self.decoder = TransformerEncoder(config)

        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.txt_embed

    def set_input_embeddings(self, value):
        self.txt_embed = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def tie_weights(self):
        if getattr(self.config, "tie_word_embeddings", True):
            self._tie_or_clone_weights(self.lm_head, self.txt_embed)

    def prepare_inputs_for_generation( # type: ignore
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        if past_key_values is not None and past_key_values.get_seq_length() > 0:
            input_ids = input_ids[:, -1:]
            attention_mask = None

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "use_cache": kwargs.get("use_cache", True),
        }

    def _reorder_cache(self, past_key_values: Cache, beam_idx: torch.LongTensor):
        past_key_values.batch_select_indices(beam_idx)
        return past_key_values

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Cache | None = None,
        use_cache: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        use_cache = True if use_cache is None else use_cache
        cache_in = past_key_values if use_cache else None

        B, Q_LEN = input_ids.size()

        if cache_in is None and use_cache:
            cache_in = DynamicCache()

        embeds = self.txt_embed(input_ids)
        # Offset PE by cache length
        cache_len = 0
        if use_cache and cache_in is not None:
            cache_len = cache_in.get_seq_length()

        # print(f"{Q_LEN=}")
        # print(f"{cache_len=}")

        embeds = self.pos_encoding(embeds, cache_len)

        hidden_states = self.dec_adapter(embeds)

        hidden_states, kv_cache = self.decoder(
            hidden_states,
            attn_mask=attention_mask,
            is_causal=Q_LEN > 1 and attention_mask is None,
            kv_cache=cache_in,
            use_cache=use_cache,
        )

        hidden_states = F.linear(hidden_states, self.dec_adapter.weight.T)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=self.cfg.pad_token_id,
            )

        return CausalLMOutputWithPast(
            loss=loss, logits=logits, past_key_values=kv_cache # type: ignore
        )

    def _init_weights(self, module):
        std = 0.02
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=1 / math.sqrt(self.cfg.embed_dim))
            module.weight.data[self.cfg.pad_token_id].zero_()
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        # Residual branch scaling
        scale = 1.0 / math.sqrt(2 * self.cfg.num_layers)
        for name, p in module.named_parameters():
            if name.endswith(
                ("self_attn.out_proj.weight", "ffn.w2.weight", "ffn.w3.weight")
            ):
                with torch.no_grad():
                    p.mul_(scale)

AutoModelForCausalLM.register(ModelConfig, GPTVForCausalLM)
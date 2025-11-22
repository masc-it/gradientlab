import math
from typing import Optional
import torch
from torch import nn

import torch.nn.functional as F

from transformers import GenerationMixin, PreTrainedModel, AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.cache_utils import Cache, DynamicCache

from gradientlab.experiments.exp20251118_0_lm_30m.modeling.attention import Attention
from gradientlab.experiments.exp20251118_0_lm_30m.modeling.model_cfg import ModelConfig
from gradientlab.experiments.exp20251118_0_lm_30m.modeling.transformer import TransformerEncoder

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


class GPTForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = ModelConfig
    supports_gradient_checkpointing = False

    def __init__(self, config: ModelConfig):
        super().__init__(config)

        self.cfg = config
        self.num_layers = config.num_layers
        self.vocab_size = config.vocab_size
        self.pad_token_id = config.pad_token_id

        self.txt_embed = nn.Embedding(
            config.vocab_size, config.hidden_dim, padding_idx=config.pad_token_id
        )
        self.pos_encoding = PositionalEncoding(
            config.hidden_dim, max_len=config.max_position_embeddings or 2048, dropout=config.dropout
        )

        self.decoder = TransformerEncoder(config)

        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        self.post_init()
        self._scale_residual_branches()

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
            self._tied_weights_keys = ["lm_head.weight"]

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
        elif not self.training:
            attention_mask = None

        #assert attention_mask is None
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

        hidden_states, kv_cache = self.decoder(
            embeds,
            attn_mask=attention_mask,
            is_causal=Q_LEN > 1 and attention_mask is None,
            kv_cache=cache_in,
            use_cache=use_cache,
        )

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
            loss=loss, logits=logits, past_key_values=kv_cache, hidden_states=embeds # type: ignore
        )

    def _init_weights(self, module):
        """Initialize weights for deep narrow models."""
        
        # 1. Embedding initialization
        if isinstance(module, nn.Embedding):
            std = 1.0 / math.sqrt(self.cfg.hidden_dim)
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if self.cfg.pad_token_id is not None:
                module.weight.data[self.cfg.pad_token_id].zero_()
        
        # 2. Linear layer initialization
        elif isinstance(module, nn.Linear):
            # Adjusted std for narrow models (256 vs typical 768)
            width_scale = math.sqrt(self.cfg.embed_dim / 768.0)
            std = 0.02 * width_scale  # ~0.0117 for 256-dim
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        
        # 3. LayerNorm initialization
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        
    def _scale_residual_branches(self):
        """
        Scale residual branch outputs for deep models.
        Called ONCE after base initialization to avoid multiple scaling.
        """
        scale = 1.0 / math.sqrt(2 * self.cfg.num_layers)  # ~0.15 for 22 layers
        
        residual_patterns = [
            'attn.out_proj.weight',
            'ffn.w3.weight',
        ]
        
        scaled_params = []
        for name, p in self.named_parameters():  # Use self, not module
            if 'weight' in name and any(pattern in name for pattern in residual_patterns):
                with torch.no_grad():
                    p.mul_(scale)
                scaled_params.append(name)
        
        # Log what was scaled
        print(f"\n{'='*60}")
        print(f"Residual Scaling: {scale:.6f} (1/sqrt(2×{self.cfg.num_layers}))")
        print(f"Scaled {len(scaled_params)} parameters:")
        for name in scaled_params[:10]:  # Show first 10
            print(f"  ✓ {name}")
        if len(scaled_params) > 10:
            print(f"  ... and {len(scaled_params) - 10} more")
        print(f"{'='*60}\n")

    def validate_initialization(self):
        """Check if initialization was applied correctly."""
        print("\n" + "="*70)
        print("INITIALIZATION VALIDATION")
        print("="*70)
        
        residual_outputs = []
        other_weights = []
        model = self

        for name, param in model.named_parameters():
            # Skip non-weight parameters
            if 'weight' not in name:
                continue
            
            std = param.std().item()
            
            # Check if this should be a scaled residual output
            is_residual = any(p in name for p in ['out_proj.weight', '.w3.weight'])
            
            if is_residual:
                residual_outputs.append((name, std))
            elif 'embed' not in name:
                other_weights.append((name, std))
        
        # Expected values
        base_std = 0.02 * math.sqrt(self.cfg.hidden_dim / 768.0)
        expected_residual_std = base_std / math.sqrt(2 * self.num_layers)
        
        print(f"\nExpected base std: {base_std:.6f}")
        print(f"Expected residual std: {expected_residual_std:.6f}")
        
        # Check residual outputs
        print(f"\n{'Residual branch outputs':}")
        print(f"{'─'*70}")
        for name, std in residual_outputs[:5]:
            status = "✓" if 0.0010 < std < 0.0030 else "✗"
            print(f"  {status} {name:45s} std={std:.6f}")
        
        # Check other weights
        print(f"\n{'Other weights':}")
        print(f"{'─'*70}")
        for name, std in other_weights[:5]:
            status = "✓" if 0.008 < std < 0.015 else "✗"
            print(f"  {status} {name:45s} std={std:.6f}")
        
        # Summary checks
        print(f"\n{'SUMMARY':}")
        print(f"{'─'*70}")
        
        checks_passed = 0
        total_checks = 0
        
        # Check 2: Residual scaling applied
        if residual_outputs:
            total_checks += 1
            avg_residual = sum(s for _, s in residual_outputs) / len(residual_outputs)
            if 0.0010 < avg_residual < 0.0030:
                print(f"  ✓ Residual scaling applied (avg std={avg_residual:.6f})")
                checks_passed += 1
            else:
                print(f"  ✗ Residual scaling issue (avg std={avg_residual:.6f})")
        
        # Check 3: Base initialization reasonable
        if other_weights:
            total_checks += 1
            avg_other = sum(s for _, s in other_weights) / len(other_weights)
            if 0.008 < avg_other < 0.020:
                print(f"  ✓ Base initialization correct (avg std={avg_other:.6f})")
                checks_passed += 1
            else:
                print(f"  ✗ Base initialization issue (avg std={avg_other:.6f})")
        
        # Check 4: Scaling ratio
        if residual_outputs and other_weights:
            total_checks += 1
            avg_residual = sum(s for _, s in residual_outputs) / len(residual_outputs)
            avg_other = sum(s for _, s in other_weights) / len(other_weights)
            ratio = avg_other / avg_residual
            expected_ratio = math.sqrt(2 * self.num_layers)
            if abs(ratio - expected_ratio) < 1.5:
                print(f"  ✓ Scaling ratio correct ({ratio:.2f} vs {expected_ratio:.2f})")
                checks_passed += 1
            else:
                print(f"  ✗ Scaling ratio off ({ratio:.2f} vs {expected_ratio:.2f})")
        
        print(f"\n{'─'*70}")
        print(f"  Checks passed: {checks_passed}/{total_checks}")
        if checks_passed == total_checks:
            print("  All initialization checks passed!")
        else:
            print("  Some checks failed - review initialization")
        print("="*70 + "\n")
AutoModelForCausalLM.register(ModelConfig, GPTForCausalLM)
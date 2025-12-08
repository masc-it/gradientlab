from typing import List, Optional, Dict
from transformers import PreTrainedTokenizer, Qwen3Config, Qwen3ForCausalLM
import os
import json

# =========================================================
# Byte-level tokenizer (256 bytes + special tokens)
# =========================================================


class ByteLevelTokenizer(PreTrainedTokenizer):
    """
    Very simple byte-level tokenizer:

    - Vocabulary:
        0: <pad>
        1: <unk>
        2: <bos>
        3: <eos>
        4..259: <b0> .. <b255>  (one token per byte value)
    - Encoding: UTF-8 bytes, each byte -> <bX> token.
    - Decoding: inverse mapping back to bytes (best-effort).
    """

    def __init__(self, **kwargs):
        # Define special tokens
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"

        # Build vocab
        vocab: Dict[str, int] = {}
        vocab[self.pad_token] = 0
        vocab[self.unk_token] = 1
        vocab[self.bos_token] = 2
        vocab[self.eos_token] = 3

        # One token per possible byte
        self._byte_tokens: Dict[int, str] = {}
        for b in range(256):
            tok = f"<b{b}>"
            idx = len(vocab)
            vocab[tok] = idx
            self._byte_tokens[b] = tok

        self._vocab = vocab
        self._ids_to_tokens = {i: t for t, i in vocab.items()}

        super().__init__(
            bos_token=self.bos_token,
            eos_token=self.eos_token,
            unk_token=self.unk_token,
            pad_token=self.pad_token,
            **kwargs,
        )

        # Set some useful attributes
        self.model_max_length = 4096
        self._decode_errors = "replace"

    # ----- Required API -----

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    def get_vocab(self) -> Dict[str, int]:
        return dict(self._vocab)

    def _tokenize(self, text: str) -> List[str]:
        """
        Convert text -> UTF-8 bytes -> <bX> tokens.
        """
        b = text.encode("utf-8", errors="replace")
        return [self._byte_tokens[byte] for byte in b]

    def _convert_token_to_id(self, token: str) -> int:
        return self._vocab.get(token, self._vocab[self.unk_token])

    def _convert_id_to_token(self, index: int) -> str:
        return self._ids_to_tokens.get(index, self.unk_token)

    def build_inputs_with_special_tokens(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
    ) -> List[int]:
        # Simple [BOS] ... [EOS] wrapper
        if token_ids_1 is not None:
            # We don't really use pair sequences here, but you can extend if needed
            return (
                [self.bos_token_id]
                + token_ids_0
                + [self.eos_token_id]
                + token_ids_1
                + [self.eos_token_id]
            )
        return [self.bos_token_id] + token_ids_0 + [self.eos_token_id]

    def save_vocabulary(
        self, save_directory: str, filename_prefix: Optional[str] = None
    ):
        """
        Minimal implementation so tokenizer.save_pretrained() works.
        This writes a vocab.json file with token -> id mapping.
        """

        if not os.path.isdir(save_directory):
            os.makedirs(save_directory, exist_ok=True)
        vocab_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "") + "vocab.json",
        )
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(self._vocab, f, ensure_ascii=False)
        return (vocab_file,)

    # Optional: decode for debugging
    def _decode_bytes(self, tokens: List[str]) -> bytes:
        out_bytes = []
        for tok in tokens:
            if tok in {self.pad_token, self.unk_token, self.bos_token, self.eos_token}:
                continue
            if tok.startswith("<b") and tok.endswith(">"):
                try:
                    b_val = int(tok[2:-1])
                    if 0 <= b_val <= 255:
                        out_bytes.append(b_val)
                except ValueError:
                    continue
        return bytes(out_bytes)

    def _decode(self, token_ids, skip_special_tokens=False, **kwargs) -> str:
        tokens = [self._convert_id_to_token(int(i)) for i in token_ids]
        if skip_special_tokens:
            tokens = [
                t
                for t in tokens
                if t
                not in {self.pad_token, self.unk_token, self.bos_token, self.eos_token}
            ]
        b = self._decode_bytes(tokens)
        return b.decode("utf-8", errors=self._decode_errors)


# =========================================================
# Qwen3 ~10M config + model
# =========================================================


def init_qwen3_10m_bytelevel(
    max_position_embeddings: int = 4096,
) -> tuple[Qwen3ForCausalLM, ByteLevelTokenizer]:
    """
    Initialize a tiny Qwen3ForCausalLM (~10M params) with a byte-level tokenizer.

    - hidden_size = 320
    - num_hidden_layers = 8
    - num_attention_heads = 8
    - intermediate_size = 4 * hidden_size = 1280

    Rough param count (excluding a few biases/norms):
        embeddings: vocab_size * hidden_size
        + 8 layers of attention + MLP
        + LM head
      ≈ 10M params total.
    """

    tokenizer = ByteLevelTokenizer()
    vocab_size = tokenizer.vocab_size  # 260 = 4 specials + 256 bytes

    hidden_size = 320
    num_layers = 8
    num_heads = 8
    head_dim = hidden_size // num_heads
    intermediate_size = 4 * hidden_size  # simple choice

    config = Qwen3Config(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        num_key_value_heads=num_heads // 2,  # small GQA
        head_dim=head_dim,
        max_position_embeddings=max_position_embeddings,
        attention_bias=False,
        use_sliding_window=False,
        sliding_window=None,
        max_window_layers=num_layers,
        tie_word_embeddings=False,  # LM head separate (adds ~0.08M params)
        attention_dropout=0.0,
        hidden_act="silu",
    )

    # Make sure special token IDs line up
    config.bos_token_id = tokenizer.bos_token_id
    config.eos_token_id = tokenizer.eos_token_id
    config.pad_token_id = tokenizer.pad_token_id

    model = Qwen3ForCausalLM(config)

    # (Optional) print parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"Initialized Qwen3 ~10M model: {total_params / 1e6:.2f}M parameters, "
        f"vocab_size={vocab_size}"
    )

    return model, tokenizer


if __name__ == "__main__":
    model, tokenizer = init_qwen3_10m_bytelevel()

    # Quick sanity check
    text = ["coins: 1,2,5\namount: 11\n\n", "coins: 1,2,6\namountt: 11\n\n"]
    enc = tokenizer(text, return_tensors="pt", padding_side="left", padding="longest")
    print(enc)
    out = model(**enc)
    print("Input shape:", enc["input_ids"].shape)
    print("Logits shape:", out.logits.shape)
    out = model.generate(enc["input_ids"])
    print(out.shape)
    print(tokenizer.decode(out[0]))
    print(tokenizer.decode(out[1]))

from pydantic import BaseModel


class ModelConfig(BaseModel):
    enc_num_layers: int = 16
    enc_dropout: float = 0.05
    enc_attn_dropout: float = 0.01

    vocab_size: int = 512
    bos_token_id: int
    pad_token_id: int = 0
    eos_token_id: int = 1

    embed_dim: int = 64
    hidden_dim: int = 256
    hidden_squeeze_ratio: float = 0.5
    max_len: int = 4096
    use_bias: bool = True
    num_heads: int = 4
    num_kv_groups: int = 2

    ffn_mult: int = 4

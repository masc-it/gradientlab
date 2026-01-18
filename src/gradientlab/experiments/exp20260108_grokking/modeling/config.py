from pydantic import BaseModel


class ModelConfig(BaseModel):
    vocab_size: int
    d_model: int = 256
    n_layers: int = 2
    num_heads: int = 4  # 256 / 4 = 64 dim per head
    dropout: float = 0.1
    ffn_mult: float = 4.0
    use_alibi: bool = True
    alibi_max_positions: int = 4096
    pad_token_id: int

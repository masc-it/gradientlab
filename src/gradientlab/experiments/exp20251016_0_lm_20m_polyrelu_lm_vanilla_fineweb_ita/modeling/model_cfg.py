from transformers import PretrainedConfig


class ModelConfig(PretrainedConfig):
    model_type = "custom-gpt"

    def __init__(
        self,
        num_layers: int = 16,
        dropout: float = 0.05,
        attn_dropout: float = 0.0,
        vocab_size: int = 512,
        bos_token_id: int = 0,
        pad_token_id: int = 1,
        eos_token_id: int = 2,
        embed_dim: int = 64,
        hidden_dim: int = 256,
        hidden_squeeze_ratio: float = 0.5,
        max_length: int = 4096,
        use_bias: bool = True,
        num_heads: int = 4,
        num_kv_groups: int = 2,
        ffn_mult: int = 4,
        tie_word_embeddings: bool = True,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            num_hidden_layers=num_layers,
            max_length=max_length,
        )

        self.num_layers = num_layers
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.vocab_size = vocab_size
        self.bos_token_id = bos_token_id
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.hidden_squeeze_ratio = hidden_squeeze_ratio
        self.max_length = max_length
        self.use_bias = use_bias
        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.ffn_mult = ffn_mult
        self.tie_word_embeddings = tie_word_embeddings

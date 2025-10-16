from transformers.models.byt5 import ByT5Tokenizer
from tokenizers import AddedToken


def byte_tokenizer():
    special_tokens_str = [
        "<|im_start|>",
        "<|im_end|>",
        "<|im_dec_start|>",
        "<|im_dec_end|>",
        "<|img_start|>",
        "<|img_end|>",
        "<|attn_sink|>",
        *[f"<|extra_{i}|>" for i in range(509 - 256 - 7)],
    ]

    special_tokens = [
        AddedToken(tok, lstrip=True, rstrip=True, special=False)
        for tok in special_tokens_str
    ]
    tok = ByT5Tokenizer(
        additional_special_tokens=special_tokens,
        extra_ids=0,
    )

    return tok

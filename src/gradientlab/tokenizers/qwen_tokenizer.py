from transformers import AutoTokenizer, Qwen2TokenizerFast


def qwen3_tokenizer() -> Qwen2TokenizerFast:
    return AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

from transformers import AutoTokenizer


def qwen3_tokenizer():
    return AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

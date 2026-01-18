"""
Test script for batched greedy generation.
"""

import torch
from datasets import load_from_disk
from gradientlab.tokenizers.byte_tokenizer import byte_tokenizer
from gradientlab.experiments.exp20260108_grokking.modeling.factory import ModelFactory


def test_batched_generation():
    """Test the generate function with batched inputs."""

    print("=" * 80)
    print("BATCHED GENERATION TEST")
    print("=" * 80)

    # Load model
    checkpoint_path = "src/gradientlab/experiments/exp20260108_grokking/data/model"
    model, tokenizer, config = ModelFactory.build_grokking_model(resume_from=checkpoint_path)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    model.eval()

    print(f"Device: {device}")
    print(f"Model loaded from: {checkpoint_path}")
    print()

    # Load dataset
    ds = load_from_disk("src/gradientlab/experiments/exp20260108_grokking/data/date_to_iso")

    # Get a batch of test examples
    test_samples = [
        ds['train'][i] for i in range(5)
    ]

    print("Test samples:")
    for i, sample in enumerate(test_samples):
        print(f"  {i+1}. {sample['input']} -> {sample['target']}")
    print()

    # Format as prompts (only input, no target)
    prompts = [f"<|im_start|>{sample['input']} >" for sample in test_samples]

    print("Prompts:")
    for i, prompt in enumerate(prompts):
        print(f"  {i+1}. {prompt!r}")
    print()

    # Tokenize prompts
    encoded = tokenizer(
        prompts,
        padding="longest",
        padding_side="left",
        return_tensors="pt",
        add_special_tokens=False,
    )

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    print(f"Tokenized batch shape: {input_ids.shape}")
    print()

    # Generate
    print("=" * 80)
    print("GENERATING...")
    print("=" * 80)

    # Get EOS token ID
    eos_token_id = tokenizer("<|im_end|>", add_special_tokens=False)["input_ids"][0]
    print(f"EOS token ID: {eos_token_id}")
    print()

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=15,  # Enough for " YYYY-MM-DD <|im_end|>"
            eos_token_id=eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    print(f"Generated shape: {generated_ids.shape}")
    print()

    # Decode results
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)

    for i in range(len(prompts)):
        # Get the generated tokens (exclude padding)
        gen_ids = generated_ids[i]
        attn_mask = (gen_ids != tokenizer.pad_token_id)
        real_tokens = gen_ids[attn_mask]

        # Decode
        generated_text = tokenizer.decode(real_tokens.tolist())
        expected = test_samples[i]['target']

        # Extract just the prediction (after ">")
        if " > " in generated_text:
            parts = generated_text.split(" > ", 1)
            prediction = parts[1] if len(parts) > 1 else ""
        else:
            prediction = generated_text

        # Remove <|im_end|> if present
        if "<|im_end|>" in prediction:
            prediction = prediction.replace("<|im_end|>", "").strip()

        match = "✓" if prediction == expected else "✗"

        print(f"\n{i+1}. {match}")
        print(f"   Input:      {test_samples[i]['input']!r}")
        print(f"   Expected:   {expected!r}")
        print(f"   Generated:  {prediction!r}")
        print(f"   Full text:  {generated_text!r}")

    print()
    print("=" * 80)


if __name__ == "__main__":
    test_batched_generation()

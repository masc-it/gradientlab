"""
Debug script to inspect the dataset and label creation.
"""

import torch
from datasets import load_from_disk
from gradientlab.tokenizers.byte_tokenizer import byte_tokenizer
from gradientlab.experiments.exp20260108_grokking.dataset.torch_dataset import DateCollate, DateToISODataset


def inspect_dataset():
    """Inspect the dataset and label creation."""

    # Load dataset
    ds = load_from_disk("src/gradientlab/experiments/exp20260108_grokking/data/date_to_iso")

    print("=" * 60)
    print("DATASET INSPECTION")
    print("=" * 60)
    print(f"Train size: {len(ds['train'])}")
    print(f"Test size: {len(ds['test'])}")
    print()

    # Show a few examples
    print("Sample raw data:")
    for i in range(3):
        sample = ds['train'][i]
        print(f"  {i+1}. Input: {sample['input']!r} -> Target: {sample['target']!r}")
    print()

    # Create dataset and collate
    tokenizer = byte_tokenizer()
    collate = DateCollate(tokenizer)
    torch_ds = DateToISODataset(ds['train'])

    print("=" * 60)
    print("FORMATTED SAMPLES (with special tokens)")
    print("=" * 60)
    for i in range(3):
        formatted = torch_ds[i]
        print(f"{i+1}. {formatted}")
    print()

    # Create a batch and inspect tokenization + labels
    print("=" * 60)
    print("BATCH INSPECTION")
    print("=" * 60)

    samples = [torch_ds[i] for i in range(3)]
    batch = collate(samples)

    print(f"Batch shapes:")
    print(f"  input_ids: {batch['input_ids'].shape}")
    print(f"  attention_mask: {batch['attention_mask'].shape}")
    print(f"  labels: {batch['labels'].shape}")
    print()

    # Decode and inspect each sample
    for i in range(3):
        print(f"\nSample {i+1}:")
        input_ids = batch['input_ids'][i]
        labels = batch['labels'][i]
        attention_mask = batch['attention_mask'][i]

        # Find real tokens (non-padding)
        real_mask = attention_mask == 1
        real_input_ids = input_ids[real_mask]
        real_labels = labels[real_mask]

        # Decode
        input_text = tokenizer.decode(real_input_ids.tolist())
        print(f"  Input text: {input_text!r}")

        # Find separator position
        sep_token_id = collate.sep_token_id
        sep_positions = (real_input_ids == sep_token_id).nonzero(as_tuple=True)[0]
        if len(sep_positions) > 0:
            sep_idx = sep_positions[0].item()
            print(f"  Separator '>' at position: {sep_idx}")

            # Show what's masked vs not masked
            prompt_ids = real_input_ids[:sep_idx + 1]
            target_ids = real_input_ids[sep_idx + 1:]

            prompt_labels = real_labels[:sep_idx + 1]
            target_labels = real_labels[sep_idx + 1:]

            print(f"  Prompt ({len(prompt_ids)} tokens): {tokenizer.decode(prompt_ids.tolist())!r}")
            print(f"  Target ({len(target_ids)} tokens): {tokenizer.decode(target_ids.tolist())!r}")
            print(f"  Prompt labels all masked: {(prompt_labels == tokenizer.pad_token_id).all().item()}")
            print(f"  Target labels masked: {(target_labels == tokenizer.pad_token_id).sum().item()}/{len(target_labels)}")

            # Count target tokens (should be used for accuracy)
            target_count = (target_labels != tokenizer.pad_token_id).sum().item()
            print(f"  Target tokens for loss/accuracy: {target_count}")


if __name__ == "__main__":
    inspect_dataset()

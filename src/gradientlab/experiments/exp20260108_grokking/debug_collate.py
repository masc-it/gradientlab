"""
Debug script to inspect collate function and decoded labels.
"""

import torch
from datasets import load_from_disk
from gradientlab.tokenizers.byte_tokenizer import byte_tokenizer
from gradientlab.experiments.exp20260108_grokking.dataset.torch_dataset import DateCollate, DateToISODataset


def debug_collate():
    """Debug the collate function and print decoded labels."""

    # Load dataset
    ds = load_from_disk("src/gradientlab/experiments/exp20260108_grokking/data/date_to_iso")
    torch_ds = DateToISODataset(ds['train'])

    # Create tokenizer and collate
    tokenizer = byte_tokenizer()
    collate = DateCollate(tokenizer)

    print("=" * 80)
    print("COLLATE DEBUG - DECODED LABELS")
    print("=" * 80)
    print(f"Pad token ID: {tokenizer.pad_token_id}")
    print(f"Separator token ID: {collate.sep_token_id}")
    print()

    # Get a few samples
    samples = [torch_ds[i] for i in range(5)]

    print("Raw formatted samples:")
    for i, sample in enumerate(samples):
        print(f"  {i+1}. {sample}")
    print()

    # Collate
    batch = collate(samples)

    print("=" * 80)
    print("AFTER COLLATION")
    print("=" * 80)
    print(f"Batch shapes:")
    print(f"  input_ids: {batch['input_ids'].shape}")
    print(f"  attention_mask: {batch['attention_mask'].shape}")
    print(f"  labels: {batch['labels'].shape}")
    print()

    # Decode each sample
    for i in range(len(samples)):
        print("=" * 80)
        print(f"SAMPLE {i+1}")
        print("=" * 80)

        input_ids = batch['input_ids'][i]
        labels = batch['labels'][i]
        attention_mask = batch['attention_mask'][i]

        # Decode full input
        input_text = tokenizer.decode(input_ids.tolist())
        print(f"Input (full sequence):")
        print(f"  {input_text!r}")
        print()

        # Show token-by-token breakdown
        print(f"Token-by-token breakdown:")
        print(f"  {'Pos':<4} {'Attn':<5} {'Input Token':<20} {'Label Token':<20} {'Masked':<7}")
        print(f"  {'-'*4} {'-'*5} {'-'*20} {'-'*20} {'-'*7}")

        for pos in range(len(input_ids)):
            attn = attention_mask[pos].item()
            input_tok_id = input_ids[pos].item()
            label_tok_id = labels[pos].item()

            input_tok = tokenizer.decode([input_tok_id])
            label_tok = tokenizer.decode([label_tok_id]) if label_tok_id != tokenizer.pad_token_id else "<PAD>"

            is_masked = label_tok_id == tokenizer.pad_token_id

            # Only print real tokens (non-padding)
            if attn == 1:
                print(f"  {pos:<4} {attn:<5} {input_tok!r:<20} {label_tok!r:<20} {'Yes' if is_masked else 'No':<7}")

        print()

        # Decode only non-masked labels
        non_masked_labels = labels[labels != tokenizer.pad_token_id]
        if len(non_masked_labels) > 0:
            decoded_labels = tokenizer.decode(non_masked_labels.tolist())
            print(f"Decoded labels (only non-masked):")
            print(f"  {decoded_labels!r}")
            print(f"  Length: {len(non_masked_labels)} tokens")
        else:
            print(f"Decoded labels: ALL MASKED")
        print()

        # Find separator and show prompt vs target
        sep_positions = (input_ids == collate.sep_token_id).nonzero(as_tuple=True)[0]
        if len(sep_positions) > 0:
            sep_idx = sep_positions[0].item()

            # Only consider real tokens (attention_mask == 1)
            real_mask = attention_mask == 1
            real_positions = real_mask.nonzero(as_tuple=True)[0]

            # Find separator in real tokens
            real_input_ids = input_ids[real_mask]
            real_labels = labels[real_mask]

            sep_in_real = (real_input_ids == collate.sep_token_id).nonzero(as_tuple=True)[0]
            if len(sep_in_real) > 0:
                sep_real_idx = sep_in_real[0].item()

                prompt_ids = real_input_ids[:sep_real_idx + 1]
                target_ids = real_input_ids[sep_real_idx + 1:]

                prompt_labels = real_labels[:sep_real_idx + 1]
                target_labels = real_labels[sep_real_idx + 1:]

                print(f"Split at separator position (in real tokens): {sep_real_idx}")
                print(f"Prompt: {tokenizer.decode(prompt_ids.tolist())!r}")
                print(f"Target input: {tokenizer.decode(target_ids.tolist())!r}")
                print(f"Target labels (decoded): {tokenizer.decode(target_labels[target_labels != tokenizer.pad_token_id].tolist())!r}")
                print(f"Prompt masked: {(prompt_labels == tokenizer.pad_token_id).all().item()}")
                print(f"Target masked count: {(target_labels == tokenizer.pad_token_id).sum().item()}/{len(target_labels)}")
        print()


if __name__ == "__main__":
    debug_collate()

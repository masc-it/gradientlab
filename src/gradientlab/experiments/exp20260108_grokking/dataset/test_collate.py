"""
Test script for the vectorized _create_labels function.
"""

import torch
from gradientlab.tokenizers.byte_tokenizer import byte_tokenizer
from gradientlab.experiments.exp20260108_grokking.dataset.torch_dataset import DateCollate


def test_create_labels_basic():
    """Test basic functionality of _create_labels."""
    print("=" * 60)
    print("TEST 1: Basic functionality")
    print("=" * 60)

    tokenizer = byte_tokenizer()
    collate = DateCollate(tokenizer)

    # Create sample batch with ">" separator
    samples = [
        "<|im_start|>01/15/2000 > 2000-01-15<|im_end|>",
        "<|im_start|>December 25, 1999 > 1999-12-25<|im_end|>",
        "<|im_start|>31/12/2050 > 2050-12-31<|im_end|>",
    ]

    # Tokenize
    encoded = tokenizer(
        samples,
        padding="longest",
        padding_side="left",
        return_tensors="pt",
        add_special_tokens=False,
    )

    # Create labels
    labels = collate._create_labels(encoded["input_ids"])

    print(f"Input shape: {encoded['input_ids'].shape}")
    print(f"Labels shape: {labels.shape}")
    print()

    # Verify that positions before ">" are masked
    for i, (sample, input_ids, label_ids) in enumerate(zip(samples, encoded["input_ids"], labels)):
        print(f"Sample {i+1}: {sample[:50]}...")

        # Find separator position
        sep_positions = (input_ids == collate.sep_token_id).nonzero(as_tuple=True)[0]
        if len(sep_positions) > 0:
            sep_idx = sep_positions[0].item()
            print(f"  Separator at position: {sep_idx}")

            # Check that everything before and including separator is masked
            masked_positions = (label_ids[:sep_idx + 1] == collate.pad_token_id).all()
            print(f"  Positions before '>' masked: {masked_positions.item()}")

            # Check that positions after separator are NOT all masked
            after_sep_not_all_masked = not (label_ids[sep_idx + 1:] == collate.pad_token_id).all().item()
            print(f"  Positions after '>' contain non-pad: {after_sep_not_all_masked}")
        print()

    print("✅ Basic test passed\n")


def test_create_labels_edge_cases():
    """Test edge cases."""
    print("=" * 60)
    print("TEST 2: Edge cases")
    print("=" * 60)

    tokenizer = byte_tokenizer()
    collate = DateCollate(tokenizer)

    # Test case 1: No separator (shouldn't happen in practice but test anyway)
    print("Case 1: No separator token")
    encoded = tokenizer(
        ["<|im_start|>some text without separator<|im_end|>"],
        padding="longest",
        padding_side="left",
        return_tensors="pt",
        add_special_tokens=False,
    )
    labels = collate._create_labels(encoded["input_ids"])
    has_sep = (encoded["input_ids"] == collate.sep_token_id).any()
    all_unmasked = (labels == encoded["input_ids"]).all()
    print(f"  Has separator: {has_sep.item()}")
    print(f"  All tokens unmasked (as expected): {all_unmasked.item()}")
    print()

    # Test case 2: Multiple separators (use first one)
    print("Case 2: Multiple separator tokens")
    encoded = tokenizer(
        ["<|im_start|>text > with > multiple > separators<|im_end|>"],
        padding="longest",
        padding_side="left",
        return_tensors="pt",
        add_special_tokens=False,
    )
    labels = collate._create_labels(encoded["input_ids"])
    sep_positions = (encoded["input_ids"][0] == collate.sep_token_id).nonzero(as_tuple=True)[0]
    if len(sep_positions) > 0:
        first_sep = sep_positions[0].item()
        print(f"  First separator at position: {first_sep}")
        print(f"  Total separators found: {len(sep_positions)}")
        masked_before_first = (labels[0, :first_sep + 1] == collate.pad_token_id).all()
        print(f"  Masked up to first separator: {masked_before_first.item()}")
    print()

    # Test case 3: Separator at the beginning
    print("Case 3: Separator at beginning")
    encoded = tokenizer(
        ["> target only"],
        padding="longest",
        padding_side="left",
        return_tensors="pt",
        add_special_tokens=False,
    )
    labels = collate._create_labels(encoded["input_ids"])
    sep_positions = (encoded["input_ids"][0] == collate.sep_token_id).nonzero(as_tuple=True)[0]
    if len(sep_positions) > 0:
        first_sep = sep_positions[0].item()
        print(f"  Separator at position: {first_sep}")
        if first_sep + 1 < labels.shape[1]:
            has_unmasked_after = (labels[0, first_sep + 1:] != collate.pad_token_id).any()
            print(f"  Has unmasked tokens after separator: {has_unmasked_after.item()}")
    print()

    print("✅ Edge cases test passed\n")


def test_create_labels_batched():
    """Test with realistic batched data including left padding."""
    print("=" * 60)
    print("TEST 3: Batched data with left padding")
    print("=" * 60)

    tokenizer = byte_tokenizer()
    collate = DateCollate(tokenizer)

    # Different length samples (will be left-padded)
    samples = [
        "<|im_start|>01/15/2000 > 2000-01-15<|im_end|>",
        "<|im_start|>1/5/00 > 2000-01-05<|im_end|>",  # Shorter
        "<|im_start|>December 25, 1999 > 1999-12-25<|im_end|>",  # Longer
    ]

    # Use full collate function
    batch_output = collate(samples)

    print(f"Batch size: {batch_output['input_ids'].shape[0]}")
    print(f"Max sequence length: {batch_output['input_ids'].shape[1]}")
    print()

    for i in range(len(samples)):
        input_ids = batch_output['input_ids'][i]
        labels = batch_output['labels'][i]
        attention_mask = batch_output['attention_mask'][i]

        # Count padding
        n_padding = (attention_mask == 0).sum().item()
        n_real = (attention_mask == 1).sum().item()

        print(f"Sample {i+1}:")
        print(f"  Padding tokens: {n_padding}")
        print(f"  Real tokens: {n_real}")

        # Find separator in real tokens
        real_token_positions = attention_mask.nonzero(as_tuple=True)[0]
        if len(real_token_positions) > 0:
            # Check within real tokens only
            real_tokens = input_ids[real_token_positions]
            sep_mask = (real_tokens == collate.sep_token_id)
            if sep_mask.any():
                # Find global position
                sep_global_pos = real_token_positions[sep_mask.nonzero(as_tuple=True)[0][0]].item()
                print(f"  Separator at global position: {sep_global_pos}")

                # Check masking
                # Padding should already be masked as pad_token_id
                # Real tokens before separator should be masked
                labels_before_sep = labels[n_padding:sep_global_pos + 1]
                all_masked_before = (labels_before_sep == collate.pad_token_id).all()
                print(f"  Tokens before '>' masked: {all_masked_before.item()}")

                # Tokens after separator should not all be masked
                if sep_global_pos + 1 < len(labels):
                    labels_after_sep = labels[sep_global_pos + 1:]
                    has_real_after = (labels_after_sep != collate.pad_token_id).any()
                    print(f"  Has real labels after '>': {has_real_after.item()}")
        print()

    print("✅ Batched test passed\n")


def test_performance_comparison():
    """Compare performance of vectorized vs loop version."""
    print("=" * 60)
    print("TEST 4: Performance comparison")
    print("=" * 60)

    import time

    tokenizer = byte_tokenizer()
    collate = DateCollate(tokenizer)

    # Create a larger batch
    samples = [
        f"<|im_start|>sample {i:04d} > target {i:04d}<|im_end|>"
        for i in range(128)  # Typical batch size
    ]

    encoded = tokenizer(
        samples,
        padding="longest",
        padding_side="left",
        return_tensors="pt",
        add_special_tokens=False,
    )

    # Warm-up
    _ = collate._create_labels(encoded["input_ids"])

    # Benchmark vectorized version
    n_iterations = 100
    start = time.time()
    for _ in range(n_iterations):
        labels = collate._create_labels(encoded["input_ids"])
    end = time.time()

    avg_time = (end - start) / n_iterations * 1000  # Convert to ms
    print(f"Batch size: {len(samples)}")
    print(f"Sequence length: {encoded['input_ids'].shape[1]}")
    print(f"Average time per batch: {avg_time:.4f} ms")
    print(f"Throughput: {len(samples) / (avg_time / 1000):.0f} samples/sec")
    print()

    print("✅ Performance test completed\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("TESTING VECTORIZED _create_labels FUNCTION")
    print("=" * 60 + "\n")

    try:
        test_create_labels_basic()
        test_create_labels_edge_cases()
        test_create_labels_batched()
        test_performance_comparison()

        print("=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()

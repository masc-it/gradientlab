"""
Test that the loss computation fix works correctly.
"""

import torch
from datasets import load_from_disk

from gradientlab.experiments.exp20260108_grokking.modeling.factory import ModelFactory
from gradientlab.experiments.exp20260108_grokking.dataset.torch_dataset import DateCollate


def test_loss_computation():
    """Test loss and accuracy computation after fix."""

    print("=" * 80)
    print("TESTING LOSS COMPUTATION FIX")
    print("=" * 80)

    # Build a FRESH model (not from checkpoint)
    model, tokenizer, config = ModelFactory.build_grokking_model(resume_from=None)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    model.eval()

    print(f"Device: {device}")
    print(f"Model vocab size: {config.vocab_size}")
    print()

    # Load dataset and create a single batch
    ds_path = "src/gradientlab/experiments/exp20260108_grokking/data/date_to_iso"
    ds = load_from_disk(ds_path)

    # Get a few training samples
    samples = [
        f"<|im_start|>{ds['train'][i]['input']} > {ds['train'][i]['target']}<|im_end|>"
        for i in range(3)
    ]

    print("Test samples:")
    for i, sample in enumerate(samples):
        print(f"  {i+1}. {sample}")
    print()

    # Create batch
    collate = DateCollate(tokenizer)
    batch = collate(samples)

    # Move to device
    batch = {k: v.to(device) for k, v in batch.items()}

    print(f"Batch shape:")
    print(f"  input_ids: {batch['input_ids'].shape}")
    print(f"  attention_mask: {batch['attention_mask'].shape}")
    print(f"  labels: {batch['labels'].shape}")
    print()

    # Forward pass
    with torch.no_grad():
        output = model(**batch)
        loss = output["loss"]
        logits = output["logits"]

    print(f"Loss: {loss.item():.4f}")
    print(f"Logits shape: {logits.shape}")
    print()

    # Check accuracy computation
    print("=" * 80)
    print("CHECKING ACCURACY COMPUTATION")
    print("=" * 80)

    # Shift for next-token prediction
    predictions = logits[:, :-1, :].argmax(dim=-1)
    shift_labels = batch["labels"][:, 1:]
    mask = shift_labels != tokenizer.pad_token_id

    print(f"Predictions shape: {predictions.shape}")
    print(f"Shift labels shape: {shift_labels.shape}")
    print(f"Mask shape: {mask.shape}")
    print()

    # Compute accuracy
    correct = ((predictions == shift_labels) & mask).sum().item()
    total = mask.sum().item()
    accuracy = correct / total if total > 0 else 0.0

    print(f"Correct: {correct}/{total}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print()

    # Show token-by-token for first sample
    print("=" * 80)
    print("FIRST SAMPLE DETAILED BREAKDOWN")
    print("=" * 80)

    sample_idx = 0
    sample_input_ids = batch["input_ids"][sample_idx]
    sample_labels = batch["labels"][sample_idx]
    sample_predictions = predictions[sample_idx]
    sample_shift_labels = shift_labels[sample_idx]
    sample_mask = mask[sample_idx]

    print(f"Input text: {samples[sample_idx]}")
    print()

    print(f"Input IDs: {sample_input_ids.tolist()}")
    print(f"Labels:    {sample_labels.tolist()}")
    print()

    # Show predictions for non-masked positions
    valid_positions = sample_mask.nonzero(as_tuple=True)[0]
    print(f"Valid prediction positions (after shifting): {len(valid_positions)}")
    print()

    print("Token-by-token (first 15 valid predictions):")
    print(f"{'Pos':>4} | {'Pred':>5} | {'Label':>5} | {'Pred Char':<10} | {'Label Char':<10} | {'Match'}")
    print("-" * 70)

    for i, pos in enumerate(valid_positions[:15]):
        pos_val = pos.item()
        pred = sample_predictions[pos_val].item()
        label = sample_shift_labels[pos_val].item()
        pred_char = tokenizer.decode([pred])
        label_char = tokenizer.decode([label])
        match = "✓" if pred == label else "✗"

        print(f"{pos_val:4d} | {pred:5d} | {label:5d} | {pred_char!r:<10} | {label_char!r:<10} | {match}")

    print()
    print("=" * 80)

    # Expected behavior
    print("\nEXPECTED BEHAVIOR:")
    print("- Fresh untrained model should have ~random predictions")
    print(f"- Accuracy should be ~1/{config.vocab_size} = {1/config.vocab_size:.4f}")
    print(f"- Actual accuracy: {accuracy:.4f}")
    print()

    if accuracy < 0.05:  # Less than 5% for random init
        print("✓ Accuracy looks reasonable for untrained model")
    else:
        print("✗ Accuracy suspiciously high for untrained model")

    print()
    print("=" * 80)


if __name__ == "__main__":
    test_loss_computation()

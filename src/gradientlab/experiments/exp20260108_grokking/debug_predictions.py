"""
Debug script to analyze which tokens are being predicted incorrectly.
"""

import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from gradientlab.tokenizers.byte_tokenizer import byte_tokenizer
from gradientlab.experiments.exp20260108_grokking.dataset.torch_dataset import DateCollate, DateToISODataset
from gradientlab.experiments.exp20260108_grokking.modeling.factory import ModelFactory


def analyze_predictions():
    """Analyze which tokens are consistently predicted incorrectly."""

    # Load model from checkpoint
    checkpoint_path = "src/gradientlab/experiments/exp20260108_grokking/data/model"
    print(f"Loading checkpoint from: {checkpoint_path}")
    model, tokenizer, config = ModelFactory.build_grokking_model(resume_from=checkpoint_path)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    model.eval()

    print("=" * 80)
    print("PREDICTION ANALYSIS")
    print("=" * 80)
    print(f"Device: {device}")
    print()

    # Load dataset
    ds = load_from_disk("src/gradientlab/experiments/exp20260108_grokking/data/date_to_iso")
    torch_ds = DateToISODataset(ds['train'])
    collate = DateCollate(tokenizer)

    # Create dataloader
    dataloader = DataLoader(
        torch_ds,
        batch_size=32,
        shuffle=False,
        collate_fn=collate,
    )

    # Analyze predictions per position
    position_errors = {}  # position -> count of errors
    total_per_position = {}  # position -> total samples

    total_correct = 0
    total_tokens = 0

    print("Analyzing predictions across training set...")

    # Store some example predictions
    example_predictions = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= 50:  # Check first 50 batches for better statistics
                break

            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass
            output = model(**batch)
            logits = output["logits"]

            # Get predictions
            predictions = logits.argmax(dim=-1)
            labels = batch["labels"]

            # Compute accuracy per position
            mask = labels != tokenizer.pad_token_id

            for i in range(labels.shape[0]):  # For each sample in batch
                sample_mask = mask[i]
                sample_preds = predictions[i]
                sample_labels = labels[i]

                # Only look at non-masked positions
                valid_positions = sample_mask.nonzero(as_tuple=True)[0]

                # Store first few examples
                if len(example_predictions) < 10:
                    valid_preds = sample_preds[valid_positions]
                    valid_labels = sample_labels[valid_positions]
                    pred_text = tokenizer.decode(valid_preds.tolist())
                    label_text = tokenizer.decode(valid_labels.tolist())
                    example_predictions.append((pred_text, label_text))

                for pos_idx, pos in enumerate(valid_positions):
                    pos_val = pos.item()
                    pred = sample_preds[pos].item()
                    label = sample_labels[pos].item()

                    if pos_idx not in position_errors:
                        position_errors[pos_idx] = 0
                        total_per_position[pos_idx] = 0

                    total_per_position[pos_idx] += 1

                    if pred != label:
                        position_errors[pos_idx] += 1

                    if pred == label:
                        total_correct += 1
                    total_tokens += 1

    overall_accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0

    print(f"Overall accuracy: {overall_accuracy:.4f} ({total_correct}/{total_tokens})")
    print()

    # Print per-position error rates
    print("=" * 80)
    print("ERROR RATES BY POSITION (in target sequence)")
    print("=" * 80)
    print(f"{'Pos':<5} {'Errors':<8} {'Total':<8} {'Error Rate':<12} {'Typical Token'}")
    print("-" * 80)

    # Get a sample to show what each position represents
    sample_batch = next(iter(DataLoader(torch_ds, batch_size=1, collate_fn=collate)))
    sample_labels = sample_batch['labels'][0]
    sample_mask = sample_labels != tokenizer.pad_token_id
    sample_label_ids = sample_labels[sample_mask]

    for pos_idx in sorted(position_errors.keys()):
        errors = position_errors[pos_idx]
        total = total_per_position[pos_idx]
        error_rate = errors / total if total > 0 else 0.0

        # Get typical token at this position
        if pos_idx < len(sample_label_ids):
            typical_token = tokenizer.decode([sample_label_ids[pos_idx].item()])
        else:
            typical_token = "?"

        marker = " ← HIGH ERROR" if error_rate > 0.5 else ""
        print(f"{pos_idx:<5} {errors:<8} {total:<8} {error_rate:<12.2%} {typical_token!r:<20}{marker}")

    print()
    print("=" * 80)
    print("Sample target structure:")
    print("  Position 0: ' ' (space)")
    print("  Positions 1-4: 'YYYY' (year)")
    print("  Position 5: '-'")
    print("  Positions 6-7: 'MM' (month)")
    print("  Position 8: '-'")
    print("  Positions 9-10: 'DD' (day)")
    print("  Position 11: '<|im_end|>'")
    print("=" * 80)

    # Show example predictions
    print()
    print("=" * 80)
    print("EXAMPLE PREDICTIONS")
    print("=" * 80)
    for i, (pred, label) in enumerate(example_predictions[:10]):
        match = "✓" if pred == label else "✗"
        print(f"{i+1}. {match}")
        print(f"   Predicted: {pred!r}")
        print(f"   Expected:  {label!r}")
        if pred != label:
            # Show character differences
            print(f"   Diff:      ", end="")
            for p_char, l_char in zip(pred, label):
                if p_char != l_char:
                    print(f"[{p_char!r}≠{l_char!r}]", end="")
                else:
                    print(p_char, end="")
            print()
        print()
    print("=" * 80)


if __name__ == "__main__":
    analyze_predictions()

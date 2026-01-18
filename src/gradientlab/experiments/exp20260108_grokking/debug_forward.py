"""
Debug script to inspect a forward pass and understand the loss/accuracy.
"""

import torch
from datasets import load_from_disk
from gradientlab.tokenizers.byte_tokenizer import byte_tokenizer
from gradientlab.experiments.exp20260108_grokking.dataset.torch_dataset import DateCollate, DateToISODataset
from gradientlab.experiments.exp20260108_grokking.modeling.factory import ModelFactory


def debug_forward_pass():
    """Run a forward pass and inspect loss and predictions."""

    # Load model and tokenizer
    model, tokenizer, config = ModelFactory.build_grokking_model()
    model.eval()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    print("=" * 60)
    print("FORWARD PASS DEBUG")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Model config: {config.model_dump_json(indent=2)}")
    print()

    # Load dataset
    ds = load_from_disk("src/gradientlab/experiments/exp20260108_grokking/data/date_to_iso")
    torch_ds = DateToISODataset(ds['train'])
    collate = DateCollate(tokenizer)

    # Create a small batch
    batch_size = 4
    samples = [torch_ds[i] for i in range(batch_size)]
    batch = collate(samples)

    # Move to device
    batch = {k: v.to(device) for k, v in batch.items()}

    print(f"Batch size: {batch_size}")
    print(f"Sequence length: {batch['input_ids'].shape[1]}")
    print()

    # Forward pass
    with torch.no_grad():
        output = model(**batch)
        loss = output["loss"]
        logits = output["logits"]

    print(f"Loss: {loss.item():.6f}")
    print(f"Loss shape: {loss.shape}")
    print(f"Logits shape: {logits.shape}")
    print()

    # Compute predictions and accuracy
    predictions = logits.argmax(dim=-1)
    mask = batch["labels"] != tokenizer.pad_token_id

    # Per-sample analysis
    print("=" * 60)
    print("PER-SAMPLE ANALYSIS")
    print("=" * 60)

    for i in range(batch_size):
        print(f"\nSample {i+1}:")

        input_ids = batch['input_ids'][i]
        label_ids = batch['labels'][i]
        pred_ids = predictions[i]
        sample_mask = mask[i]

        # Only look at non-masked positions
        valid_positions = sample_mask.nonzero(as_tuple=True)[0]
        n_valid = len(valid_positions)

        if n_valid == 0:
            print("  No valid tokens (all masked)")
            continue

        valid_labels = label_ids[valid_positions]
        valid_preds = pred_ids[valid_positions]
        valid_inputs = input_ids[valid_positions]

        # Compute per-token accuracy
        matches = (valid_labels == valid_preds)
        n_correct = matches.sum().item()
        accuracy = n_correct / n_valid

        print(f"  Valid tokens: {n_valid}")
        print(f"  Correct: {n_correct}/{n_valid} ({accuracy:.2%})")

        # Find mismatched positions
        mismatches = ~matches
        if mismatches.any():
            mismatch_positions = mismatches.nonzero(as_tuple=True)[0]
            print(f"  Mismatched positions: {mismatch_positions.tolist()}")

            # Show first few mismatches
            for j, pos in enumerate(mismatch_positions[:3]):
                pos_idx = valid_positions[pos].item()
                input_token = tokenizer.decode([input_ids[pos_idx].item()])
                pred_token = tokenizer.decode([valid_preds[pos].item()])
                label_token = tokenizer.decode([valid_labels[pos].item()])
                print(f"    Position {pos.item()}: predicted {pred_token!r}, expected {label_token!r}")

    # Overall batch accuracy
    total_correct = ((predictions == batch["labels"]) & mask).sum().item()
    total_tokens = mask.sum().item()
    overall_acc = total_correct / total_tokens if total_tokens > 0 else 0.0

    print()
    print("=" * 60)
    print(f"Overall batch accuracy: {overall_acc:.4f} ({total_correct}/{total_tokens})")
    print("=" * 60)


if __name__ == "__main__":
    debug_forward_pass()

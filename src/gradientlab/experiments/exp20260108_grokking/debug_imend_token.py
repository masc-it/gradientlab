"""
Debug script to investigate the <|im_end|> token prediction issue.
"""

import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from gradientlab.tokenizers.byte_tokenizer import byte_tokenizer
from gradientlab.experiments.exp20260108_grokking.dataset.torch_dataset import DateCollate, DateToISODataset
from gradientlab.experiments.exp20260108_grokking.modeling.factory import ModelFactory


def debug_imend_token():
    """Investigate why <|im_end|> token cannot be predicted."""

    tokenizer = byte_tokenizer()

    print("=" * 80)
    print("TOKENIZER ANALYSIS")
    print("=" * 80)

    # Test tokenization of special tokens
    test_strings = [
        "<|im_start|>",
        "<|im_end|>",
        " <|im_end|>",
        "8 <|im_end|>",
        " 1954-03-28 <|im_end|>",
    ]

    for s in test_strings:
        tokens = tokenizer(s, add_special_tokens=False)
        token_ids = tokens["input_ids"]
        decoded = tokenizer.decode(token_ids)
        print(f"\nString: {s!r}")
        print(f"  Token IDs: {token_ids}")
        print(f"  Num tokens: {len(token_ids)}")
        print(f"  Decoded: {decoded!r}")

        # Decode each token individually
        print(f"  Individual tokens:")
        for i, tid in enumerate(token_ids):
            tok = tokenizer.decode([tid])
            print(f"    {i}: {tid} -> {tok!r}")

    print("\n" + "=" * 80)
    print("VOCABULARY CHECK")
    print("=" * 80)

    # Check if special tokens are in vocab
    vocab_size = tokenizer.vocab_size
    print(f"Vocab size: {vocab_size}")

    # Check specific token IDs
    special_token_ids = [0, 1, 2, 3, 4, 5, 253, 254, 255]
    print("\nSpecial token IDs:")
    for tid in special_token_ids:
        if tid < vocab_size:
            try:
                decoded = tokenizer.decode([tid])
                print(f"  {tid}: {decoded!r}")
            except:
                print(f"  {tid}: <ERROR>")

    print("\n" + "=" * 80)
    print("MODEL PREDICTION ANALYSIS")
    print("=" * 80)

    # Load model
    checkpoint_path = "src/gradientlab/experiments/exp20260108_grokking/data/model"
    model, tokenizer, config = ModelFactory.build_grokking_model(resume_from=checkpoint_path)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Load dataset
    ds = load_from_disk("src/gradientlab/experiments/exp20260108_grokking/data/date_to_iso")
    torch_ds = DateToISODataset(ds['train'])
    collate = DateCollate(tokenizer)

    # Get a single example
    sample = [torch_ds[0]]
    batch = collate(sample)
    batch = {k: v.to(device) for k, v in batch.items()}

    print(f"\nSample input: {torch_ds[0]}")

    with torch.no_grad():
        output = model(**batch)
        logits = output["logits"]

    # Focus on position 11 (the <|im_end|> position)
    input_ids = batch['input_ids'][0]
    labels = batch['labels'][0]
    mask = labels != tokenizer.pad_token_id

    valid_positions = mask.nonzero(as_tuple=True)[0]
    print(f"\nValid positions (non-masked): {len(valid_positions)}")

    # Analyze position 11 (last position - should be <|im_end|>)
    if len(valid_positions) >= 12:
        pos_11_global = valid_positions[11].item()

        print(f"\n" + "=" * 80)
        print(f"POSITION 11 ANALYSIS (global position {pos_11_global})")
        print("=" * 80)

        # Expected token
        expected_token_id = labels[pos_11_global].item()
        expected_token = tokenizer.decode([expected_token_id])
        print(f"Expected token ID: {expected_token_id}")
        print(f"Expected token: {expected_token!r}")

        # Get logits at this position
        pos_logits = logits[0, pos_11_global, :]
        print(f"Logits shape at position: {pos_logits.shape}")
        print(f"Logits min/max: {pos_logits.min():.4f} / {pos_logits.max():.4f}")

        # Top-k predictions
        topk_logits, topk_ids = torch.topk(pos_logits, k=10)
        print(f"\nTop 10 predictions:")
        for i, (logit, tid) in enumerate(zip(topk_logits, topk_ids)):
            token = tokenizer.decode([tid.item()])
            is_correct = "✓" if tid.item() == expected_token_id else ""
            print(f"  {i+1}. ID={tid.item():<4} logit={logit:.4f}  token={token!r:<20} {is_correct}")

        # Check logit for expected token
        expected_logit = pos_logits[expected_token_id].item()
        print(f"\nExpected token logit: {expected_logit:.4f}")
        print(f"Rank of expected token: ", end="")
        sorted_indices = torch.argsort(pos_logits, descending=True)
        rank = (sorted_indices == expected_token_id).nonzero(as_tuple=True)[0].item() + 1
        print(f"{rank}/{len(pos_logits)}")

        # Check previous positions (context)
        print(f"\n" + "=" * 80)
        print("CONTEXT (positions 9-10, the last two digits)")
        print("=" * 80)
        for offset in [-2, -1]:
            ctx_pos_idx = 11 + offset
            if ctx_pos_idx >= 0:
                ctx_global_pos = valid_positions[ctx_pos_idx].item()
                ctx_input_id = input_ids[ctx_global_pos].item()
                ctx_pred_logits = logits[0, ctx_global_pos, :]
                ctx_pred_id = ctx_pred_logits.argmax().item()

                ctx_input_token = tokenizer.decode([ctx_input_id])
                ctx_pred_token = tokenizer.decode([ctx_pred_id])

                print(f"\nPosition {ctx_pos_idx} (global {ctx_global_pos}):")
                print(f"  Input token: {ctx_input_token!r} (ID={ctx_input_id})")
                print(f"  Predicted: {ctx_pred_token!r} (ID={ctx_pred_id})")

        # Check if the model is just copying the last digit
        pos_10_global = valid_positions[10].item()
        last_digit_id = input_ids[pos_10_global].item()
        last_digit = tokenizer.decode([last_digit_id])
        print(f"\n" + "=" * 80)
        print(f"HYPOTHESIS: Model copies last digit")
        print("=" * 80)
        print(f"Last digit (position 10): {last_digit!r} (ID={last_digit_id})")
        print(f"Predicted at position 11: {tokenizer.decode([topk_ids[0].item()])!r} (ID={topk_ids[0].item()})")
        print(f"Are they the same? {last_digit_id == topk_ids[0].item()}")


if __name__ == "__main__":
    debug_imend_token()

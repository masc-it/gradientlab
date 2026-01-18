"""
Focused debugging of generation on training samples.

Compares:
1. Teacher-forced predictions (what model sees during training)
2. Autoregressive generation (what model produces in reality)
"""

import torch
from datasets import load_from_disk

from gradientlab.experiments.exp20260108_grokking.modeling.factory import ModelFactory


def debug_generation_on_training():
    """Debug generation vs teacher forcing on training samples."""

    print("=" * 80)
    print("GENERATION DEBUGGING - TRAINING SAMPLES")
    print("=" * 80)

    # Load model
    checkpoint_path = "src/gradientlab/experiments/exp20260108_grokking/data/model"
    model, tokenizer, config = ModelFactory.build_grokking_model(resume_from=checkpoint_path)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    model.eval()

    print(f"Device: {device}")
    print(f"Model vocab size: {config.vocab_size}")
    print(f"Tokenizer total_vocab_size: {tokenizer.total_vocab_size}")
    print()

    # Load dataset
    ds_path = "src/gradientlab/experiments/exp20260108_grokking/data/date_to_iso"
    ds = load_from_disk(ds_path)

    # Get first 5 TRAINING samples
    train_samples = [ds['train'][i] for i in range(5)]

    print("Training samples:")
    for i, sample in enumerate(train_samples):
        print(f"  {i+1}. {sample['input']} -> {sample['target']}")
    print()

    # Get EOS token
    eos_token_id = tokenizer("<|im_end|>", add_special_tokens=False)["input_ids"][0]
    print(f"EOS token: <|im_end|> = {eos_token_id}")
    print()

    for idx, sample in enumerate(train_samples):
        print("=" * 80)
        print(f"SAMPLE {idx + 1}: {sample['input']} -> {sample['target']}")
        print("=" * 80)

        # ==================================================================
        # TEST 1: TEACHER FORCING (what model sees during training)
        # ==================================================================
        print("\n[1] TEACHER FORCING (model sees correct context)")
        print("-" * 80)

        # Create full sequence as in training
        full_text = f"<|im_start|>{sample['input']} > {sample['target']}<|im_end|>"
        encoded = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)
        input_ids = encoded["input_ids"].to(device)

        print(f"Full sequence: {full_text!r}")
        print(f"Token IDs: {input_ids[0].tolist()}")
        print(f"Decoded: {tokenizer.decode(input_ids[0].tolist())!r}")
        print()

        # Forward pass
        with torch.no_grad():
            output = model(input_ids=input_ids)
            logits = output["logits"]
            predictions = logits.argmax(dim=-1)

        # Find separator position
        sep_token_id = tokenizer(">", add_special_tokens=False)["input_ids"][0]
        sep_positions = (input_ids[0] == sep_token_id).nonzero(as_tuple=True)[0]
        if len(sep_positions) > 0:
            sep_idx = sep_positions[0].item()

            # Get predictions after separator (the target part)
            target_start = sep_idx + 1
            target_predictions = predictions[0, target_start:]
            target_labels = input_ids[0, target_start + 1:]  # Shifted by 1 (next token prediction)

            # Decode predictions
            pred_tokens = target_predictions[:len(target_labels)].tolist()
            label_tokens = target_labels.tolist()

            pred_text = tokenizer.decode(pred_tokens)
            label_text = tokenizer.decode(label_tokens)

            print(f"Target predictions (token-by-token):")
            print(f"  Predicted tokens: {pred_tokens}")
            print(f"  Expected tokens:  {label_tokens}")
            print(f"  Predicted text: {pred_text!r}")
            print(f"  Expected text:  {label_text!r}")

            # Token-by-token comparison
            correct = sum(p == l for p, l in zip(pred_tokens, label_tokens))
            print(f"  Accuracy: {correct}/{len(label_tokens)} = {correct/len(label_tokens):.2%}")

            # Show mismatches
            print(f"  Mismatches:")
            for i, (p, l) in enumerate(zip(pred_tokens, label_tokens)):
                if p != l:
                    p_char = tokenizer.decode([p])
                    l_char = tokenizer.decode([l])
                    print(f"    Position {i}: predicted {p} ({p_char!r}) vs expected {l} ({l_char!r})")

        # ==================================================================
        # TEST 2: AUTOREGRESSIVE GENERATION (reality)
        # ==================================================================
        print("\n[2] AUTOREGRESSIVE GENERATION (model uses own predictions)")
        print("-" * 80)

        # Create prompt without target
        prompt = f"<|im_start|>{sample['input']} >"
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = encoded["input_ids"].to(device)

        print(f"Prompt: {prompt!r}")
        print(f"Token IDs: {input_ids[0].tolist()}")
        print()

        # Generate step-by-step to see what happens
        print("Generating step-by-step:")
        current_ids = input_ids.clone()

        for step in range(15):  # Max 15 tokens
            with torch.no_grad():
                output = model(input_ids=current_ids)
                logits = output["logits"]
                next_token_logits = logits[0, -1, :]  # Last position
                next_token = next_token_logits.argmax(dim=-1).item()

            next_char = tokenizer.decode([next_token])
            print(f"  Step {step+1}: token {next_token} = {next_char!r}")

            # Stop if EOS
            if next_token == eos_token_id:
                print(f"  → EOS reached!")
                break

            # Append and continue
            current_ids = torch.cat([current_ids, torch.tensor([[next_token]], device=device)], dim=1)

        # Final result
        generated_text = tokenizer.decode(current_ids[0].tolist())
        print()
        print(f"Full generated: {generated_text!r}")

        # Extract prediction
        if " > " in generated_text:
            parts = generated_text.split(" > ", 1)
            prediction = parts[1] if len(parts) > 1 else ""
        else:
            prediction = generated_text

        if "<|im_end|>" in prediction:
            prediction = prediction.replace("<|im_end|>", "").strip()

        expected = sample['target']
        match = "✓" if prediction == expected else "✗"

        print(f"\n{match} Expected:  {expected!r}")
        print(f"{match} Generated: {prediction!r}")

        print()

    # ==================================================================
    # TEST 3: Check model's generate function
    # ==================================================================
    print("\n" + "=" * 80)
    print("TEST 3: Using model.generate() function")
    print("=" * 80)

    prompts = [f"<|im_start|>{sample['input']} >" for sample in train_samples[:3]]
    encoded = tokenizer(
        prompts,
        padding="longest",
        padding_side="left",
        return_tensors="pt",
        add_special_tokens=False,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    print(f"Batch input shape: {input_ids.shape}")
    print()

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=15,
            eos_token_id=eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    print("Results:")
    for i, sample in enumerate(train_samples[:3]):
        gen_ids = generated_ids[i]
        attn_mask = (gen_ids != tokenizer.pad_token_id)
        real_tokens = gen_ids[attn_mask]
        generated_text = tokenizer.decode(real_tokens.tolist())

        if " > " in generated_text:
            parts = generated_text.split(" > ", 1)
            prediction = parts[1] if len(parts) > 1 else ""
        else:
            prediction = generated_text

        if "<|im_end|>" in prediction:
            prediction = prediction.replace("<|im_end|>", "").strip()

        expected = sample['target']
        match = "✓" if prediction == expected else "✗"

        print(f"\n{i+1}. {match}")
        print(f"   Input:     {sample['input']!r}")
        print(f"   Expected:  {expected!r}")
        print(f"   Generated: {prediction!r}")
        print(f"   Full:      {generated_text!r}")


if __name__ == "__main__":
    debug_generation_on_training()

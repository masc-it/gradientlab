"""Test script to verify model changes."""
import torch
from modeling.factory import ModelFactory


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def test_model():
    print("=" * 60)
    print("Testing ImageTextSlot Model")
    print("=" * 60)

    # 1. Build model
    print("\n1. Building model...")
    model, tokenizer, cfg = ModelFactory.build_5m()
    print("   ✓ Model instantiated successfully")

    # 2. Count parameters
    total_params, trainable_params = count_parameters(model)
    print(f"\n2. Parameter count:")
    print(f"   Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"   Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")

    # 3. Test forward pass
    print("\n3. Testing forward pass...")
    batch_size = 2
    pixel_values = torch.randn(batch_size, 1, 128, 128)
    labels = torch.randint(0, 512, (batch_size, 100))
    attention_mask = torch.ones(batch_size, 100)

    # Forward pass
    with torch.no_grad():
        output = model(
            pixel_values=pixel_values,
            labels=labels,
            attention_mask=attention_mask,
        )

    # Verify shapes
    print(f"   Input shape: {pixel_values.shape}")
    print(f"   Logits shape: {output['logits'].shape}")
    print(f"   Count pred shape: {output['count_pred'].shape}")

    assert output['logits'].shape == (batch_size, 1024, 512), "Logits shape mismatch"
    assert output['count_pred'].shape == (batch_size, 1), "Count pred shape mismatch"
    print("   ✓ Forward pass successful")

    # 4. Check loss computation
    print("\n4. Testing loss computation...")
    assert 'loss' in output, "Loss not in output"
    assert 'loss_ce' in output, "CE loss not in output"
    assert 'loss_count' in output, "Count loss not in output"
    print(f"   Total loss: {output['loss']:.4f}")
    print(f"   CE loss: {output['loss_ce']:.4f}")
    print(f"   Count loss: {output['loss_count']:.4f}")
    print("   ✓ Loss computation successful")

    # 5. Test generate method
    print("\n5. Testing generate method...")
    outputs = model.generate(pixel_values, bos_token_id=0, eos_token_id=1)
    assert len(outputs) == batch_size, "Generate output batch size mismatch"
    print(f"   Generated {len(outputs)} sequences")
    print(f"   Sequence lengths: {[len(seq) for seq in outputs]}")
    print("   ✓ Generate method successful")

    # 6. Test backward pass
    print("\n6. Testing backward pass...")
    pixel_values.requires_grad = True
    output = model(
        pixel_values=pixel_values,
        labels=labels,
        attention_mask=attention_mask,
    )
    loss = output['loss']
    loss.backward()
    print(f"   Loss backward computed: {loss.item():.4f}")
    print("   ✓ Backward pass successful")

    # 7. Verify positional embeddings exist
    print("\n7. Verifying new components...")
    assert hasattr(model, 'slot_pos_embed'), "Missing slot_pos_embed parameter"
    print(f"   slot_pos_embed shape: {model.slot_pos_embed.shape}")
    print("   ✓ Positional embeddings present")

    # 8. Verify residual scaling removed
    assert not hasattr(model, '_apply_residual_scaling'), "Residual scaling method still exists"
    print("   ✓ Residual scaling method removed")

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_model()

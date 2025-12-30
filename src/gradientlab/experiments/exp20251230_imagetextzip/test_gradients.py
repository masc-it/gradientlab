"""Test script to verify gradient explosion fixes."""
import torch
from modeling.factory import ModelFactory


def test_gradient_norms():
    print("=" * 70)
    print("Testing Gradient Norms After Fixes")
    print("=" * 70)

    # 1. Build model
    print("\n1. Building model...")
    model, tokenizer, cfg = ModelFactory.build_5m()
    print("   ✓ Model instantiated")
    print(f"   Label smoothing: {cfg.label_smoothing}")
    print(f"   Counter loss weight: {cfg.counter_loss_weight}")

    # 2. Check classifier initialization
    print("\n2. Checking classifier initialization...")
    classifier_weight_std = model.classifier.weight.std().item()
    expected_std = 0.02 / (512 ** 0.5)  # sqrt(512) ≈ 22.6
    print(f"   Classifier weight std: {classifier_weight_std:.6f}")
    print(f"   Expected std: {expected_std:.6f}")
    print(f"   Ratio (should be ~1.0): {classifier_weight_std / expected_std:.2f}")

    # 3. Check positional embeddings initialization
    print("\n3. Checking positional embeddings...")
    pos_embed_std = model.slot_pos_embed.std().item()
    pos_embed_mean = model.slot_pos_embed.mean().item()
    print(f"   Pos embed std: {pos_embed_std:.6f} (should be ~0.02)")
    print(f"   Pos embed mean: {pos_embed_mean:.6f} (should be ~0.0)")
    is_zeros = (model.slot_pos_embed == 0).all().item()
    print(f"   All zeros: {is_zeros} (should be False)")

    # 4. Run forward and backward pass
    print("\n4. Testing forward and backward pass...")
    batch_size = 4
    pixel_values = torch.randn(batch_size, 1, 128, 128)
    labels = torch.randint(0, 512, (batch_size, 100))
    attention_mask = torch.ones(batch_size, 100)

    # Forward
    output = model(
        pixel_values=pixel_values,
        labels=labels,
        attention_mask=attention_mask,
    )

    print(f"   Total loss: {output['loss']:.4f}")
    print(f"   CE loss: {output['loss_ce']:.4f}")
    print(f"   Count loss: {output['loss_count']:.4f}")

    # Backward
    loss = output['loss']
    loss.backward()

    # 5. Check gradient norms per layer
    print("\n5. Per-layer gradient norms:")
    print("   " + "-" * 60)

    grad_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_norms[name] = grad_norm

    # Sort by magnitude
    sorted_grads = sorted(grad_norms.items(), key=lambda x: x[1], reverse=True)

    # Print top 10 largest gradients
    print("   Top 10 largest gradient norms:")
    for i, (name, norm) in enumerate(sorted_grads[:10], 1):
        status = "⚠️" if norm > 5.0 else "✓"
        print(f"   {i:2d}. {status} {name:50s}: {norm:8.4f}")

    # 6. Calculate total gradient norm
    print("\n6. Overall gradient statistics:")
    total_grad_norm = (sum(g**2 for g in grad_norms.values()) ** 0.5)
    print(f"   Total gradient norm: {total_grad_norm:.4f}")
    print(f"   Number of parameters with gradients: {len(grad_norms)}")
    print(f"   Max gradient norm: {max(grad_norms.values()):.4f}")
    print(f"   Mean gradient norm: {sum(grad_norms.values()) / len(grad_norms):.4f}")

    # 7. Assessment
    print("\n7. Assessment:")
    print("   " + "-" * 60)

    issues = []
    if total_grad_norm > 10.0:
        issues.append(f"Total grad norm too high ({total_grad_norm:.2f} > 10)")
    if max(grad_norms.values()) > 5.0:
        issues.append(f"Max layer grad norm too high ({max(grad_norms.values()):.2f} > 5)")
    if classifier_weight_std / expected_std > 1.5 or classifier_weight_std / expected_std < 0.5:
        issues.append(f"Classifier init off (ratio: {classifier_weight_std / expected_std:.2f})")
    if is_zeros:
        issues.append("Positional embeddings still zeros!")

    if issues:
        print("   ⚠️  Issues detected:")
        for issue in issues:
            print(f"      - {issue}")
    else:
        print("   ✅ All checks passed!")
        print("   Gradient norms are within acceptable range")
        print("   Expected improvement: norms 0.5-3.0 instead of >10")

    # 8. Comparison with clipping threshold
    print("\n8. Gradient clipping analysis:")
    clip_threshold = 1.0
    print(f"   Current clip threshold: {clip_threshold:.1f}")
    print(f"   Total grad norm: {total_grad_norm:.4f}")
    clip_ratio = total_grad_norm / clip_threshold
    print(f"   Clipping ratio: {clip_ratio:.2f}x")

    if clip_ratio > 2.0:
        print(f"   ⚠️  Gradients being heavily clipped ({clip_ratio:.1f}x reduction)")
        print("   Consider:")
        print("   - Increasing clip threshold to 2.0-5.0")
        print("   - Further reducing learning rate")
        print("   - Adding dropout before classifier")
    elif clip_ratio > 1.0:
        print(f"   ⚠️  Gradients being clipped ({clip_ratio:.1f}x reduction)")
    else:
        print("   ✅ Gradients within clip threshold")

    print("\n" + "=" * 70)
    print("Test complete")
    print("=" * 70)


if __name__ == "__main__":
    test_gradient_norms()

"""
Test autoregressive accuracy computation.
"""

from gradientlab.experiments.exp20260108_grokking.exp_config import ExpConfig
from gradientlab.experiments.exp20260108_grokking.modeling.factory import ModelFactory
from gradientlab.experiments.exp20260108_grokking.trainer import Trainer


def test_autoregressive_accuracy():
    """Test the autoregressive accuracy computation."""

    print("=" * 80)
    print("TESTING AUTOREGRESSIVE ACCURACY COMPUTATION")
    print("=" * 80)

    # Load configuration
    exp_cfg = ExpConfig()
    exp_cfg.resume_from = "src/gradientlab/experiments/exp20260108_grokking/data/model"

    # Build model from factory
    model, tokenizer, model_cfg = ModelFactory.build_grokking_model(exp_cfg.resume_from)

    print("\nInitializing trainer...")
    # Initialize trainer (this builds dataloaders)
    trainer = Trainer(model, tokenizer, model_cfg, exp_cfg)

    print("\nComputing eval metrics...")
    # Test the eval accuracy function (should compute both teacher forced and autoregressive)
    eval_loss, teacher_forced_acc, autoregressive_acc = trainer._eval_accuracy()

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Eval Loss: {eval_loss:.4f}")
    print(f"Teacher Forced Accuracy: {teacher_forced_acc:.4f} (model sees correct context)")
    print(f"Autoregressive Accuracy: {autoregressive_acc:.4f} (true generation quality)")
    print("=" * 80)

    # The key insight
    print("\nKEY INSIGHT:")
    print("Teacher forced accuracy can be high (even 100%) because the model")
    print("sees correct previous tokens during evaluation.")
    print()
    print("Autoregressive accuracy measures true generation quality where the")
    print("model uses its own predictions as context.")
    print()
    print("A large gap indicates the model hasn't truly learned to generate.")


if __name__ == "__main__":
    test_autoregressive_accuracy()

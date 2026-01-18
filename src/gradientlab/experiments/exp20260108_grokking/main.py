"""
Main entry point for grokking experiment.
"""

from gradientlab.experiments.exp20260108_grokking.exp_config import ExpConfig
from gradientlab.experiments.exp20260108_grokking.modeling.factory import ModelFactory
from gradientlab.experiments.exp20260108_grokking.trainer import Trainer


def pretty_print_model(model):
    """Print model architecture summary."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n=== MODEL SUMMARY ===")
    print(f"Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    print("=" * 50)


def main():
    """Main entry point."""
    print("=" * 50)
    print("=== GROKKING EXPERIMENT ===")
    print("=" * 50)

    # Load configuration
    exp_cfg = ExpConfig()

    # Build model from factory
    model, tokenizer, model_cfg = ModelFactory.build_grokking_model(exp_cfg.resume_from)

    # Print model information
    pretty_print_model(model)
    print("\n=== MODEL CONFIG ===")
    print(model_cfg.model_dump_json(indent=2))
    print("\n=== EXPERIMENT CONFIG ===")
    print(exp_cfg.model_dump_json(indent=2))
    print()

    # Initialize trainer
    trainer = Trainer(model, tokenizer, model_cfg, exp_cfg)

    # Train
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n[Interrupted] Saving checkpoint...")
        trainer._save_state()


if __name__ == "__main__":
    main()

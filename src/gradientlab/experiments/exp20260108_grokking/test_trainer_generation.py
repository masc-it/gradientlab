"""
Test the trainer's generation examples functionality.
"""

import torch
from gradientlab.experiments.exp20260108_grokking.exp_config import ExpConfig
from gradientlab.experiments.exp20260108_grokking.modeling.factory import ModelFactory
from gradientlab.experiments.exp20260108_grokking.trainer import Trainer


def test_generation_in_trainer():
    """Test that the generation examples work in the trainer."""

    print("=" * 80)
    print("TESTING TRAINER GENERATION EXAMPLES")
    print("=" * 80)

    # Load configuration
    exp_cfg = ExpConfig()
    exp_cfg.resume_from = "src/gradientlab/experiments/exp20260108_grokking/data/model"

    # Build model from factory
    model, tokenizer, model_cfg = ModelFactory.build_grokking_model(exp_cfg.resume_from)

    print("\nInitializing trainer...")
    # Initialize trainer
    trainer = Trainer(model, tokenizer, model_cfg, exp_cfg)

    print("\nCalling _show_generation_examples...")
    # Test the generation examples function
    trainer._show_generation_examples(num_examples=5)

    print("\n✓ Generation examples working correctly!")


if __name__ == "__main__":
    test_generation_in_trainer()

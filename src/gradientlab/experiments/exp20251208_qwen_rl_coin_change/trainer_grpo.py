"""
GRPO fine-tuning for a decoder-only LM on the Coin Change task using HuggingFace TRL.

This trainer:
1. First runs SFT warmup to teach the model the output format
2. Then runs GRPO (Group Relative Policy Optimization) for RL fine-tuning

Requirements:
  pip install torch transformers trl datasets trackio

Notes:
  - Uses `rl_env.py` for problem generation and reward computation
  - Uses `modeling.py` for the custom Qwen3 model and ByteLevelTokenizer
"""

import os
import random
from dataclasses import dataclass, field
from typing import List, Dict

import torch
from datasets import Dataset
from transformers import TrainingArguments, Trainer
from trl import GRPOTrainer, GRPOConfig
import trackio

from gradientlab.experiments.exp20251208_qwen_rl_coin_change.modeling import (
    init_qwen3_10m_bytelevel,
)
from gradientlab.experiments.exp20251208_qwen_rl_coin_change.rl_env import (
    generate_min_coins_instance,
    verify_solution,
)


# ---------------------------------------------------------
#  CONFIG
# ---------------------------------------------------------


@dataclass
class TrainConfig:
    device: torch.device = field(
        default_factory=lambda: torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    )

    # SFT warmup phase
    sft_steps: int = 500
    sft_batch_size: int = 32
    sft_lr: float = 1e-4
    sft_warmup_ratio: float = 0.1
    sft_eval_every: int = 200

    # GRPO configuration
    grpo_steps: int = 100
    grpo_batch_size: int = 32  # per-device batch size
    grpo_num_generations: int = 8  # number of completions per prompt (group size)
    grpo_lr: float = 5e-5
    grpo_warmup_ratio: float = 0.01
    grpo_beta: float = 0.2  # KL penalty coefficient
    grpo_max_new_tokens: int = 256
    grpo_temperature: float = 0.8

    # General
    max_grad_norm: float = 2.0
    weight_decay: float = 0.0

    # Logging and saving
    log_every: int = 10
    save_every: int = 500
    eval_every: int = 100
    save_dir: str = "./checkpointscoinchangegrpo"
    project_name: str = "coinchangegrpo"

    # Evaluation
    num_eval_problems: int = 50

    # Dataset
    num_train_prompts: int = 100000  # number of prompts in the training dataset
    seed: int = 42


# ---------------------------------------------------------
#  PROMPTING UTILITIES
# ---------------------------------------------------------


def build_prompt(coins: List[int], amount: int) -> str:
    """Build the text the model sees as input."""
    coins_str = ",".join(str(c) for c in coins)
    return f"coins: {coins_str}\namount: {amount}\n\n"


def build_completion(answer: int, solution_coins: List[int]) -> str:
    """Build the target completion text for SFT."""
    solution_str = "+".join(str(c) for c in solution_coins) if answer != -1 else ""
    return f"answer: {answer}\nsolution: {solution_str}\n"


# ---------------------------------------------------------
#  DATASET GENERATION
# ---------------------------------------------------------


def generate_sft_dataset(num_samples: int, seed: int = 42) -> Dataset:
    """
    Generate a dataset of (prompt, completion) pairs for SFT.
    """
    rng = random.Random(seed)
    data = {"text": []}

    for _ in range(num_samples):
        coins, amount, answer, solution_coins = generate_min_coins_instance(rng=rng)
        prompt = build_prompt(coins, amount)
        completion = build_completion(answer, solution_coins)
        # Full text for causal LM training
        data["text"].append(prompt + completion)

    return Dataset.from_dict(data)


def generate_grpo_dataset(num_samples: int, seed: int = 42) -> Dataset:
    """
    Generate a dataset of prompts for GRPO training.
    The dataset must have a 'prompt' column.
    We also store ground truth for reward computation.
    """
    rng = random.Random(seed)
    data = {
        "prompt": [],
        "ground_truth_answer": [],
        "ground_truth_solution": [],
        "coins": [],
        "amount": [],
    }

    for _ in range(num_samples):
        coins, amount, answer, solution_coins = generate_min_coins_instance(rng=rng)
        prompt = build_prompt(coins, amount)
        data["prompt"].append(prompt)
        data["ground_truth_answer"].append(answer)
        data["ground_truth_solution"].append(solution_coins)
        data["coins"].append(coins)
        data["amount"].append(amount)

    return Dataset.from_dict(data)


# ---------------------------------------------------------
#  REWARD FUNCTION
# ---------------------------------------------------------


def coin_change_reward(
    prompts: List[str],
    completions: List[str],
    ground_truth_answer: List[int],
    ground_truth_solution: List[List[int]],
    coins: List[List[int]],
    amount: List[int],
    **kwargs,
) -> List[float]:
    """
    Reward function for GRPO that computes reward based on verify_solution.

    Args:
        prompts: List of prompt strings
        completions: List of completion strings from the model
        ground_truth_answer: List of correct answers
        ground_truth_solution: List of correct solutions
        coins: List of coin denominations per problem
        amount: List of target amounts per problem

    Returns:
        List of float rewards in [0, 1]
    """
    rewards = []
    for prompt, completion in zip(prompts, completions):
        # verify_solution expects the problem text and predicted text
        reward = verify_solution(prompt, completion)
        rewards.append(float(reward))
    return rewards


# ---------------------------------------------------------
#  SFT TRAINER
# ---------------------------------------------------------


def run_sft_phase(
    model,
    tokenizer,
    cfg: TrainConfig,
) -> None:
    """
    Run supervised fine-tuning phase to teach the model the output format.
    """
    print(f"\n{'='*60}")
    print(f"PHASE 1: SFT Warmup ({cfg.sft_steps} steps)")
    print(f"{'='*60}\n")

    # Generate SFT dataset
    sft_dataset = generate_sft_dataset(
        num_samples=cfg.sft_steps * cfg.sft_batch_size,
        seed=cfg.seed,
    )

    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
            padding="max_length",
        )

    tokenized_dataset = sft_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
    )

    # Set up labels for causal LM (labels = input_ids)
    def add_labels(examples):
        examples["labels"] = examples["input_ids"].copy()
        return examples

    tokenized_dataset = tokenized_dataset.map(add_labels, batched=True)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(cfg.save_dir, "sft"),
        num_train_epochs=1,
        per_device_train_batch_size=cfg.sft_batch_size,
        learning_rate=cfg.sft_lr,
        warmup_ratio=cfg.sft_warmup_ratio,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        logging_steps=cfg.log_every,
        save_steps=cfg.save_every,
        eval_strategy="no",
        save_total_limit=2,
        bf16=torch.cuda.is_available(),
        dataloader_pin_memory=True,
        report_to="none",  # We use trackio instead
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    # Train
    trainer.train()

    print(f"\n{'='*60}")
    print("SFT Warmup Complete!")
    print(f"{'='*60}\n")


# ---------------------------------------------------------
#  GRPO TRAINING
# ---------------------------------------------------------


def run_grpo_phase(
    model,
    tokenizer,
    cfg: TrainConfig,
) -> None:
    """
    Run GRPO fine-tuning phase for RL optimization.
    """
    print(f"\n{'='*60}")
    print(f"PHASE 2: GRPO Fine-tuning ({cfg.grpo_steps} steps)")
    print(f"{'='*60}\n")

    # Generate GRPO dataset
    grpo_dataset = generate_grpo_dataset(
        num_samples=cfg.num_train_prompts,
        seed=cfg.seed + 1000,  # Different seed from SFT
    )

    # If using KL penalty (beta > 0), we need to save model to disk first
    # so GRPOTrainer can load it as a reference model
    if cfg.grpo_beta > 0:
        ref_model_path = os.path.join(cfg.save_dir, "ref_model")
        print(f"Saving reference model to {ref_model_path} for KL regularization...")
        model.save_pretrained(ref_model_path)
        tokenizer.save_pretrained(ref_model_path)
        # Set the model's _name_or_path so GRPOTrainer can find it
        model.config._name_or_path = ref_model_path
        print(f"Reference model saved (beta={cfg.grpo_beta})")

    # GRPO configuration
    grpo_config = GRPOConfig(
        output_dir=os.path.join(cfg.save_dir, "grpo"),
        num_train_epochs=1,
        per_device_train_batch_size=cfg.grpo_batch_size,
        learning_rate=cfg.grpo_lr,
        warmup_ratio=cfg.grpo_warmup_ratio,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        logging_steps=cfg.log_every,
        save_steps=cfg.save_every,
        eval_strategy="no",
        save_total_limit=2,
        bf16=torch.cuda.is_available(),
        report_to="none",
        # GRPO-specific parameters
        num_generations=cfg.grpo_num_generations,
        max_completion_length=cfg.grpo_max_new_tokens,
        beta=cfg.grpo_beta,
        temperature=cfg.grpo_temperature,

    )

    # Create GRPO trainer
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=grpo_dataset,
        processing_class=tokenizer,
        reward_funcs=coin_change_reward,
    )

    # Train
    trainer.train()

    # Save final model
    trainer.save_model(os.path.join(cfg.save_dir, "grpo_final"))

    print(f"\n{'='*60}")
    print("GRPO Fine-tuning Complete!")
    print(f"{'='*60}\n")


# ---------------------------------------------------------
#  EVALUATION
# ---------------------------------------------------------


def evaluate(
    model,
    tokenizer,
    cfg: TrainConfig,
    eval_seed: int = 123,
) -> Dict[str, float]:
    """
    Evaluate the model with greedy decoding on a fixed set of problems.
    """
    device = cfg.device
    model.eval()

    rng = random.Random(eval_seed)
    problems = []
    prompts = []

    for _ in range(cfg.num_eval_problems):
        coins, amount, answer, solution_coins = generate_min_coins_instance(rng=rng)
        prompt = build_prompt(coins, amount)
        problems.append({
            "prompt": prompt,
            "answer": answer,
            "solution_coins": solution_coins,
        })
        prompts.append(prompt)

    # Tokenize
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding="longest",
        truncation=True,
    ).to(device)

    # Generate
    with torch.no_grad():
        gen_out = model.generate(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            max_new_tokens=cfg.grpo_max_new_tokens,
            do_sample=False,  # greedy
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode completions
    prefix_len = enc["input_ids"].size(1)
    completions = []
    for i in range(len(prompts)):
        comp_ids = gen_out[i, prefix_len:]
        completions.append(tokenizer.decode(comp_ids, skip_special_tokens=True))

    # Compute rewards
    rewards = []
    for prob, comp in zip(problems, completions):
        reward = verify_solution(prob["prompt"], comp)
        rewards.append(float(reward))

    accuracy = sum(1 for r in rewards if r == 1.0) / len(rewards)
    reward_mean = sum(rewards) / len(rewards)

    model.train()

    return {
        "eval_accuracy": accuracy,
        "eval_reward_mean": reward_mean,
    }


# ---------------------------------------------------------
#  MAIN
# ---------------------------------------------------------


def main():
    # Setup
    torch.set_float32_matmul_precision("high")
    random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    cfg = TrainConfig()
    os.makedirs(cfg.save_dir, exist_ok=True)

    # Initialize tracking
    trackio.init(project=cfg.project_name)

    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model, tokenizer = init_qwen3_10m_bytelevel()

    # Ensure tokenizer has pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(cfg.device)

    # Phase 1: SFT warmup
    if cfg.sft_steps > 0:
        run_sft_phase(model, tokenizer, cfg)

        # Evaluate after SFT
        eval_results = evaluate(model, tokenizer, cfg)
        print(f"Post-SFT Evaluation: {eval_results}")
        trackio.log(eval_results, step=cfg.sft_steps)

    # Phase 2: GRPO fine-tuning
    if cfg.grpo_steps > 0:
        run_grpo_phase(model, tokenizer, cfg)

        # Final evaluation
        eval_results = evaluate(model, tokenizer, cfg)
        print(f"Final Evaluation: {eval_results}")
        trackio.log(eval_results, step=cfg.sft_steps + cfg.grpo_steps)

    print(f"\nTraining complete! Models saved to {cfg.save_dir}")


if __name__ == "__main__":
    main()

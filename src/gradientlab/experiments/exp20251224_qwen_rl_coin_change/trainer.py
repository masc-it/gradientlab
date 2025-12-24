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

import json
import os
import random
from dataclasses import dataclass, field
from typing import List, Dict

import torch
from datasets import load_from_disk
from transformers import TrainingArguments, Trainer, TrainerCallback
import trackio

from gradientlab.experiments.exp20251224_qwen_rl_coin_change.modeling import (
    init_qwen3_10m_bytelevel,
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
    grpo_max_new_tokens: int = 4096
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
    dataset_path: str = "./data/coin_change"  # path to saved DatasetDict
    seed: int = 42


# ---------------------------------------------------------
#  UTILS
# ---------------------------------------------------------


def make_eval_prompt(text: str) -> str:
    """
    Truncate JSON text right before 'solution' key for evaluation.

    Example:
        Input:  '{\n  "coins": [1, 2, 5],\n  "amount": 11,\n  "solution": [5, 5, 1],\n  "answer": 3\n}'
        Output: '{\n  "coins": [1, 2, 5],\n  "amount": 11,\n  '
    """
    idx = text.find('"solution"')
    if idx == -1:
        return text
    return text[:idx]


# ---------------------------------------------------------
#  SFT TRAINER
# ---------------------------------------------------------


class EvalCallback(TrainerCallback):
    """Callback to run custom evaluation at the end of each epoch."""

    def __init__(self, model, tokenizer, test_dataset, cfg):
        self.model = model
        self.tokenizer = tokenizer
        self.test_dataset = test_dataset
        self.cfg = cfg

    def on_epoch_end(self, args, state, control, **kwargs):
        eval_results = evaluate(
            self.model, self.tokenizer, self.test_dataset, self.cfg
        )
        print(f"Epoch {state.epoch:.0f} Evaluation: {eval_results}")
        trackio.log(eval_results, step=state.global_step)


def run_sft_phase(
    model,
    tokenizer,
    train_dataset,
    test_dataset,
    cfg: TrainConfig,
) -> None:
    """
    Run supervised fine-tuning phase to teach the model the output format.
    """
    print(f"\n{'=' * 60}")
    print(f"PHASE 1: SFT Training ({cfg.sft_steps} steps)")
    print(f"{'=' * 60}\n")

    # Tokenize dataset with left padding
    def tokenize_function(examples):
        encodings = tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
            padding="max_length",
        )
        # Set labels = input_ids, but mask padding tokens with -100
        labels = []
        for input_ids in encodings["input_ids"]:
            label = [
                -100 if tok == tokenizer.pad_token_id else tok for tok in input_ids
            ]
            labels.append(label)
        encodings["labels"] = labels
        return encodings

    tokenized_dataset = train_dataset.map(
        tokenize_function, batched=True, remove_columns=["text"], num_proc=4
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(cfg.save_dir, "sft"),
        num_train_epochs=1,
        max_steps=cfg.sft_steps,
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

    # Create trainer with eval callback
    eval_callback = EvalCallback(model, tokenizer, test_dataset, cfg)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        callbacks=[eval_callback],
    )

    # Train
    trainer.train()

    print(f"\n{'=' * 60}")
    print("SFT Training Complete!")
    print(f"{'=' * 60}\n")


# ---------------------------------------------------------
#  EVALUATION
# ---------------------------------------------------------


def evaluate(
    model,
    tokenizer,
    test_dataset,
    cfg: TrainConfig,
    eval_seed: int = 123,
) -> Dict[str, float]:
    """
    Evaluate the model with greedy decoding on a fixed set of problems.
    """
    device = cfg.device
    model.eval()

    # Sample problems from test dataset
    rng = random.Random(eval_seed)
    num_samples = min(cfg.num_eval_problems, len(test_dataset))
    indices = rng.sample(range(len(test_dataset)), num_samples)

    problems = []  # parsed JSON
    prompts = []  # truncated text for generation

    for idx in indices:
        text = test_dataset[idx]["text"]
        example = json.loads(text)
        problems.append(example)
        prompts.append(make_eval_prompt(text))

    # Tokenize with left padding
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

    # Decode completions (only new tokens)
    prefix_len = enc["input_ids"].size(1)
    completions = []
    for i in range(len(prompts)):
        comp_ids = gen_out[i, prefix_len:]
        completions.append(tokenizer.decode(comp_ids, skip_special_tokens=True))

    # Compute accuracy
    correct = 0
    for prob, prompt, comp in zip(problems, prompts, completions):
        # Try to parse the completion as JSON continuation
        full_text = prompt + comp
        try:
            pred = json.loads(full_text)
            # Check if solution and answer match
            if (
                pred.get("solution") == prob["solution"]
                and pred.get("answer") == prob["answer"]
            ):
                correct += 1
        except json.JSONDecodeError:
            pass  # Failed to parse, counts as incorrect

    accuracy = correct / len(problems) if problems else 0.0

    model.train()

    return {
        "eval_accuracy": accuracy,
        "eval_num_correct": correct,
        "eval_num_total": len(problems),
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

    # Load dataset from disk
    print(f"Loading dataset from {cfg.dataset_path}...")
    dataset = load_from_disk(cfg.dataset_path)
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    print(f"  Train: {len(train_dataset)} examples")
    print(f"  Test: {len(test_dataset)} examples")

    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model, tokenizer = init_qwen3_10m_bytelevel()

    # Ensure tokenizer has pad token and use left padding for generation
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(cfg.device)

    # SFT training
    if cfg.sft_steps > 0:
        run_sft_phase(model, tokenizer, train_dataset, test_dataset, cfg)

    print(f"\nTraining complete! Models saved to {cfg.save_dir}")


if __name__ == "__main__":
    main()

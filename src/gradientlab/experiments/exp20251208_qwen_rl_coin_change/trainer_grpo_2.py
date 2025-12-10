#!/usr/bin/env python
import json
import math
import random
import re
from typing import Any, Dict, List, Tuple

import torch
from datasets import Dataset
from transformers import AutoTokenizer

from trl import GRPOTrainer, GRPOConfig


MODEL_ID = "Qwen/Qwen3-0.6B" #"LiquidAI/LFM2-350M"

SYSTEM_PROMPT = (
    "You are an expert problem solver trained to solve the Coin Change problem from LeetCode. "
    "Given a list of coin denominations and a target amount, you must return ONLY a JSON object "
    'with two fields: "answer" (the minimum number of coins needed to make the amount, or -1 if it '
    'is not possible) and "solution" (a string representation of one optimal multiset of coins, '
    'written as "c1+c2+...+ck", or an empty string if no solution exists). '
    "Do not add any explanations, comments, or text outside the JSON."
)


# ----------------------------
#  Coin change solver (ground truth)
# ----------------------------

def solve_coin_change(coins: List[int], amount: int) -> Tuple[int, List[int]]:
    """
    Classic unbounded Coin Change (LeetCode 322):
    Returns (min_number_of_coins, example_solution_list).
    If impossible, returns (-1, []).
    """
    coins = sorted(set(coins))
    INF = amount + 1
    dp = [INF] * (amount + 1)
    prev = [-1] * (amount + 1)

    dp[0] = 0
    for a in range(1, amount + 1):
        for c in coins:
            if c <= a and dp[a - c] + 1 < dp[a]:
                dp[a] = dp[a - c] + 1
                prev[a] = c

    if dp[amount] == INF:
        return -1, []

    solution = []
    a = amount
    while a > 0:
        c = prev[a]
        if c == -1:
            # Should not really happen if dp[amount] != INF, but guard anyway
            break
        solution.append(c)
        a -= c

    solution.sort()
    return dp[amount], solution


# ----------------------------
#  Synthetic dataset generation
# ----------------------------

def build_example(coins: List[int], amount: int) -> Dict[str, Any]:
    optimal_answer, solution_list = solve_coin_change(coins, amount)
    solution_str = "+".join(str(c) for c in solution_list) if optimal_answer != -1 else ""

    user_prompt = (
        "You are an expert problem solver trained to solve the Coin Change problem. "
        "Given a list of coin denominations and a target amount, you must return ONLY a JSON object "
        'with two fields: "answer" (the minimum number of coins needed to make the amount, or -1 if it '
        'is not possible) and "solution" (a list of numbers containing one optimal multiset of coins, '
        'e.g. [2, 3, 3, 5], or an empty list if no solution exists). '
        "Do not add any explanations, comments, or text outside the JSON.\n\n"
        f"coins = {coins}, amount = {amount}"
    )

    messages = [
        #{"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    return {
        "prompt": messages,             # Conversational prompt for TRL
        "coins": coins,
        "amount": amount,
        "optimal_answer": optimal_answer,
        "optimal_solution": solution_str,
    }


def make_coin_change_datasets(
    n_train: int = 1024,
    n_eval: int = 256,
    min_amount: int = 4,
    max_amount: int = 60,
    min_coin: int = 1,
    max_coin: int = 25,
    seed: int = 42,
) -> Tuple[Dataset, Dataset]:
    """
    Build two prompt-only conversational datasets:
    - train_dataset
    - eval_dataset

    Each row has:
      - prompt: [{"role":..., "content":...}, ...]
      - coins: List[int]
      - amount: int
      - optimal_answer: int
      - optimal_solution: str
    """
    rng = random.Random(seed)
    total = n_train + n_eval
    samples: List[Dict[str, Any]] = []

    for _ in range(total):
        num_coins = rng.randint(2, 5)
        # Random distinct positive coins
        coins = sorted(set(rng.randint(min_coin, max_coin) for _ in range(num_coins)))
        # Guard against degenerate case where set collapses to 1 element
        if len(coins) < 2:
            coins.append(coins[0] + rng.randint(1, 3))
            coins = sorted(set(coins))

        amount = rng.randint(min_amount, max_amount)

        example = build_example(coins, amount)
        samples.append(example)

    ds = Dataset.from_list(samples)
    split = ds.train_test_split(test_size=n_eval, seed=seed)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    return train_dataset, eval_dataset


# ----------------------------
#  Reward function for GRPO
# ----------------------------

def extract_completion_text(completion: Any) -> str:
    """
    GRPO passes either:
      - Standard format: completion is a string
      - Conversational format: completion is a list of messages
    We normalize to a plain string (assistant's content).
    """
    if isinstance(completion, str):
        return completion

    if isinstance(completion, list):
        if not completion:
            return ""
        # Conversational: list of {"role":..., "content":...}
        last = completion[-1]
        if isinstance(last, dict) and "content" in last:
            return str(last["content"])
        # Sometimes it's a list with a single string
        first = completion[0]
        if isinstance(first, str):
            return first

    return str(completion)


def extract_json_obj(text: str) -> Dict[str, Any] | None:
    """
    Try to robustly extract a JSON object from the model output.
    """
    text = text.strip()
    # Best case: whole output is the JSON
    if text.startswith("{"):
        try:
            return json.loads(text)
        except Exception:
            pass

    # Fallback: substring between first "{" and last "}"
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            return json.loads(candidate)
        except Exception:
            return None
    return None


import json
from typing import Any, List


def parse_solution_string(solution_str: Any) -> List[int] | None:
    """
    Parse solution in list form, e.g. [2, 10, 20].

    Supported formats:
      - Python list/tuple: [2, 10, 20] or (2, 10, 20)
      - JSON list string:  "[2, 10, 20]"

    Returns:
      - List[int] on success
      - [] if the input is an empty list or "[]"
      - None if parsing fails
    """
    if solution_str is None:
        return None

    # 1) Already a list/tuple
    if isinstance(solution_str, (list, tuple)):
        coins: List[int] = []
        try:
            for x in solution_str:
                coins.append(int(x))
        except (TypeError, ValueError):
            return None
        return coins

    # 2) String: expect a JSON-like list representation
    s = str(solution_str).strip()
    if not s:
        # Treat empty string as empty solution
        return []

    try:
        value = json.loads(s)
    except Exception:
        return None

    if not isinstance(value, list):
        return None

    coins: List[int] = []
    try:
        for x in value:
            coins.append(int(x))
    except (TypeError, ValueError):
        return None

    return coins



def coin_change_reward(
    prompts: List[Any],
    completions: List[Any],
    coins: List[List[int]],
    amount: List[int],
    optimal_answer: List[int],
    **kwargs,
) -> List[float]:
    """
    Custom reward function for GRPO.

    Reward heuristics:
      - 1.0  if JSON parses and:
               * answer == optimal_answer AND
               * solution uses allowed coins, sums to amount, and has len == answer
      - 0.75 if answer == optimal_answer and solution sums correctly but len mismatch
      - 0.50 if solution is valid but answer != optimal_answer (non-optimal length)
      - 0.25 if answer == optimal_answer but solution invalid / unparsable
      - 0.0  otherwise
    """
    rewards: List[float] = []

    for comp, clist, amt, opt_ans in zip(completions, coins, amount, optimal_answer):
        text = extract_completion_text(comp)
        obj = extract_json_obj(text)
        reward = 0.0

        if obj is None:
            rewards.append(reward)
            continue

        ans = obj.get("answer", None)
        sol_str = obj.get("solution", "")

        try:
            ans_int = int(ans)
        except (TypeError, ValueError):
            rewards.append(reward)
            continue

        try:
            opt_ans_int = int(opt_ans)
        except (TypeError, ValueError):
            opt_ans_int = opt_ans

        clist = list(clist)
        amt = int(amt)

        sol_coins = parse_solution_string(sol_str)

        if sol_coins is None:
            # Invalid solution string
            if ans_int == opt_ans_int:
                reward = 0.25  # at least the numeric answer matches
        else:
            if ans_int == -1:
                # Model thinks it's impossible
                if opt_ans_int == -1:
                    reward = 1.0
            else:
                # Validate solution structure
                is_valid_sum = sum(sol_coins) == amt
                coins_ok = all(c in clist for c in sol_coins)

                if is_valid_sum and coins_ok:
                    if len(sol_coins) == ans_int:
                        if ans_int == opt_ans_int:
                            reward = 1.0
                        else:
                            reward = 0.5  # valid decomposition but not optimal length
                    else:
                        if ans_int == opt_ans_int:
                            reward = 0.75  # correct count but decomposition length mismatch
                        else:
                            reward = 0.25
                else:
                    if ans_int == opt_ans_int:
                        reward = 0.25

        rewards.append(float(reward))

    return rewards


# ----------------------------
#  Evaluation / vibe check
# ----------------------------

@torch.no_grad()
def evaluate_on_dataset(
    trainer: GRPOTrainer,
    eval_dataset: Dataset,
    num_examples: int = 64,
    print_examples: int = 5,
) -> None:
    """
    Simple offline evaluation after training:
      - Generate for a small eval subset
      - Parse JSON
      - Compute accuracy for:
          * numeric answer only
          * numeric answer + solution validity
      - Print a few sample generations
    """
    model = trainer.model
    tokenizer = trainer.processing_class

    model.eval()

    # Device for tokens
    try:
        device = model.device
    except AttributeError:
        device = next(model.parameters()).device

    n = min(num_examples, len(eval_dataset))
    correct_answer = 0
    correct_full = 0

    print("\n==== Offline eval on coin-change dev set ====\n")

    for idx in range(n):
        sample = eval_dataset[idx]
        messages = sample["prompt"]
        coins = sample["coins"]
        amount = int(sample["amount"])
        opt_ans = int(sample["optimal_answer"])

        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(device)

        output_ids = model.generate(
            input_ids,
            do_sample=False,
            max_new_tokens=128,
        )

        # Slice off the prompt to get only the completion
        completion_ids = output_ids[0, input_ids.shape[1] :]
        completion_text = tokenizer.decode(
            completion_ids, skip_special_tokens=True
        )

        obj = extract_json_obj(completion_text)
        parsed_ans = None
        parsed_sol = None
        if obj is not None:
            parsed_ans = obj.get("answer", None)
            parsed_sol = obj.get("solution", "")

        # Answer-only correctness
        ans_ok = False
        try:
            ans_ok = int(parsed_ans) == opt_ans
        except Exception:
            ans_ok = False

        # Full correctness (answer + valid decomposition)
        full_ok = False
        if ans_ok:
            sol_coins = parse_solution_string(parsed_sol)
            if sol_coins is not None:
                is_valid_sum = sum(sol_coins) == amount
                coins_ok = all(c in coins for c in sol_coins)
                len_ok = (opt_ans == -1 and len(sol_coins) == 0) or (
                    opt_ans != -1 and len(sol_coins) == opt_ans
                )
                full_ok = is_valid_sum and coins_ok and len_ok

        if ans_ok:
            correct_answer += 1
        if full_ok:
            correct_full += 1

        if idx < print_examples:
            print(f"[Example {idx}]")
            print(f"  coins     = {coins}")
            print(f"  amount    = {amount}")
            print(f"  expected  = {{'answer': {opt_ans}, 'solution': '{sample['optimal_solution']}'}}")
            print(f"  raw out   = {completion_text!r}")
            print(f"  parsed    = {{'answer': {parsed_ans}, 'solution': {parsed_sol!r}}}")
            print(f"  ans_ok    = {ans_ok}, full_ok = {full_ok}")
            print()

    acc_answer = correct_answer / n if n > 0 else 0.0
    acc_full = correct_full / n if n > 0 else 0.0

    print(f"Eval size: {n}")
    print(f"Answer-only accuracy: {acc_answer:.3f}")
    print(f"Full JSON+solution accuracy: {acc_full:.3f}")
    print("\n============================================\n")


# ----------------------------
#  Main: GRPO training
# ----------------------------

def main():
    # 1) Build datasets
    train_dataset, eval_dataset = make_coin_change_datasets(
        n_train=1024,
        n_eval=256,
        min_amount=4,
        max_amount=60,
    )

    # 2) Tokenizer / processing class
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3) Device configuration (CUDA / MPS / CPU)
    device_kwargs: Dict[str, Any] = {}
    if not torch.cuda.is_available():
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # Apple Silicon
            device_kwargs["use_mps_device"] = True
            print("Using MPS device.")
        else:
            # CPU fallback
            device_kwargs["use_cpu"] = True
            device_kwargs["no_cuda"] = True
            print("Using CPU only.")
    else:
        print("Using CUDA.")

    # 4) GRPO configuration
    grpo_config = GRPOConfig(
        output_dir="./lfm2-350m-coinchange-grpo",
        num_train_epochs=1.0,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=16,
        logging_steps=10,
        save_steps=10,
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=200,
        do_eval=True,
        eval_on_start=False,
        # GRPO-specific generation parameters
        num_generations=8,
        num_generations_eval=4,
        push_to_hub=False,
        trackio_space_id=None,
        max_completion_length=64,
        temperature=0.3,
        #min_p=0.15,
        torch_compile=True,
        bf16=True,
        repetition_penalty=1.05,
        # Logging some generations for "vibe check"
        log_completions=True,
        num_completions_to_print=5,
        learning_rate=8e-5,
        warmup_steps=10,

        lr_scheduler_type="cosine_warmup_with_min_lr",
        lr_scheduler_kwargs={"min_lr": 1e-5},
        # Keep extra columns (coins, amount, optimal_answer) for the reward func
        remove_unused_columns=False,
        # A small run_name for logging (wandb/trackio if you enable them)
        #run_name="lfm2-coinchange-grpo",
        **device_kwargs,
    )

    # 5) GRPO trainer
    trainer = GRPOTrainer(
        model=MODEL_ID,
        reward_funcs=coin_change_reward,
        args=grpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    # 6) Train
    trainer.train()

    # 7) Save final checkpoint
    trainer.save_model(grpo_config.output_dir)

    # 8) Offline eval + sample generations
    evaluate_on_dataset(trainer, eval_dataset, num_examples=64, print_examples=5)


if __name__ == "__main__":
    main()

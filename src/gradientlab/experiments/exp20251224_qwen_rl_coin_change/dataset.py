"""
Min Coin Change Dataset Generator with DP Thinking Trace.

Generates a dataset of coin change problems with step-by-step DP algorithm traces
for training language models on algorithmic reasoning.
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Optional, Tuple

from datasets import Dataset, DatasetDict


def solve_min_coins(
    coins: List[int], amount: int
) -> Tuple[Optional[int], Optional[List[int]]]:
    """
    Solve the min coin change problem and return the solution.

    Args:
        coins: List of coin denominations (must be positive integers)
        amount: Target amount to make

    Returns:
        (answer, solution) where:
        - answer: minimum number of coins, or None if impossible
        - solution: list of coins used sorted small to large, or None if impossible
    """
    if amount < 0:
        return None, None

    if amount == 0:
        return 0, []

    INF = amount + 1
    dp = [0] + [INF] * amount
    used_coin = [-1] * (amount + 1)

    for c in coins:
        if c <= 0:
            continue
        for x in range(c, amount + 1):
            if dp[x - c] + 1 < dp[x]:
                dp[x] = dp[x - c] + 1
                used_coin[x] = c

    if dp[amount] == INF:
        return None, None

    # Reconstruct solution
    solution = []
    curr = amount
    while curr > 0:
        c = used_coin[curr]
        solution.append(c)
        curr -= c

    # solution.sort()
    return dp[amount], solution


def generate_example(
    rng: random.Random,
    min_coin_value: int = 1,
    max_coin_value: int = 50,
    min_num_coins: int = 2,
    max_num_coins: int = 6,
    min_amount: int = 1,
    max_amount: int = 100,
) -> dict:
    """
    Generate a single coin change example with solution sequence.

    Args:
        rng: Random number generator
        min_coin_value: Minimum value for a coin denomination
        max_coin_value: Maximum value for a coin denomination
        min_num_coins: Minimum number of different coin types
        max_num_coins: Maximum number of different coin types
        min_amount: Minimum target amount
        max_amount: Maximum target amount

    Returns:
        Dictionary with keys: coins, amount, solution, answer
    """
    k = rng.randint(min_num_coins, max_num_coins)
    coins = sorted(rng.sample(range(min_coin_value, max_coin_value + 1), k))
    amount = rng.randint(min_amount, max_amount)

    answer, solution = solve_min_coins(coins, amount)

    return {
        "coins": coins,
        "amount": amount,
        "solution": solution,
        "answer": answer,
    }


def generate_coin_change_dataset(
    total_size: int,
    out_path: str,
    train_ratio: float = 0.9,
    seed: int = 42,
    min_coin_value: int = 1,
    max_coin_value: int = 50,
    min_num_coins: int = 2,
    max_num_coins: int = 6,
    min_amount: int = 1,
    max_amount: int = 100,
) -> DatasetDict:
    """
    Generate a coin change dataset with train/test splits.

    Args:
        total_size: Total number of examples to generate
        out_path: Path to save the dataset
        train_ratio: Fraction of data for training (default 0.9)
        seed: Random seed for reproducibility
        min_coin_value: Minimum value for a coin denomination
        max_coin_value: Maximum value for a coin denomination
        min_num_coins: Minimum number of different coin types
        max_num_coins: Maximum number of different coin types
        min_amount: Minimum target amount
        max_amount: Maximum target amount

    Returns:
        DatasetDict with 'train' and 'test' splits
    """
    rng = random.Random(seed)

    data = {"text": []}
    for _ in range(total_size):
        example = generate_example(
            rng=rng,
            min_coin_value=min_coin_value,
            max_coin_value=max_coin_value,
            min_num_coins=min_num_coins,
            max_num_coins=max_num_coins,
            min_amount=min_amount,
            max_amount=max_amount,
        )
        data["text"].append(json.dumps(example, indent=2))

    ds = Dataset.from_dict(data)
    test_size = 1.0 - train_ratio
    split = ds.train_test_split(test_size=test_size, seed=seed)

    dataset_dict = DatasetDict({"train": split["train"], "test": split["test"]})

    out_path_dir = Path(out_path)
    out_path_dir.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(str(out_path_dir))

    print(f"Dataset saved to {out_path}")
    print(f"  Train: {len(dataset_dict['train'])} examples")
    print(f"  Test: {len(dataset_dict['test'])} examples")

    return dataset_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a min coin change dataset with DP thinking traces."
    )
    parser.add_argument(
        "--total-size",
        type=int,
        default=10000,
        help="Total number of examples to generate (default: 10000)",
    )
    parser.add_argument(
        "--out-path",
        type=str,
        required=True,
        help="Path to save the dataset",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.9,
        help="Fraction of data for training (default: 0.9)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--min-coin-value",
        type=int,
        default=1,
        help="Minimum coin denomination (default: 1)",
    )
    parser.add_argument(
        "--max-coin-value",
        type=int,
        default=50,
        help="Maximum coin denomination (default: 50)",
    )
    parser.add_argument(
        "--min-num-coins",
        type=int,
        default=2,
        help="Minimum number of coin types (default: 2)",
    )
    parser.add_argument(
        "--max-num-coins",
        type=int,
        default=20,
        help="Maximum number of coin types (default: 6)",
    )
    parser.add_argument(
        "--min-amount",
        type=int,
        default=1,
        help="Minimum target amount (default: 1)",
    )
    parser.add_argument(
        "--max-amount",
        type=int,
        default=1000,
        help="Maximum target amount (default: 100)",
    )

    args = parser.parse_args()

    generate_coin_change_dataset(
        total_size=args.total_size,
        out_path=args.out_path,
        train_ratio=args.train_ratio,
        seed=args.seed,
        min_coin_value=args.min_coin_value,
        max_coin_value=args.max_coin_value,
        min_num_coins=args.min_num_coins,
        max_num_coins=args.max_num_coins,
        min_amount=args.min_amount,
        max_amount=args.max_amount,
    )

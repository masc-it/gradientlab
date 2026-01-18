"""
Minimum Coin Change dataset generator for grokking experiments.

Generates synthetic data for the minimum coin change problem using dynamic programming.
Given an amount and available coin denominations, the model learns to output the minimum
number of coins needed to make that amount.
"""

import random
from pathlib import Path
from typing import List, Optional, Tuple

from datasets import Dataset, DatasetDict


def solve_min_coin_change(amount: int, coins: List[int]) -> Tuple[int, List[int]]:
    """
    Solve the minimum coin change problem using dynamic programming.

    Args:
        amount: Target amount to make change for
        coins: List of available coin denominations

    Returns:
        Tuple of (minimum number of coins, list of coins used)
        Returns (-1, []) if amount cannot be made with given coins
    """
    if amount == 0:
        return 0, []

    # DP array: dp[i] = minimum coins needed to make amount i
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0

    # Track which coin was used to reach each amount
    coin_used = [-1] * (amount + 1)

    for i in range(1, amount + 1):
        for coin in coins:
            if coin <= i and dp[i - coin] + 1 < dp[i]:
                dp[i] = dp[i - coin] + 1
                coin_used[i] = coin

    if dp[amount] == float('inf'):
        return -1, []

    # Reconstruct the coins used
    result_coins = []
    current_amount = amount
    while current_amount > 0:
        coin = coin_used[current_amount]
        result_coins.append(coin)
        current_amount -= coin

    return int(dp[amount]), sorted(result_coins, reverse=True)


def format_coin_change_problem(
    amount: int,
    coins: List[int],
    include_coins: bool = False
) -> str:
    """
    Format the coin change problem as input string.

    Args:
        amount: Target amount
        coins: Available coin denominations
        include_coins: Whether to include coin denominations in input

    Returns:
        Formatted input string
    """
    if include_coins:
        coins_str = ",".join(map(str, sorted(coins, reverse=True)))
        return f"{amount}|{coins_str}"
    else:
        return str(amount)


def format_solution(min_coins: int, coins_used: List[int], output_type: str = "count") -> str:
    """
    Format the solution as target string.

    Args:
        min_coins: Minimum number of coins needed
        coins_used: List of coins used in solution
        output_type: One of "count" (just number), "coins" (list of coins), or "both"

    Returns:
        Formatted target string
    """
    if output_type == "count":
        return str(min_coins)
    elif output_type == "coins":
        return ",".join(map(str, coins_used))
    elif output_type == "both":
        coins_str = ",".join(map(str, coins_used))
        return f"{min_coins}|{coins_str}"
    else:
        raise ValueError(f"Unknown output type: {output_type}")


def generate_min_coin_change_dataset(
    save_path: str,
    seed: int = 42,
    min_amount: int = 1,
    max_amount: int = 100,
    coin_sets: Optional[List[List[int]]] = None,
    include_coins_in_input: bool = False,
    output_type: str = "count",
    train_split: float = 0.5,
):
    """
    Generate Minimum Coin Change dataset.

    Args:
        save_path: Path to save the dataset
        seed: Random seed for reproducibility
        min_amount: Minimum amount to generate
        max_amount: Maximum amount to generate
        coin_sets: List of coin denomination sets to use. If None, uses standard US coins
        include_coins_in_input: Whether to include coin denominations in input
        output_type: Type of output - "count", "coins", or "both"
        train_split: Fraction of data to use for training (rest goes to test)
    """
    random.seed(seed)

    # Default to US coin denominations
    if coin_sets is None:
        coin_sets = [[1, 5, 10, 25]]  # pennies, nickels, dimes, quarters

    examples = []

    print("Generating coin change problems...")
    for coins in coin_sets:
        coins_sorted = sorted(coins)
        print(f"  Using coin set: {coins_sorted}")

        for amount in range(min_amount, max_amount + 1):
            min_coins, coins_used = solve_min_coin_change(amount, coins_sorted)

            if min_coins == -1:
                # Skip amounts that cannot be made (shouldn't happen with coins including 1)
                continue

            input_str = format_coin_change_problem(amount, coins_sorted, include_coins_in_input)
            target_str = format_solution(min_coins, coins_used, output_type)

            examples.append({
                "input": input_str,
                "target": target_str,
                "amount": amount,
                "coins": coins_sorted,
            })

    print(f"Generated {len(examples)} coin change examples")

    # Shuffle and split
    random.shuffle(examples)
    train_size = int(len(examples) * train_split)
    train_data = examples[:train_size]
    eval_data = examples[train_size:]

    print(f"Train: {len(train_data)} examples ({train_split*100:.1f}%)")
    print(f"Test: {len(eval_data)} examples ({(1-train_split)*100:.1f}%)")

    # Create HuggingFace DatasetDict
    dataset_dict = DatasetDict({
        "train": Dataset.from_list(train_data),
        "test": Dataset.from_list(eval_data),
    })

    # Save to disk
    save_path_obj = Path(save_path)
    save_path_obj.parent.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(str(save_path_obj))
    print(f"Dataset saved to {save_path}")

    # Print some examples
    print("\nSample examples from training set:")
    for i in range(min(10, len(train_data))):
        example = train_data[i]
        print(f"  {example['input']} -> {example['target']}")

    # Print statistics
    if output_type in ["count", "both"]:
        train_targets = [int(ex['target'].split('|')[0]) if '|' in ex['target'] else int(ex['target'])
                        for ex in train_data]
        print(f"\nTarget statistics (train):")
        print(f"  Min coins needed: {min(train_targets)}")
        print(f"  Max coins needed: {max(train_targets)}")
        print(f"  Avg coins needed: {sum(train_targets) / len(train_targets):.2f}")


if __name__ == "__main__":
    # Default save path
    #default_path = Path(__file__).parent.parent / "data" / "min_coin_change"
    default_path = "/media/mascit/Lexar/datasets/min_coin_change/"

    # Generate dataset with different configurations
    # Example 1: Simple - just predict the count

    generate_min_coin_change_dataset(
        str(default_path),
        seed=42,
        min_amount=1,
        max_amount=2000,  # 3500 * 6 coin sets = 21,000 examples
        coin_sets=[
            [1, 5, 10, 25, 50],      # US coins with half-dollar
            [1, 10, 50],             # Sparse set
            [1, 10, 20],             # Alternative sparse
            [1, 5, 10, 25],          # US coins without half-dollar
            [1, 2, 5, 10, 20, 50],   # Euro-like
            [1, 3, 7, 15],           # Non-standard (harder)
            [1, 3, 5],           # Non-standard (harder)
            [1, 3],           # Non-standard (harder)
            [1, 7],           # Non-standard (harder)
            [1, 2, 3, 4],           # Non-standard (harder)
        ],
        include_coins_in_input=True,
        output_type="count",
        train_split=0.2,
    )
    
    """ generate_min_coin_change_dataset(
        str(default_path),
        seed=42,
        min_amount=1,
        max_amount=500,
        coin_sets=[[1, 5, 10, 25, 50], [1, 10, 50], [1, 10, 20]],  # US coins
        include_coins_in_input=True,
        output_type="count",
        train_split=0.05,
    ) """

    # Uncomment for alternative configurations:

    # Example 2: Predict both count and coins used
    # generate_min_coin_change_dataset(
    #     str(default_path / "with_coins"),
    #     seed=42,
    #     min_amount=1,
    #     max_amount=100,
    #     coin_sets=[[1, 5, 10, 25]],
    #     include_coins_in_input=False,
    #     output_type="both",
    #     train_split=0.5,
    # )

    # Example 3: Multiple coin systems (US + Euro-like)
    # generate_min_coin_change_dataset(
    #     str(default_path / "multi_currency"),
    #     seed=42,
    #     min_amount=1,
    #     max_amount=100,
    #     coin_sets=[
    #         [1, 5, 10, 25],      # US coins
    #         [1, 2, 5, 10, 20, 50],  # Euro-like coins
    #     ],
    #     include_coins_in_input=True,  # Need to specify which coins are available
    #     output_type="count",
    #     train_split=0.5,
    # )

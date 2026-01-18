"""
Date to ISO dataset generator for grokking experiments.

Generates dates from 1900-01-01 to 2100-01-01 in various formats and converts them to ISO format.
"""

import random
from datetime import datetime, timedelta
from pathlib import Path

from datasets import Dataset, DatasetDict


def format_date(date: datetime, format_type: str) -> str:
    """
    Format a date according to the specified format type.

    Args:
        date: datetime object
        format_type: One of "eu", "written_long", "written_short", "iso"

    Returns:
        Formatted date string

    Note:
        Using EU format (DD/MM/YYYY) is unambiguous when used alone.
        Each date has a unique representation.
    """
    if format_type == "eu":
        # EU format: DD/MM/YYYY
        return date.strftime("%d/%m/%Y")
    elif format_type == "written_long":
        # Written long: "January 15, 2000"
        return date.strftime("%B %d, %Y")
    elif format_type == "written_short":
        # Written short: "15 Jan 2000"
        return date.strftime("%d %b %Y")
    elif format_type == "iso":
        # ISO format: YYYY-MM-DD (identity mapping)
        return date.strftime("%Y-%m-%d")
    else:
        raise ValueError(f"Unknown format type: {format_type}")


def generate_date_to_iso_dataset(save_path: str, seed: int = 42):
    """
    Generate Date to ISO conversion dataset.

    Args:
        save_path: Path to save the dataset
        seed: Random seed for reproducibility
    """
    random.seed(seed)

    start_date = datetime(1500, 1, 1)
    end_date = datetime(2500, 1, 1)

    dates = []
    current_date = start_date

    print("Generating dates...")
    while current_date < end_date:
        # ISO format (target)
        iso_date = current_date.strftime("%Y-%m-%d")

        # Randomly pick one input format
        format_type = random.choice(["eu", "written_long", "written_short", "iso"])
        input_date = format_date(current_date, format_type)

        dates.append({"input": input_date, "target": iso_date})
        current_date += timedelta(days=1)

    print(f"Generated {len(dates)} date examples")

    # Shuffle and split
    random.shuffle(dates)
    train_size = int(len(dates) * 0.01)
    train_data = dates[:train_size]
    eval_data = dates[train_size:]

    print(f"Train: {len(train_data)} examples (1%)")
    print(f"Eval: {len(eval_data)} examples (99%)")

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
    print("\nSample examples:")
    for i in range(min(5, len(train_data))):
        example = train_data[i]
        print(f"  {example['input']} -> {example['target']}")


if __name__ == "__main__":
    # Default save path
    default_path = Path(__file__).parent.parent / "data" / "date_to_iso"
    generate_date_to_iso_dataset(str(default_path))

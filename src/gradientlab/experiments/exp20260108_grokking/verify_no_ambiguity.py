"""
Verify that the dataset has no ambiguous inputs.

An ambiguous input is one that maps to multiple different targets.
"""

from collections import defaultdict
from datasets import load_from_disk


def verify_no_ambiguity():
    """Check for ambiguous inputs in the dataset."""

    print("=" * 80)
    print("VERIFYING DATASET HAS NO AMBIGUOUS INPUTS")
    print("=" * 80)

    # Load dataset
    ds_path = "src/gradientlab/experiments/exp20260108_grokking/data/date_to_iso"
    try:
        ds = load_from_disk(ds_path)
    except FileNotFoundError:
        print(f"\n❌ Dataset not found at {ds_path}")
        print("Please generate the dataset first:")
        print("  python src/gradientlab/experiments/exp20260108_grokking/dataset/date_to_iso.py")
        return

    print(f"\nDataset loaded:")
    print(f"  Train: {len(ds['train'])} examples")
    print(f"  Test:  {len(ds['test'])} examples")
    print()

    # Check both splits
    for split_name in ["train", "test"]:
        print("=" * 80)
        print(f"CHECKING {split_name.upper()} SPLIT")
        print("=" * 80)

        split = ds[split_name]

        # Map input → list of targets
        input_to_targets = defaultdict(list)

        for example in split:
            input_str = example['input']
            target_str = example['target']
            input_to_targets[input_str].append(target_str)

        # Find ambiguous inputs (map to multiple targets)
        ambiguous = {}
        for input_str, targets in input_to_targets.items():
            unique_targets = set(targets)
            if len(unique_targets) > 1:
                ambiguous[input_str] = unique_targets

        # Report results
        total_inputs = len(input_to_targets)
        print(f"\nTotal unique inputs: {total_inputs}")
        print(f"Total examples: {len(split)}")
        print(f"Ambiguous inputs: {len(ambiguous)}")

        if ambiguous:
            print("\n❌ FOUND AMBIGUOUS INPUTS:")
            for i, (input_str, targets) in enumerate(sorted(ambiguous.items())[:10]):
                print(f"\n  {i+1}. Input: {input_str!r}")
                print(f"     Maps to: {sorted(targets)}")
            if len(ambiguous) > 10:
                print(f"\n  ... and {len(ambiguous) - 10} more")
        else:
            print("\n✅ NO AMBIGUOUS INPUTS FOUND!")

        print()

    # Show format distribution
    print("=" * 80)
    print("INPUT FORMAT DISTRIBUTION")
    print("=" * 80)

    format_counts = defaultdict(int)

    for split_name in ["train", "test"]:
        split = ds[split_name]
        for example in split:
            input_str = example['input']

            # Detect format
            if input_str[0].isdigit() and " " in input_str and input_str[-4:].isdigit():
                # "15 Jan 2000" format
                format_type = "written_short"
            elif input_str[0].isalpha() and "," in input_str:
                # "January 15, 2000" format
                format_type = "written_long"
            elif "-" in input_str:
                # "2000-01-15" format (ISO)
                format_type = "iso"
            else:
                format_type = "unknown"

            format_counts[format_type] += 1

    print("\nFormat counts across entire dataset:")
    for format_type, count in sorted(format_counts.items()):
        percentage = count / sum(format_counts.values()) * 100
        print(f"  {format_type:15} : {count:7,} ({percentage:5.2f}%)")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    total_ambiguous = sum(
        len([v for v in input_to_targets.values() if len(set(v)) > 1])
        for input_to_targets in [
            defaultdict(list, {ex['input']: [ex['target']] for ex in ds[split]})
            for split in ["train", "test"]
        ]
    )

    if total_ambiguous == 0:
        print("\n✅ DATASET IS VALID - NO AMBIGUITIES FOUND")
        print("\nThe model can learn a deterministic mapping from each input to its target.")
        print("This is necessary for the grokking phenomenon to occur.")
    else:
        print(f"\n❌ DATASET HAS {total_ambiguous} AMBIGUOUS INPUTS")
        print("\nThe model CANNOT learn this dataset correctly.")
        print("Please regenerate the dataset with only unambiguous formats.")

    print()


if __name__ == "__main__":
    verify_no_ambiguity()

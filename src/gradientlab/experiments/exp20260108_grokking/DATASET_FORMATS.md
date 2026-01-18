# Dataset Format Configuration

## Current Formats (Unambiguous)

The dataset uses **4 formats**, all unambiguous:

1. **EU Format**: `DD/MM/YYYY` (e.g., `24/01/2000` → `2000-01-24`)
2. **Written Long**: `Month DD, YYYY` (e.g., `January 24, 2000` → `2000-01-24`)
3. **Written Short**: `DD Mon YYYY` (e.g., `24 Jan 2000` → `2000-01-24`)
4. **ISO Format**: `YYYY-MM-DD` (e.g., `2000-01-24` → `2000-01-24`)

Each format produces a unique string for each date. No ambiguity exists.

## Why Not Both US and EU Formats?

Including **both** US (`MM/DD/YYYY`) and EU (`DD/MM/YYYY`) formats creates contradictory examples:

### The Ambiguity Problem

The string `"01/05/2000"` would appear **twice** with **different targets**:

1. **From date 2000-01-05 (January 5th):**
   - US format: `01/05/2000` → Target: `2000-01-05` ✓

2. **From date 2000-05-01 (May 1st):**
   - EU format: `01/05/2000` → Target: `2000-05-01` ✓

**Result:** Same input, two different outputs = impossible to learn!

### Scale of the Problem

- Ambiguous when: `day ≤ 12` AND `day ≠ month`
- Examples per year: ~132 ambiguous date strings
- Total over 1000 years: ~132,000 contradictory examples

### Which Dates Are Ambiguous?

Examples of strings that would appear twice:
- `01/02/2000` - Could be Jan 2 (US) or Feb 1 (EU)
- `01/03/2000` - Could be Jan 3 (US) or Mar 1 (EU)
- `12/11/2000` - Could be Dec 11 (US) or Nov 12 (EU)
- ... and so on for all dates where 1 ≤ day ≤ 12 and day ≠ month

### Which Dates Are NOT Ambiguous?

When including both US and EU:
- `13/01/2000` - Only valid as EU (month 13 doesn't exist in US)
- `01/13/2000` - Only valid as US (month 13 doesn't exist in EU)
- `01/01/2000` - Unambiguous (day = month, same in both formats)

But ~35% of dates would still be ambiguous.

## Current Solution: EU Format Only

By using **only EU format** (not both US and EU):
- ✅ Each date has exactly one EU representation
- ✅ No contradictory examples in the dataset
- ✅ Model can achieve 100% accuracy
- ✅ Task is deterministic and learnable

Combined with the other 3 unambiguous formats, the dataset has:
- **4 formats total**
- **0 ambiguous examples**
- **Fully learnable task**

## Regenerating the Dataset

After updating format configuration:

```bash
# Delete old dataset if it exists
rm -rf src/gradientlab/experiments/exp20260108_grokking/data/date_to_iso

# Generate new dataset with EU format
uv run python src/gradientlab/experiments/exp20260108_grokking/dataset/date_to_iso.py

# Verify no ambiguities
uv run python src/gradientlab/experiments/exp20260108_grokking/verify_no_ambiguity.py
```

Expected output:
```
Total unique inputs: ~365,000 (one per day × 4 formats / 4)
Ambiguous inputs: 0
✅ DATASET IS VALID - NO AMBIGUITIES FOUND
```

## Format Distribution

With 4 formats chosen uniformly at random:
- EU format: ~25%
- Written long: ~25%
- Written short: ~25%
- ISO: ~25%

Each format provides different learning signals to the model.

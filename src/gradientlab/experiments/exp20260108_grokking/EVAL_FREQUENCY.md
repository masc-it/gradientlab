# Evaluation Frequency Configuration

## Overview

The `eval_every_n_epochs` parameter in [exp_config.py](exp_config.py) controls how often full evaluation runs during training.

## Why This Matters

Full evaluation includes:
- Teacher-forced accuracy (fast)
- **Autoregressive generation accuracy** (slow - generates 100 samples)
- **Generation examples display** (shows 3 sample predictions)

For long training runs (thousands of epochs), running full eval every epoch adds significant overhead.

## Configuration

```python
class ExpConfig(BaseModel):
    eval_every_n_epochs: int = 50  # Run full eval every N epochs
```

### Recommended Settings

| Training Duration | Recommended Value | Rationale |
|------------------|-------------------|-----------|
| < 100 epochs | `1` | Minimal overhead, detailed monitoring |
| 100-1000 epochs | `10` | Balanced overhead vs monitoring |
| 1000+ epochs | `50` | Significant speedup for long runs |

## Behavior

### When Evaluation Runs

1. **Every N epochs**: When `(epoch + 1) % eval_every_n_epochs == 0`
2. **Last epoch**: Always runs on final epoch regardless of N

### Example with `eval_every_n_epochs = 10`

```
Epoch 1: train only (eval skipped, next in 9 epoch(s))
Epoch 2: train only (eval skipped, next in 8 epoch(s))
...
Epoch 9: train only (eval skipped, next in 1 epoch(s))
Epoch 10: train + FULL EVAL ✅
Epoch 11: train only (eval skipped, next in 9 epoch(s))
...
Epoch 20: train + FULL EVAL ✅
...
Epoch 1000 (last): train + FULL EVAL ✅  (even if not a multiple of 10)
```

## Output Format

### With Evaluation (every Nth epoch):
```
Epoch 50: train_loss=0.1234, train_acc=0.9567, eval_loss=0.2345, eval_tf_acc=0.9456, eval_ar_acc=0.8901, lr=0.001000

Computing autoregressive accuracy on 100 samples...
Autoregressive accuracy: 89/100 = 0.8901

======================================================================
GENERATION EXAMPLES
======================================================================
...
```

### Without Evaluation (skipped epochs):
```
Epoch 51: train_loss=0.1230, train_acc=0.9570, lr=0.001000 (eval skipped, next in 49 epoch(s))
```

## Impact on Checkpointing

- Checkpoints are saved **only during evaluation epochs**
- Best checkpoint is based on **autoregressive accuracy**
- If you use `eval_every_n_epochs = 50`, you'll only save checkpoints every 50 epochs
- The last epoch always evaluates and may save a final checkpoint

## Impact on Auto-Stopping

Auto-stopping (when `eval_ar_acc >= target_eval_accuracy`) only checks during evaluation epochs.

Example with `eval_every_n_epochs = 10` and `target_eval_accuracy = 0.995`:
- Model might reach 99.5% accuracy at epoch 87
- But stopping check only happens at epoch 90 (next eval epoch)
- Training continues for 3 extra epochs

This is acceptable for grokking experiments where you want to see the full learning curve.

## Metrics Logged

### Every Epoch (always):
- `train_loss`
- `train_accuracy`
- `lr`

### Evaluation Epochs Only:
- `eval_loss`
- `eval_teacher_forced_accuracy`
- `eval_autoregressive_accuracy`

## Example Usage

### For Quick Testing (detailed monitoring):
```python
exp_cfg = ExpConfig()
exp_cfg.eval_every_n_epochs = 1  # Eval every epoch
exp_cfg.num_epochs = 100  # Short run
```

### For Production Grokking Run (efficient):
```python
exp_cfg = ExpConfig()
exp_cfg.eval_every_n_epochs = 50  # Eval every 50 epochs
exp_cfg.num_epochs = 20000  # Long run for grokking
```

With `eval_every_n_epochs = 50`:
- Total evaluations: ~400 (20000 / 50)
- vs eval every epoch: 20,000 evaluations
- **50x speedup** on evaluation overhead!

## Trade-offs

### Higher Values (e.g., 50):
✅ **Pros:**
- Much faster training
- Reduced disk I/O (fewer checkpoints)
- Less computational overhead

❌ **Cons:**
- Miss fine-grained tracking of eval metrics
- Delayed detection of overfitting/grokking moment
- Coarser checkpoint granularity

### Lower Values (e.g., 1):
✅ **Pros:**
- Detailed monitoring of all metrics
- Fine-grained checkpoints
- Immediate detection of grokking moment

❌ **Cons:**
- Significant overhead for long training
- More frequent generation (slow)
- More disk space for checkpoints

## Recommendation for Grokking

Use `eval_every_n_epochs = 50` for efficiency:
- Grokking happens gradually over many epochs
- You don't need to see every single epoch's eval metrics
- The grokking curve will still be visible at 50-epoch granularity
- Training is 10-50x faster depending on eval overhead

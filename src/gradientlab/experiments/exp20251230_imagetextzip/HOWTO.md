# ImageTextSlot Model - Usage Guide

## Overview

ImageTextSlot is a non-autoregressive text recognition model that uses CNN encoder + cross-attention slot mechanism to decode text from 128x128 grayscale images. All 1024 character slots are predicted in parallel, making inference very fast.

**Key Features:**
- **Non-autoregressive**: All tokens predicted simultaneously
- **Fixed compute**: O(1024×16) cross-attention regardless of text length
- **Dual heads**: Classification (text prediction) + Counter (sequence length prediction)
- **Model size**: ~4.5-5M parameters

---

## Table of Contents

1. [Model Instantiation](#1-model-instantiation)
2. [Forward Pass (Training)](#2-forward-pass-training)
3. [Loss Functions](#3-loss-functions)
4. [Dataset Preparation](#4-dataset-preparation)
5. [Inference/Generation](#5-inferencegeneration)
6. [Training Tips](#6-training-tips)
7. [Expected Metrics](#7-expected-metrics)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Model Instantiation

### Using ModelFactory

The simplest way to instantiate the model is via `ModelFactory.build_5m()`:

```python
from gradientlab.experiments.exp20251230_imagetextzip.modeling.factory import ModelFactory

# Create fresh model
model, tokenizer, model_cfg = ModelFactory.build_5m()

# Or resume from checkpoint
model, tokenizer, model_cfg = ModelFactory.build_5m(
    resume_from="path/to/checkpoint/dir"
)
```

The factory returns:
- `model`: ImageTextSlotModel instance (~4.5M params)
- `tokenizer`: Byte-level tokenizer with V=512 vocab size
- `model_cfg`: ModelConfig with all hyperparameters

### Manual Instantiation

For custom configurations:

```python
from gradientlab.experiments.exp20251230_imagetextzip.modeling.model import (
    ModelConfig,
    CNNEncoderConfig,
    SlotAttentionConfig,
    SequenceCounterConfig,
    ImageTextSlotModel,
)
from gradientlab.tokenizers.byte_tokenizer import byte_tokenizer

tokenizer = byte_tokenizer()

# Customize configs
encoder_cfg = CNNEncoderConfig(
    stem_channels=96,  # Larger encoder
    depths=(2, 2, 8, 2),  # More blocks in stage 3
    dims=(96, 192, 384, 768),  # Wider channels
)

slot_attn_cfg = SlotAttentionConfig(
    num_slots=1024,
    d_model=512,  # Wider slots
    num_heads=8,
)

counter_cfg = SequenceCounterConfig(
    d_model=512,
    hidden_dim=256,
)

model_cfg = ModelConfig(
    encoder=encoder_cfg,
    slot_attention=slot_attn_cfg,
    counter=counter_cfg,
    vocab_size=512,
    pad_token_id=tokenizer.pad_token_id,
    num_slots=1024,
    label_smoothing=0.1,
    counter_loss_weight=0.1,
)

model = ImageTextSlotModel(model_cfg)
```

### Model Architecture Summary

```
Input: (B, 1, 128, 128) grayscale images

CNN Encoder (ConvNeXt-inspired):
  Stem: Conv4x4/4 → (B, 80, 32, 32)
  Stage 1: 2 blocks @ 80 dims → (B, 80, 32, 32)
  Stage 2: 2 blocks @ 160 dims → (B, 160, 16, 16)
  Stage 3: 6 blocks @ 320 dims → (B, 320, 8, 8)
  Stage 4: 2 blocks @ 640 dims → (B, 640, 4, 4)
  Flatten: → (B, 16, 640)

Projection: → (B, 16, 384)

Slot Attention:
  Queries: 1024 learnable slots
  Cross-attention: slots × CNN features
  SwiGLU FFN
  Output: (B, 1024, 384)

Dual Heads:
  Classifier: → (B, 1024, 512 vocab)
  Counter: → (B, 1) sequence length
```

---

## 2. Forward Pass (Training)

### Input Format

The model expects a batch dict with the following keys:

```python
batch = {
    "pixel_values": torch.Tensor,    # (B, 1, 128, 128), normalized grayscale images
    "input_ids": torch.Tensor,       # (B, seq_len), tokenized text
    "attention_mask": torch.Tensor,  # (B, seq_len), 1 for real tokens, 0 for padding
    "labels": torch.Tensor,          # (B, seq_len), target tokens (clone of input_ids)
}
```

**Important:** The existing `torch_dataset.py` and `Collate` class already handle this format correctly. Images are normalized with `mean=0.1533, std=0.3034`, and text is wrapped with `<|im_start|>` and `<|im_end|>` special tokens.

### Forward Call

```python
import torch

# Example forward pass
model.train()
device = torch.device("cuda")
model.to(device)

# Move batch to device
batch = {k: v.to(device) for k, v in batch.items()}

# Forward
output = model(**batch)

# Output dict contains:
print(output.keys())
# dict_keys(['logits', 'count_pred', 'loss', 'loss_ce', 'loss_count'])
```

### Output Format

```python
output = {
    "logits": torch.Tensor,      # (B, 1024, 512) raw logits for each slot
    "count_pred": torch.Tensor,  # (B, 1) predicted character count
    "loss": torch.Tensor,        # () scalar total loss (if labels provided)
    "loss_ce": torch.Tensor,     # () classification loss component
    "loss_count": torch.Tensor,  # () counting loss component
}
```

### Training Loop Example

```python
from torch.optim import AdamW
from gradientlab.neuralblocks.optim.adamw_params import get_adamw_parameters

# Setup optimizer
optimizer = AdamW(
    get_adamw_parameters(model, weight_decay=1e-3),
    lr=1e-4,
    betas=(0.9, 0.95),
)

# Training step
for batch in dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}

    optimizer.zero_grad()
    output = model(**batch)
    loss = output["loss"]

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    # Log losses
    print(f"Loss: {loss.item():.4f}, CE: {output['loss_ce'].item():.4f}, Count: {output['loss_count'].item():.4f}")
```

---

## 3. Loss Functions

### Classification Loss (Cross-Entropy)

Predicts the character at each of the 1024 slots.

```python
# Internally computed as:
targets = _prepare_targets(labels)  # Pad/truncate to 1024

loss_ce = F.cross_entropy(
    logits.view(-1, 512),           # (B*1024, 512)
    targets.view(-1),                # (B*1024,)
    ignore_index=pad_token_id,       # Don't penalize padding slots
    label_smoothing=0.1,             # Prevents overconfidence
)
```

**Key points:**
- Labels automatically padded to 1024 slots with `pad_token_id`
- Padding slots ignored in loss via `ignore_index`
- Label smoothing (ε=0.1) improves generalization

### Counting Loss (Smooth L1)

Predicts the actual number of characters in the image.

```python
# Actual character counts from attention mask
actual_counts = attention_mask.sum(dim=1).float()  # (B,)

# Predicted counts
count_pred = model.counter(slot_features).squeeze(-1)  # (B,)

# Smooth L1 loss (Huber loss with δ=1.0)
loss_count = F.smooth_l1_loss(count_pred, actual_counts)
```

**Key points:**
- Uses attention_mask to count real (non-padding) tokens
- Smooth L1 more robust to outliers than MSE
- Provides supervision signal for truncating predictions

### Total Loss

```python
total_loss = loss_ce + counter_loss_weight * loss_count
# Default: counter_loss_weight = 0.1
```

The counter loss is auxiliary (weighted 0.1×) to avoid interfering with primary classification task.

---

## 4. Dataset Preparation

### Dataset Format

The dataset should be a HuggingFace `Dataset` with two columns:

```python
from datasets import Dataset

data = {
    "pixel_values": [PIL.Image, ...],  # Grayscale images (any size)
    "text": ["Hello world", ...],      # Text strings
}

dataset = Dataset.from_dict(data)
```

### Data Processing Pipeline

The existing `torch_dataset.py` handles everything:

```python
from gradientlab.experiments.exp20251230_imagetextzip.torch_dataset import (
    MyDataset,
    Collate,
)
from datasets import load_from_disk
from torch.utils.data import DataLoader

# Load dataset
ds = load_from_disk("path/to/dataset")
train_ds = ds["train"]

# Wrap in PyTorch dataset
torch_ds = MyDataset(train_ds)

# Create dataloader with collate function
tokenizer = byte_tokenizer()
collate_fn = Collate(tokenizer)

dataloader = DataLoader(
    torch_ds,
    batch_size=64,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=4,
)
```

### What MyDataset Does

```python
class MyDataset:
    def __getitem__(self, index):
        sample = self.ds[index]
        return {
            "pixel_values": self.transforms(sample["pixel_values"]),
            "text": f"<|im_start|>{sample['text']}<|im_end|>",
        }
```

**Transforms:**
- ToImage() → Convert PIL to tensor
- ToDtype(torch.float32, scale=True) → Normalize to [0, 1]
- Normalize(mean=0.1533, std=0.3034) → Standardize pixel values

**Text wrapping:**
- Adds `<|im_start|>` and `<|im_end|>` special tokens
- Tokenized in Collate function

### What Collate Does

```python
class Collate:
    def __call__(self, batch):
        images = torch.stack([el["pixel_values"] for el in batch])
        encoded = self.tokenizer(
            [el["text"] for el in batch],
            padding="longest",
            padding_side="right",
            return_tensors="pt",
            pad_to_multiple_of=8,
        )
        return {
            "pixel_values": images,
            **encoded,  # input_ids, attention_mask
            "labels": encoded["input_ids"].clone(),
        }
```

**Collate output:**
- `pixel_values`: (B, 1, 128, 128)
- `input_ids`: (B, seq_len) - padded to longest in batch
- `attention_mask`: (B, seq_len) - 1 for real, 0 for padding
- `labels`: (B, seq_len) - clone of input_ids

---

## 5. Inference/Generation

### Generation Method

```python
@torch.no_grad()
def generate(
    pixel_values,      # (B, 1, 128, 128)
    bos_token_id,      # Not used (kept for API compatibility)
    eos_token_id,      # Not used
    max_new_tokens,    # Not used (all 1024 slots predicted)
):
    # Returns: List of (seq_len,) tensors
```

**Non-autoregressive generation:**
1. Encode image through all components
2. Predict all 1024 slots in parallel (argmax)
3. Use counter prediction to determine sequence length
4. Truncate each sequence to predicted length

### Example Usage

```python
model.eval()
device = torch.device("cuda")

# Prepare image
image = ...  # PIL.Image or numpy array
from torchvision.transforms import v2

transforms = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize((0.1533,), (0.3034,)),
])

pixel_values = transforms(image).unsqueeze(0).to(device)  # (1, 1, 128, 128)

# Generate
bos_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
eos_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

predictions = model.generate(
    pixel_values,
    bos_token_id=bos_id,
    eos_token_id=eos_id,
    max_new_tokens=256,
)

# Decode
text = tokenizer.decode(predictions[0])
print(f"Predicted: {text}")
```

### Batch Generation

```python
# Multiple images
pixel_values = torch.stack([transforms(img) for img in images]).to(device)
# (B, 1, 128, 128)

predictions = model.generate(pixel_values, bos_id, eos_id, 256)
# Returns list of B tensors, each with variable length

texts = [tokenizer.decode(pred) for pred in predictions]
```

### Generation Speed

Since all slots are predicted in parallel:
- **Training forward pass**: ~same speed as any other transformer
- **Inference**: Much faster than autoregressive (no sequential decoding loop)
- **Time complexity**: O(1) passes through model (vs O(L) for autoregressive)

Example: Predicting 256 characters takes same time as 16 characters!

---

## 6. Training Tips

### Recommended Hyperparameters

```python
batch_size = 64
num_epochs = 50
max_lr = 1e-4
min_lr = 3e-5
warmup_ratio = 0.1
weight_decay = 1e-3
gradient_clip = 1.0
```

These are already configured in `exp_config.py`.

### Optimizer Setup

Use AdamW with proper parameter grouping (no decay on norms/biases):

```python
from gradientlab.neuralblocks.optim.adamw_params import get_adamw_parameters

params = get_adamw_parameters(model, weight_decay=1e-3)
optimizer = AdamW(params, lr=1e-4, betas=(0.9, 0.95))
```

### Learning Rate Schedule

Cosine schedule with linear warmup:

```python
from gradientlab.neuralblocks.schedulers.cosine_with_warmup import (
    get_cosine_scheduler_with_warmup
)

total_steps = len(dataloader) * num_epochs
warmup_steps = int(total_steps * 0.1)

scheduler = get_cosine_scheduler_with_warmup(
    optimizer,
    total_steps=total_steps,
    warmup_steps=warmup_steps,
    min_lr=3e-5,
)

# Step after each batch
scheduler.step()
```

### Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()

    with autocast(dtype=torch.bfloat16):
        output = model(**batch)
        loss = output["loss"]

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()
```

### Monitoring Training

**Key metrics to track:**
- `loss`: Total loss (should decrease steadily)
- `loss_ce`: Classification loss (primary signal)
- `loss_count`: Counter loss (should converge quickly)
- `grad_norm`: Gradient norm before clipping (check for instability)
- `lr`: Learning rate (verify schedule)

**Sample generations:**
- Run `model.generate()` every N steps on validation samples
- Compare predicted vs ground truth text
- Watch for improvements over time

### Curriculum Learning (Optional)

If training is unstable or slow to converge:

1. **Start with shorter sequences**
   - Filter dataset to texts <256 chars
   - Train for few epochs
   - Gradually increase max length

2. **Progressive slot count**
   - Start with 512 slots instead of 1024
   - Fine-tune with full 1024 slots

3. **Warmup counter loss weight**
   - Start with lower weight (0.05)
   - Increase to 0.1 after few epochs

### Data Augmentation (Optional)

For better generalization:

```python
transforms = v2.Compose([
    v2.ToImage(),
    v2.RandomAffine(degrees=0, translate=(0.02, 0.02)),  # Slight shifts
    v2.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),    # Blur augmentation
    v2.ToDtype(torch.float32, scale=True),
    v2.ColorJitter(brightness=0.2, contrast=0.2),        # Brightness/contrast
    v2.Normalize((0.1533,), (0.3034,)),
])
```

**Caution:** Too much augmentation can hurt OCR performance. Start conservative.

---

## 7. Expected Metrics

### Training Progress

**Early training (Epoch 1-5):**
- Total loss: 5.0 → 2.0
- CE loss: 4.5 → 1.8
- Count loss: 50 → 10 (MAE in characters)
- Counter should converge quickly!

**Mid training (Epoch 10-25):**
- Total loss: 1.5 → 0.8
- CE loss: 1.4 → 0.7
- Count loss: 5 → 2
- Sample generations start making sense

**Late training (Epoch 25-50):**
- Total loss: 0.6 → 0.3
- CE loss: 0.5 → 0.2
- Count loss: <2 (very accurate length prediction)
- Character accuracy: 80-90%+

### Target Metrics (After 50 Epochs)

- **Validation CE loss**: <0.5
- **Character accuracy**: >90%
- **Exact match accuracy**: >70% (entire sequence correct)
- **Counter MAE**: <2 characters
- **Generation quality**: Human-readable, mostly correct

### Evaluation Metrics

```python
def compute_char_accuracy(preds, targets):
    """Character-level accuracy"""
    correct = (preds == targets).float()
    return correct.mean().item()

def compute_exact_match(pred_texts, target_texts):
    """Sequence-level exact match"""
    matches = sum(p == t for p, t in zip(pred_texts, target_texts))
    return matches / len(pred_texts)

# During eval
model.eval()
char_accs = []
exact_matches = []

for batch in val_loader:
    with torch.no_grad():
        output = model(**batch)
        logits = output["logits"]

        # Character accuracy
        preds = logits.argmax(dim=-1)
        targets = batch["labels"]
        char_acc = compute_char_accuracy(preds, targets)
        char_accs.append(char_acc)

        # Exact match (via generation)
        predictions = model.generate(batch["pixel_values"], bos_id, eos_id, 256)
        pred_texts = [tokenizer.decode(p) for p in predictions]
        target_texts = [tokenizer.decode(batch["labels"][i]) for i in range(len(predictions))]
        exact_match = compute_exact_match(pred_texts, target_texts)
        exact_matches.append(exact_match)

print(f"Character Accuracy: {np.mean(char_accs):.3f}")
print(f"Exact Match: {np.mean(exact_matches):.3f}")
```

---

## 8. Troubleshooting

### Model Not Learning (Loss Stuck)

**Symptoms:** Loss stays high (>3.0) after several epochs

**Possible causes:**
1. **Learning rate too high/low**
   - Try: 5e-5 to 3e-4
   - Check: grad norms (should be 0.1-10)

2. **Gradient clipping too aggressive**
   - Try: clip_norm=2.0 or 5.0
   - Check: are gradients being clipped every step?

3. **Data pipeline issue**
   - Verify: print batch["pixel_values"].shape, batch["labels"].shape
   - Check: are images actually grayscale? Normalized correctly?
   - Test: can you visualize a sample?

4. **Model initialization**
   - Weights initialized with trunc_normal(std=0.02)
   - Layer scales initialized to 1e-6
   - Slot queries initialized with std=0.02

### Counter Loss Not Converging

**Symptoms:** Count loss stays high (>20) after 10 epochs

**Possible causes:**
1. **Counter weight too low**
   - Try: counter_loss_weight=0.2 or 0.5
   - The counter needs stronger signal

2. **Attention mask incorrect**
   - Verify: print attention_mask, check it matches real token count
   - Debug: print actual_counts from attention_mask.sum(dim=1)

3. **Sequence lengths too varied**
   - Try: filter dataset to similar length ranges
   - Or: normalize count_pred and targets to [0,1] range

### Training Unstable (Loss Spikes)

**Symptoms:** Loss jumps up suddenly during training

**Possible causes:**
1. **Learning rate too high**
   - Reduce max_lr to 5e-5
   - Increase warmup_ratio to 0.15

2. **Batch size too small**
   - Try: batch_size=128 or 256
   - Larger batches = more stable gradients

3. **No gradient clipping**
   - Ensure: clip_grad_norm_(model.parameters(), 1.0)
   - Try: clip_value=0.5 if still unstable

4. **DropPath too high**
   - Reduce: drop_path_rate=0.05
   - Or disable: drop_path_rate=0.0

### Poor Generation Quality

**Symptoms:** Predictions are gibberish or repetitive

**Possible causes:**
1. **Underfitting**
   - Train longer (more epochs)
   - Use larger model (increase dims)
   - Reduce regularization (lower dropout, drop_path)

2. **Overfitting**
   - Increase weight_decay
   - Add more dropout
   - Get more training data

3. **Counter prediction wrong**
   - Check: are predicted lengths reasonable?
   - If counter is way off, increase counter_loss_weight

4. **Slot misalignment**
   - The model hasn't learned to align slots with character positions
   - Solution: train longer, counter loss helps with alignment

### Out of Memory (OOM)

**Solutions:**
- Reduce batch_size (try 32 or 16)
- Use gradient accumulation:
  ```python
  accum_steps = 4
  for i, batch in enumerate(dataloader):
      output = model(**batch)
      loss = output["loss"] / accum_steps
      loss.backward()

      if (i + 1) % accum_steps == 0:
          optimizer.step()
          optimizer.zero_grad()
  ```
- Enable gradient checkpointing (trade compute for memory)
- Reduce image size (though 128x128 is already small)

### Slow Training

**Solutions:**
- Use mixed precision (bfloat16)
- Increase num_workers in DataLoader
- Use persistent_workers=True
- Profile with torch.profiler to find bottleneck
- Compile model: `model = torch.compile(model)` (PyTorch 2.0+)

### Mismatch Between CE Loss and Character Accuracy

**Symptoms:** CE loss is low but accuracy is poor (or vice versa)

**Possible causes:**
1. **Label smoothing effect**
   - Label smoothing prevents overconfidence
   - Model may be well-calibrated but slightly lower accuracy
   - This is expected and good for generalization

2. **Padding slots dominant**
   - If most slots are padding, accuracy can be artificially high
   - Check: what's the average sequence length vs 1024 slots?
   - Solution: report accuracy only on non-padding positions

---

## Additional Resources

### Related Files
- [trainer.py](trainer.py) - Training loop implementation
- [torch_dataset.py](torch_dataset.py) - Dataset and collate function
- [exp_config.py](exp_config.py) - Hyperparameter configuration
- [main.py](main.py) - Entry point for training

### Key Differences from Autoregressive Models

| Aspect | Autoregressive | ImageTextSlot (Non-AR) |
|--------|---------------|------------------------|
| Decoding | Sequential (slow) | Parallel (fast) |
| Context | Previous tokens | CNN features only |
| Length | Dynamic (until EOS) | Predicted by counter |
| Training | Teacher forcing | Direct supervision |
| KV Cache | Required | Not needed |
| Inference speed | O(L) passes | O(1) pass |

### Future Improvements

If results are underwhelming, consider:
1. **Add positional encoding** to slot queries
2. **Add self-attention** among slots (before or after cross-attention)
3. **Multi-scale CNN features** (FPN-style) instead of single scale
4. **Iterative refinement** (multiple cross-attention blocks)
5. **Larger model** (more stages, wider channels)
6. **Better pre-training** (e.g., on synthetic data first)

---

## Quick Reference

### Common Commands

```bash
# Train from scratch
uv run python -m gradientlab.experiments.exp20251230_imagetextzip.main

# Resume training
# Edit exp_config.py: resume_from = "path/to/checkpoint"
uv run python -m gradientlab.experiments.exp20251230_imagetextzip.main

# Test model instantiation
uv run python -c "
from gradientlab.experiments.exp20251230_imagetextzip.modeling.factory import ModelFactory
model, tok, cfg = ModelFactory.build_5m()
print(f'Model params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M')
"
```

### Model Parameter Count

```python
from gradientlab.logging_utils.log_model_params import pretty_print_model

total_params = pretty_print_model(model)
print(f"Total trainable parameters: {total_params / 1e6:.2f}M")
```

### Quick Sanity Check

```python
import torch
from gradientlab.experiments.exp20251230_imagetextzip.modeling.factory import ModelFactory

# Build model
model, tokenizer, cfg = ModelFactory.build_5m()
model.eval()

# Dummy input
pixel_values = torch.randn(2, 1, 128, 128)
labels = torch.randint(0, 512, (2, 100))
attention_mask = torch.ones(2, 100)

# Forward
with torch.no_grad():
    output = model(
        pixel_values=pixel_values,
        labels=labels,
        attention_mask=attention_mask,
    )

print("Output keys:", output.keys())
print("Loss:", output["loss"].item())
print("Logits shape:", output["logits"].shape)
print("Count pred:", output["count_pred"].squeeze().tolist())

# Generate
preds = model.generate(pixel_values, bos_token_id=0, eos_token_id=1, max_new_tokens=256)
print("Generated sequence lengths:", [len(p) for p in preds])
```

Expected output:
```
Output keys: dict_keys(['logits', 'count_pred', 'loss', 'loss_ce', 'loss_count'])
Loss: ~5.5 (random initialization)
Logits shape: torch.Size([2, 1024, 512])
Count pred: [~512, ~512] (random, should be ~100 after training)
Generated sequence lengths: [~512, ~512] (random)
```

---

## Summary

ImageTextSlot is a fast, non-autoregressive text recognition model:
1. **Instantiate**: Use `ModelFactory.build_5m()`
2. **Train**: Use provided trainer with mixed precision
3. **Evaluate**: Monitor CE loss, counter MAE, character accuracy
4. **Generate**: Fast parallel decoding via `model.generate()`

The model should achieve >90% character accuracy after 50 epochs on clean text-in-image data.

Good luck training! 🚀

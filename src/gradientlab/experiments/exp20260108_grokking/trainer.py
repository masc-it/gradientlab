"""
Trainer for grokking experiments.

Tracks both loss and accuracy metrics, with auto-stopping when eval accuracy reaches target.
"""

import json
import sys
from pathlib import Path

import torch
from datasets import DatasetDict, load_from_disk
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from gradientlab.neuralblocks.optim.adamw_params import get_adamw_parameters

import trackio

from gradientlab.experiments.exp20260108_grokking.dataset.torch_dataset import DateCollate, DateToISODataset
from gradientlab.experiments.exp20260108_grokking.exp_config import ExpConfig
from gradientlab.experiments.exp20260108_grokking.modeling.config import ModelConfig
from gradientlab.experiments.exp20260108_grokking.modeling.model import DecoderOnlyTransformer
from gradientlab.experiments.exp20260108_grokking.training.lr_scheduler import GrokScheduler


class Trainer:
    """
    Trainer for grokking experiments.

    Key features:
    - Tracks both loss and accuracy
    - Custom GrokScheduler with triggered ramp-up at 99% train accuracy
    - Auto-stopping when eval accuracy >= target threshold
    - Comprehensive checkpointing with scheduler state
    """

    def __init__(
        self,
        model: DecoderOnlyTransformer,
        tokenizer,
        model_cfg: ModelConfig,
        exp_cfg: ExpConfig,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.model_cfg = model_cfg
        self.exp_cfg = exp_cfg

        self.best_eval_accuracy = 0.0
        self.epoch_current = 0
        self.step_current = 0

        self.device = torch.device(exp_cfg.device)
        self.model.to(self.device)

        # Sequential initialization (order matters!)
        self._build_dataloaders()
        self._compile()
        self._setup_optim()
        self._setup_scaler()
        self._restore_training_state()
        self._setup_scheduler()

    def _build_dataloaders(self):
        """Build train and validation dataloaders."""
        self.collate_fn = DateCollate(self.tokenizer)

        # Load dataset from disk
        ds = load_from_disk(self.exp_cfg.ds_name)
        assert isinstance(ds, DatasetDict), "Expected DatasetDict"

        self.ds_train, self.dl_train = self._build_dataloader(ds, "train")
        self.ds_val, self.dl_val = self._build_dataloader(ds, "test")
        self.dl_len = len(self.dl_train)

        print(f"Train: {len(self.ds_train)} examples, {self.dl_len} batches")
        print(f"Val: {len(self.ds_val)} examples, {len(self.dl_val)} batches")

        # Create fixed indices for reproducible autoregressive evaluation
        self._setup_eval_indices()

    def _build_dataloader(self, ds_dict: DatasetDict, split_name: str):
        """Build a single dataloader for a split."""
        torch_ds = DateToISODataset(ds_dict[split_name])

        return torch_ds, DataLoader(
            torch_ds,
            batch_size=self.exp_cfg.batch_size,
            shuffle=(split_name == "train"),
            num_workers=self.exp_cfg.num_workers,
            persistent_workers=True if self.exp_cfg.num_workers > 0 else False,
            pin_memory=False,
            drop_last=True,
            collate_fn=self.collate_fn,
            multiprocessing_context="fork" if sys.platform == "darwin" and self.exp_cfg.num_workers > 0 else None,
        )

    def _setup_eval_indices(self):
        """Setup fixed indices for reproducible autoregressive evaluation."""
        # Use a fixed seed for reproducibility
        rng = torch.Generator().manual_seed(42)

        # Sample 200 fixed indices for training set
        num_train_samples = min(200, len(self.ds_train.ds))
        self.train_eval_indices = torch.randperm(len(self.ds_train.ds), generator=rng)[:num_train_samples].tolist()

        # Sample 200 fixed indices for validation set
        num_val_samples = min(200, len(self.ds_val.ds))
        self.val_eval_indices = torch.randperm(len(self.ds_val.ds), generator=rng)[:num_val_samples].tolist()

        print(f"Fixed eval indices: {num_train_samples} train, {num_val_samples} val")

    def _compile(self):
        """Compile model if supported."""
        if hasattr(torch, "compile"):
            try:
                print("Compiling model...")
                self.model = torch.compile(self.model)
            except Exception as e:
                print(f"Could not compile model: {e}")

    def _setup_optim(self):
        """Setup AdamW optimizer."""
        self.optimizer = AdamW(
            get_adamw_parameters(self.model, weight_decay=self.exp_cfg.weight_decay),
            betas=(0.9, 0.95),
            weight_decay=self.exp_cfg.weight_decay,
            lr=self.exp_cfg.max_lr,
            fused=(self.device.type == "cuda"),  # Use fused AdamW on CUDA
        )

    def _setup_scaler(self):
        """Setup gradient scaler for mixed precision training."""
        self.is_mixed_precision_on = self.device.type == "cuda"
        self.scaler = torch.GradScaler(
            self.device.type, enabled=self.is_mixed_precision_on
        )
        self.dtype = torch.bfloat16 if self.is_mixed_precision_on else None

    def _setup_scheduler(self):
        """Setup custom GrokScheduler."""
        total_steps = self.dl_len * self.exp_cfg.num_epochs
        warmup_steps = int(total_steps * self.exp_cfg.warmup_ratio)

        print(f"{warmup_steps=}")
        self.scheduler = GrokScheduler(
            self.optimizer,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            steps_per_epoch=self.dl_len,
            min_lr=self.exp_cfg.min_lr,
            max_lr=self.exp_cfg.max_lr,
        )

        # Fast-forward scheduler if resuming
        if self.step_current > 0:
            print(f"Fast-forwarding scheduler to step {self.step_current}")
            for _ in range(self.step_current):
                self.scheduler.step()

    def _restore_training_state(self):
        """Restore training state from checkpoint if resuming."""
        if self.exp_cfg.resume_from is None:
            return

        print("=== RESTORING TRAINING STATE ===")
        restore_path = Path(self.exp_cfg.resume_from)

        # Load metadata
        meta_path = restore_path / "meta.json"
        if meta_path.exists():
            metadata = json.loads(meta_path.read_text())
            self.epoch_current = metadata.get("epoch_current", 0)
            self.step_current = metadata.get("step_current", 0)
            self.best_eval_accuracy = metadata.get("best_eval_accuracy", 0.0)
            print(f"Restored: epoch={self.epoch_current}, step={self.step_current}, best_acc={self.best_eval_accuracy:.4f}")

        # Load optimizer
        optim_path = restore_path / "optim.pt"
        if optim_path.exists():
            self.optimizer.load_state_dict(torch.load(optim_path, map_location=self.device, weights_only=True))
            print("Loaded optimizer state")

        # Load scaler
        scaler_path = restore_path / "scaler.pt"
        if scaler_path.exists():
            self.scaler.load_state_dict(torch.load(scaler_path, map_location=self.device, weights_only=True))
            print("Loaded scaler state")

        # Load scheduler (will be loaded after scheduler is created)
        self.scheduler_state_path = restore_path / "scheduler.pt"

    def _save_state(self, epoch_current=None):
        """Save complete training state for checkpointing."""
        exp_dir = self.exp_cfg.exp_dir
        save_dir = exp_dir / "model"
        save_dir.mkdir(exist_ok=True, parents=True)

        # Save model
        torch.save(self.model.state_dict(), save_dir / "model.pt")

        # Save optimizer
        torch.save(self.optimizer.state_dict(), save_dir / "optim.pt")

        # Save scaler
        torch.save(self.scaler.state_dict(), save_dir / "scaler.pt")

        # Save scheduler
        torch.save(self.scheduler.state_dict(), save_dir / "scheduler.pt")

        # Save metadata
        metadata = {
            "step_current": self.step_current,
            "epoch_current": epoch_current if epoch_current is not None else self.epoch_current,
            "best_eval_accuracy": self.best_eval_accuracy,
        }
        (save_dir / "meta.json").write_text(json.dumps(metadata, indent=2))

        print(f"Saved checkpoint to {save_dir}")

    def train_one_epoch(self, epoch_idx: int) :
        """
        Train for one epoch.

        Returns:
            train_loss (float)
        """
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(
            self.dl_train,
            desc=f"Epoch {epoch_idx + 1}/{self.exp_cfg.num_epochs}",
            dynamic_ncols=True,
        )

        for i, batch in enumerate(pbar, start=1):
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
            self.optimizer.zero_grad()

            # Forward pass with mixed precision
            with torch.autocast(
                device_type=self.device.type,
                dtype=self.dtype,
                enabled=self.is_mixed_precision_on,
            ):
                output = self.model(**batch)
                loss = output["loss"]

            # Backward pass
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            norm_pre_clip = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 100.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()  # Step-level scheduler update

            total_loss += loss.item()
            self.step_current += 1

            # Logging at intervals
            if i % self.exp_cfg.log_steps == 0:
                metrics = {
                    "loss": f"{loss.item():.4f}",
                    "norm": f"{norm_pre_clip.item():.2f}",
                    "lr": f"{self.scheduler.get_last_lr()[0]:.6f}",
                }
                pbar.set_postfix(metrics | {"step": f"{i}/{self.dl_len}"})

        # Epoch-level metrics
        train_loss = total_loss / len(self.dl_train)

        return train_loss, norm_pre_clip.item()

    def _show_generation_examples(self, num_examples: int = 3):
        """Show generation examples from validation set (using first N from fixed indices)."""
        print("\n" + "=" * 70)
        print("GENERATION EXAMPLES")
        print("=" * 70)

        # Use first N examples from fixed validation indices for consistency
        raw_ds = self.ds_val.ds
        indices = self.val_eval_indices[:num_examples]
        examples = [raw_ds[i] for i in indices]

        # Create prompts (input without target)
        prompts = [f"<|im_start|>{ex['input']} >" for ex in examples]

        # Tokenize
        encoded = self.tokenizer(
            prompts,
            padding="longest",
            padding_side="left",
            return_tensors="pt",
            add_special_tokens=False,
        )

        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        # Get EOS token ID
        eos_token_id = self.tokenizer("<|im_end|>", add_special_tokens=False)["input_ids"][0]

        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=15,
                eos_token_id=eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode and display
        for i, ex in enumerate(examples):
            gen_ids = generated_ids[i]
            attn_mask = (gen_ids != self.tokenizer.pad_token_id)
            real_tokens = gen_ids[attn_mask]
            generated_text = self.tokenizer.decode(real_tokens.tolist())

            # Extract prediction (after ">")
            if " > " in generated_text:
                parts = generated_text.split(" > ", 1)
                prediction = parts[1] if len(parts) > 1 else ""
            else:
                prediction = generated_text

            # Remove <|im_end|> if present
            if "<|im_end|>" in prediction:
                prediction = prediction.replace("<|im_end|>", "").strip()

            expected = ex['target']
            match = "✓" if prediction == expected else "✗"

            print(f"\n{i+1}. {match} {ex['input']}")
            print(f"   Expected:  {expected}")
            print(f"   Generated: {prediction}")

        print("=" * 70 + "\n")

    def _compute_autoregressive_accuracy(self, num_samples: int = 200, split: str = "val") -> float:
        """
        Compute autoregressive generation accuracy on a sample of data.

        This measures true generation quality by having the model generate
        predictions autoregressively (using its own outputs as context).

        Args:
            num_samples: Number of samples to evaluate (max 200, uses fixed indices)
            split: Which split to use ("val" or "train")

        Returns:
            Exact match accuracy (fraction of perfectly correct generations)
        """
        split_name = "validation" if split == "val" else "training"

        # Use fixed indices for reproducibility
        raw_ds = self.ds_val.ds if split == "val" else self.ds_train.ds
        indices = self.val_eval_indices if split == "val" else self.train_eval_indices

        # Limit to requested num_samples (but max 200)
        num_samples = min(num_samples, len(indices))
        indices = indices[:num_samples]

        print(f"\nComputing autoregressive accuracy on {num_samples} {split_name} samples (fixed indices)...")
        examples = [raw_ds[i] for i in indices]

        # Get EOS token ID
        eos_token_id = self.tokenizer("<|im_end|>", add_special_tokens=False)["input_ids"][0]

        correct = 0

        # Process in small batches to avoid memory issues
        batch_size = 32
        for batch_start in range(0, len(examples), batch_size):
            batch_examples = examples[batch_start:batch_start + batch_size]

            # Create prompts (input without target)
            prompts = [f"<|im_start|>{ex['input']} >" for ex in batch_examples]

            # Tokenize
            encoded = self.tokenizer(
                prompts,
                padding="longest",
                padding_side="left",
                return_tensors="pt",
                add_special_tokens=False,
            )

            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)

            # Generate
            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=15,
                    eos_token_id=eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            # Check predictions
            for i, ex in enumerate(batch_examples):
                gen_ids = generated_ids[i]
                attn_mask = (gen_ids != self.tokenizer.pad_token_id)
                real_tokens = gen_ids[attn_mask]
                generated_text = self.tokenizer.decode(real_tokens.tolist())

                # Extract prediction (after ">")
                if " > " in generated_text:
                    parts = generated_text.split(" > ", 1)
                    prediction = parts[1] if len(parts) > 1 else ""
                else:
                    prediction = generated_text

                # Remove <|im_end|> if present
                if "<|im_end|>" in prediction:
                    prediction = prediction.replace("<|im_end|>", "").strip()

                expected = ex['target']
                if prediction == expected:
                    correct += 1

        accuracy = correct / num_samples
        print(f"Autoregressive accuracy ({split_name}): {correct}/{num_samples} = {accuracy:.4f}")
        return accuracy

    def _compute_eval_loss(self) -> float:
        """
        Compute validation loss efficiently (loss only, no accuracy).

        Returns:
            Average validation loss
        """
        total_loss = 0.0

        print("\nComputing validation loss...")
        with torch.no_grad():
            for batch in tqdm(self.dl_val, desc="Eval loss", dynamic_ncols=True):
                batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

                with torch.autocast(
                    device_type=self.device.type,
                    dtype=self.dtype,
                    enabled=self.is_mixed_precision_on,
                ):
                    output = self.model(**batch)
                    loss = output["loss"]

                total_loss += loss.item()

        eval_loss = total_loss / len(self.dl_val)
        print(f"Validation loss: {eval_loss:.4f}")
        return eval_loss

    def _eval_accuracy(self) -> tuple[float, float, float]:
        """
        Evaluate on validation set and compute train autoregressive accuracy.

        Returns:
            Tuple of (train_ar_acc, eval_ar_acc, eval_loss)
        """
        was_model_training = self.model.training
        self.model.eval()

        # Compute validation loss
        eval_loss = self._compute_eval_loss()

        # Compute train autoregressive accuracy (200 fixed samples)
        train_ar_acc = self._compute_autoregressive_accuracy(num_samples=200, split="train")

        # Compute eval autoregressive generation accuracy (200 fixed samples)
        eval_ar_acc = self._compute_autoregressive_accuracy(num_samples=200, split="val")

        # Show generation examples
        self._show_generation_examples(num_examples=3)

        if was_model_training:
            self.model.train()

        return train_ar_acc, eval_ar_acc, eval_loss

    def train(self):
        """Main training loop with auto-stopping."""
        trackio.init(project=self.exp_cfg.exp_name)

        # Load scheduler state if resuming
        if hasattr(self, "scheduler_state_path") and self.scheduler_state_path.exists():
            scheduler_state = torch.load(self.scheduler_state_path, map_location=self.device, weights_only=True)
            self.scheduler.load_state_dict(scheduler_state)
            print("Loaded scheduler state")

        for epoch in range(self.epoch_current, self.exp_cfg.num_epochs):
            train_loss, grad_norm = self.train_one_epoch(epoch)

            # Check if we should run evaluation this epoch
            is_last_epoch = (epoch == self.exp_cfg.num_epochs - 1)
            should_eval = ((epoch + 1) % self.exp_cfg.eval_every_n_epochs == 0) or is_last_epoch

            if should_eval:
                # Run full evaluation
                train_ar_acc, eval_ar_acc, eval_loss = self._eval_accuracy()

                # CRITICAL: Trigger scheduler ramp-up if train AR accuracy >= 99%
                #self.scheduler.trigger_rampup(train_ar_acc)

                # Log metrics with eval
                trackio.log({
                    "train_ar_accuracy": train_ar_acc,
                    "eval_ar_accuracy": eval_ar_acc,
                    "train_loss": train_loss,
                    "eval_loss": eval_loss,
                    "train_grad_norm": grad_norm,
                    "lr": self.scheduler.get_last_lr()[0],
                    "epoch": epoch,
                })

                print(
                    f"Epoch {epoch + 1}: "
                    f"train_loss={train_loss:.4f}, eval_loss={eval_loss:.4f}, "
                    f"train_ar_acc={train_ar_acc:.4f}, eval_ar_acc={eval_ar_acc:.4f}, "
                    f"lr={self.scheduler.get_last_lr()[0]:.6f}"
                )

                # Save best checkpoint (based on autoregressive accuracy - the real metric)
                if eval_ar_acc > self.best_eval_accuracy:
                    self.best_eval_accuracy = eval_ar_acc
                    self._save_state(epoch_current=epoch + 1)

                # AUTO-STOPPING: Stop when autoregressive accuracy >= target
                if eval_ar_acc >= self.exp_cfg.target_eval_accuracy:
                    print(
                        f"[STOPPING] Autoregressive accuracy {eval_ar_acc:.4f} >= "
                        f"target {self.exp_cfg.target_eval_accuracy}"
                    )
                    break
            else:
                # Log training metrics only (no eval)
                trackio.log({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "lr": self.scheduler.get_last_lr()[0],
                })

                print(
                    f"Epoch {epoch + 1}: "
                    f"train_loss={train_loss:.4f}, "
                    f"lr={self.scheduler.get_last_lr()[0]:.6f} "
                    f"(eval skipped, next in {self.exp_cfg.eval_every_n_epochs - ((epoch + 1) % self.exp_cfg.eval_every_n_epochs)} epoch(s))"
                )

            self.epoch_current += 1

        trackio.finish()
        print(f"Training complete. Best eval accuracy: {self.best_eval_accuracy:.4f}")

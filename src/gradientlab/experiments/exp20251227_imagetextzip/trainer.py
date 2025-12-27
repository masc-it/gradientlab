import json
from pathlib import Path
import sys
from tqdm import tqdm
import trackio

from gradientlab.experiments.exp20251118_0_lm_30m.exp_config import (
    ExpConfig,
)
from gradientlab.experiments.exp20251118_0_lm_30m.modeling.model_cfg import (
    ModelConfig,
)
from datasets import load_from_disk, DatasetDict
from torch.utils.data import DataLoader
import torch
import random
import numpy as np
from torch.optim.adamw import AdamW
from gradientlab.experiments.exp20251227_imagetextzip.modeling.model import (
    SwinImageToText,
)
from gradientlab.experiments.exp20251227_imagetextzip.torch_dataset import (
    Collate,
    MyDataset,
)
from gradientlab.neuralblocks.optim.adamw_params import get_adamw_parameters
from gradientlab.neuralblocks.schedulers.cosine_with_warmup import (
    get_cosine_scheduler_with_warmup,
)
from gradientlab.training_utils.restore import restore_weights


if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


class Trainer:
    def __init__(
        self,
        model: SwinImageToText,
        tokenizer,
        model_cfg: ModelConfig,
        exp_cfg: ExpConfig,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.model_cfg = model_cfg
        self.exp_cfg = exp_cfg
        self.bos_token_id = tokenizer.convert_tokens_to_ids("<|im_start|>")  # type: ignore
        self.eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")  # type: ignore

        self.device = torch.device(exp_cfg.device)
        self.model.to(self.device)  # type: ignore

        self.epoch_current = 0
        self.step_current = 0
        self._build_dataloaders()
        self._compile()
        self._setup_optim()
        self._setup_scaler()

        self._restore_training_state()
        self._setup_scheduler()

    def train(
        self,
    ):
        trackio.init(project=self.exp_cfg.exp_name)
        for epoch in range(self.epoch_current, self.exp_cfg.num_epochs):
            self.train_one_epoch(epoch)
            self.epoch_current += 1

    def train_one_epoch(self, epoch_idx: int):
        self.model.train()
        pbar = tqdm(
            self.dl_train,
            desc=f"Epoch {epoch_idx + 1}/{self.exp_cfg.num_epochs}",
            dynamic_ncols=True,
        )

        for i, batch in enumerate(pbar, start=1):
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
            self.optimizer.zero_grad()
            with torch.autocast(
                device_type=self.device.type,
                dtype=self.dtype,
                enabled=self.is_mixed_precision_on,
            ):
                output = self.model(
                    **batch,
                )
                loss = output["loss"]

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            norm_pre_clip = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            self.step_current += 1

            if i % self.exp_cfg.log_steps == 0:
                metrics = {
                    "loss": f"{loss.detach():.4f}",
                    "norm": f"{norm_pre_clip.item():.2f}",
                    "lr": f"{self.scheduler.get_last_lr()[-1]:.6f}",
                }
                pbar.set_postfix(
                    metrics
                    | {
                        "step": f"{i}/{self.dl_len}",
                    }
                )
                trackio.log(
                    {
                        "epoch": epoch_idx,
                    }
                    | {f"train_{k}": float(v) for k, v in metrics.items()},
                )

            if i % self.exp_cfg.save_steps == 0:
                self._generate()
                self._save_state()

    def _build_dataloaders(self):
        self.collate_fn = Collate(self.tokenizer)
        ds = load_from_disk(self.exp_cfg.ds_name)

        self.ds_train, self.dl_train = self._build_dataloader(ds, "train")
        self.ds_val, self.dl_val = self._build_dataloader(ds, "test")
        self.dl_len = len(self.dl_train)

    def _build_dataloader(self, ds_dict: DatasetDict, split_name: str):
        torch_ds = MyDataset(ds_dict[split_name])

        print(f"len ds {split_name} ={len(torch_ds)}")
        return torch_ds, DataLoader(
            torch_ds,
            batch_size=self.exp_cfg.batch_size,
            shuffle=split_name == "train",
            num_workers=self.exp_cfg.num_workers,
            persistent_workers=True,
            pin_memory=False,
            drop_last=True,
            collate_fn=self.collate_fn,
            multiprocessing_context="fork" if sys.platform in ["darwin"] else None,
        )

    def _setup_optim(self):
        self.optimizer = AdamW(
            get_adamw_parameters(self.model, weight_decay=self.exp_cfg.weight_decay),
            betas=(0.9, 0.95),
            weight_decay=self.exp_cfg.weight_decay,
            lr=self.exp_cfg.max_lr,
            fused=self.device.type == "cuda",
        )

    def _setup_scaler(self):
        self.is_mixed_precision_on = self.device.type == "cuda"
        self.scaler = torch.GradScaler(
            self.device.type, enabled=self.is_mixed_precision_on
        )
        self.dtype = torch.bfloat16 if self.is_mixed_precision_on else None

    def _setup_scheduler(self):
        total_steps = self.dl_len * self.exp_cfg.num_epochs
        tot_warmup_steps = int(total_steps * self.exp_cfg.warmup_ratio)

        self.scheduler = get_cosine_scheduler_with_warmup(
            self.optimizer,
            total_steps,
            tot_warmup_steps,
            self.exp_cfg.min_lr,
        )

        # Fast-forward to current position
        for _ in range(self.dl_len * self.epoch_current):
            self.scheduler.step()

    def _compile(self):
        if self.device.type in ["cuda", "mps"]:
            self.model.compile()

    def _save_state(self):
        exp_dir = self.exp_cfg.exp_dir
        hf_save_dir = exp_dir / "model"
        hf_save_dir.mkdir(exist_ok=True, parents=True)

        # self.model.save_pretrained(hf_save_dir)
        # self.tokenizer.save_pretrained(hf_save_dir)
        """ hf_add_custom_model_metadata(
            hf_save_dir,
            self.exp_cfg.exp_name,
            self.model.__class__,
            self.model_cfg.__class__,
        ) """

        torch.save(self.optimizer.state_dict(), hf_save_dir / "optim.pt")
        torch.save(self.model.state_dict(), hf_save_dir / "model.pt")
        torch.save(self.scaler.state_dict(), hf_save_dir / "scaler.pt")
        torch.save(self.scheduler.state_dict(), hf_save_dir / "scheduler.pt")

        (hf_save_dir / "meta.json").write_text(
            json.dumps(
                {
                    "step_current": self.step_current,
                    "epoch_current": self.epoch_current,
                },
                indent=2,
            )
        )

    def _generate(self):
        was_model_training = self.model.training
        self.model.eval()

        i = random.randint(0, len(self.ds_val))
        sample = self.ds_val[i]
        lbl = sample["text"]

        inputs = self.collate_fn(
            [{"pixel_values": sample["pixel_values"], "text": "<|im_start|>"}]
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        preds = self.model.generate(
            inputs["pixel_values"],
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            max_new_tokens=256,
        )
        print(f"GT: '{lbl}' - PRED: '{self.tokenizer.decode(preds[0])}'")

        if was_model_training:
            self.model.train()

    def _restore_training_state(self):
        if self.exp_cfg.resume_from is None:
            return

        print("=== RESTORING TRAINING STATE ===")
        restore_path = Path(self.exp_cfg.resume_from)
        restore_weights(restore_path, optim=self.optimizer, scaler=self.scaler)

        metadata = json.loads((restore_path / "meta.json").read_bytes())
        self.epoch_current = metadata["epoch_current"]
        self.step_current = metadata["step_current"]

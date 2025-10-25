import json
import sys

from tqdm import tqdm
import trackio
from transformers import Qwen2TokenizerFast
from gradientlab.data_utils.torch_datasets.tokenized_text_ds import (
    PreTokenizedTextDataset,
)
from gradientlab.experiments.exp20251016_0_lm_20m_polyrelu_lm_vanilla_fineweb_ita.exp_config import (
    ExpConfig,
)
from gradientlab.experiments.exp20251016_0_lm_20m_polyrelu_lm_vanilla_fineweb_ita.modeling.model import (
    GPTForCausalLM,
)
from gradientlab.experiments.exp20251016_0_lm_20m_polyrelu_lm_vanilla_fineweb_ita.modeling.model_cfg import (
    ModelConfig,
)
from datasets import load_from_disk
from torch.utils.data import DataLoader
import torch
import random
import numpy as np
from torch.optim.adamw import AdamW
from gradientlab.neuralblocks.optim.adamw_params import get_adamw_parameters
from gradientlab.neuralblocks.schedulers.cosine_with_warmup import (
    get_cosine_scheduler_with_warmup,
)
from gradientlab.training_utils.hf_save import hf_add_custom_model_metadata

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
        model: GPTForCausalLM,
        tokenizer: Qwen2TokenizerFast,
        model_cfg: ModelConfig,
        exp_cfg: ExpConfig,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.model_cfg = model_cfg
        self.exp_cfg = exp_cfg

        self.device = torch.device(exp_cfg.device)
        self.model.to(self.device)

        self.epoch_current = 0
        self.step_current = 0
        # self._resume_state() TODO
        self._build_dataloaders()
        self._compile()
        self._setup_optim()
        self._setup_scaler()
        self._setup_scheduler()

    def train(
        self,
    ):
        trackio.init(project=self.exp_cfg.exp_name)
        for epoch in range(self.exp_cfg.num_epochs):
            self.train_one_epoch(epoch)
            self.epoch_current += 1

    def train_one_epoch(self, epoch_idx: int):
        self.model.train()
        pbar = tqdm(
            self.dl_train,
            desc=f"Epoch {epoch_idx + 1}/{self.exp_cfg.num_epochs}",
            dynamic_ncols=True,
        )

        for i, input_ids in enumerate(pbar, start=1):
            input_ids = input_ids.to(self.device, non_blocking=True)
            self.optimizer.zero_grad()
            with torch.autocast(
                device_type=self.device.type,
                dtype=self.dtype,
                enabled=self.is_mixed_precision_on,
            ):
                output = self.model(
                    input_ids, labels=input_ids.clone(), use_cache=False
                )
                loss = output.loss

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            norm_pre_clip = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
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
        self.dl_train = self._build_dataloader("train")
        self.dl_len = len(self.dl_train)
        # TODO val

    def _build_dataloader(self, split_name: str):
        torch_ds = self._load_dataset(split_name)
        return DataLoader(
            torch_ds,
            batch_size=self.exp_cfg.batch_size,
            shuffle=split_name == "train",
            num_workers=self.exp_cfg.num_workers,
            persistent_workers=True,
            pin_memory=False,
            drop_last=True,
            multiprocessing_context="fork" if sys.platform in ["darwin"] else None,
        )

    def _load_dataset(self, split: str):
        ds = load_from_disk(self.exp_cfg.ds_name)
        dataset = PreTokenizedTextDataset(ds)
        print(f"len ds {split} ={len(dataset)}")
        return dataset

    def _setup_optim(self):
        self.optimizer = AdamW(
            get_adamw_parameters(self.model, weight_decay=self.exp_cfg.weight_decay),
            betas=(0.9, 0.999),
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
        total_steps = self.dl_len * (self.exp_cfg.num_epochs - self.epoch_current)
        warmup_steps = int(total_steps * self.exp_cfg.warmup_ratio)

        self.scheduler = get_cosine_scheduler_with_warmup(
            self.optimizer,
            total_steps,
            warmup_steps,
            self.exp_cfg.min_lr,
        )

    def _compile(self):
        if self.device.type in ["cuda", "mps"]:
            self.model.compile()

    def _save_state(self):
        exp_dir = self.exp_cfg.exp_dir
        hf_save_dir = exp_dir / "model"
        self.model.save_pretrained(hf_save_dir)
        self.tokenizer.save_pretrained(hf_save_dir)
        hf_add_custom_model_metadata(hf_save_dir, self.exp_cfg.exp_name, self.model, self.model_cfg)

        torch.save(self.optimizer.state_dict(), exp_dir / "optim.pt")
        torch.save(self.scaler.state_dict(), self.exp_cfg.exp_dir / "scaler.pt")
        torch.save(self.scheduler.state_dict(), self.exp_cfg.exp_dir / "scheduler.pt")

        (exp_dir / "meta.json").write_text(
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

        inputs = self.tokenizer(["<|im_start|>Ciao sono Mauro e "], return_tensors="pt", add_special_tokens=False)
        inputs = {k: v.to(self.device) for k,v in inputs.items()}
        self.model.eval()
        preds = self.model.generate(
            **inputs,
            max_length=40,
            do_sample=False,
        )
        print(self.tokenizer.decode(preds[0]))

        if was_model_training:
            self.model.train()

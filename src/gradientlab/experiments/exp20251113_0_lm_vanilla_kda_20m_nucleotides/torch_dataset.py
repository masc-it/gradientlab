from typing import Any
import torch
from torch.utils.data import Dataset as TDataset
from datasets import Dataset


class NucleotidesDataset(TDataset):
    
    def __init__(self, ds: Dataset) -> None:
        super().__init__()

        self.ds = ds
    
    def __getitem__(self, index):
        
        sample = self.ds[index]["sequence"]
        return f"<|im_start|>{sample}<|im_end|>"
    
    def __len__(self):
        return len(self.ds)
    
class NucleotidesTaskDataset(TDataset):
    
    def __init__(self, ds: Dataset) -> None:
        super().__init__()

        self.ds = ds
    
    def __getitem__(self, index):
        
        sample = self.ds[index]
        seq = sample["sequence"]
        label = sample["label"]
        task = sample["task"]

        return f"<|im_start|>{seq} | {task} > {label}<|im_end|>"
    
    def __len__(self):
        return len(self.ds)
    
class NucleotidesTaskDatasetInference(TDataset):
    
    def __init__(self, ds: Dataset) -> None:
        super().__init__()

        self.ds = ds
    
    def __getitem__(self, index):
        
        sample = self.ds[index]
        seq = sample["sequence"]
        label = sample["label"]
        task = sample["task"]

        return f"<|im_start|>{seq} | {task} >", label
    
    def __len__(self):
        return len(self.ds)


class NucleotidesCollate:

    def __init__(self, tok,) -> None:
        self.tok = tok
    
    def __call__(self, labels: list) -> Any:
        encoded = self.tok(labels, padding="longest", return_tensors="pt", add_special_tokens=False, return_attention_mask=False)
        return {**encoded}

class NucleotidesTaskCollate:

    def __init__(self, tok,) -> None:
        self.tok = tok
        self.pad_token_id = tok.pad_token_id
        self.sep_token_id = tok(">", add_special_tokens=False)["input_ids"][0]
    
    def __call__(self, labels: list) -> Any:
        encoded = self.tok(labels, padding="longest", return_tensors="pt", add_special_tokens=False)
        labels = self._prepare_labels(encoded["input_ids"]) # type: ignore
        return {**encoded, "labels": labels}
    
    def _prepare_labels(self, input_ids: torch.Tensor):
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be 2D (batch_size, seq_len)")

        B, L = input_ids.size()
        labels = input_ids.clone()

        # (B, L) -> True where separator is
        sep_mask = (input_ids == self.sep_token_id)

        # Which sequences actually contain sep?
        has_sep = sep_mask.any(dim=1)

        if not has_sep.any():
            # No sequence has a separator; nothing to mask
            return labels

        # First separator index per sequence (0 if none; we'll fix that)
        # Convert to float so argmax works; positions with no True become 0
        first_sep_idx = sep_mask.float().argmax(dim=1)  # (B,)

        # For sequences without sep, set index to L (past the end)
        # so that no position satisfies pos <= first_sep_idx
        first_sep_idx = torch.where(
            has_sep,
            first_sep_idx,
            torch.full_like(first_sep_idx, L)
        )  # (B,)

        # Build a position grid: shape (1, L) = [0, 1, 2, ..., L-1]
        positions = torch.arange(L, device=input_ids.device).unsqueeze(0)  # (1, L)

        # Broadcast to (B, L): True where position <= first_sep_idx for each sequence
        prefix_mask = positions <= first_sep_idx.unsqueeze(1)  # (B, L) bool

        # Only apply mask on sequences that actually have sep
        # (for those without, first_sep_idx == L so prefix_mask is all False anyway,
        # but this keeps intent clear)
        prefix_mask = prefix_mask & has_sep.unsqueeze(1)

        # Apply pad_token_id to prefix positions
        labels[prefix_mask] = self.pad_token_id

        return labels

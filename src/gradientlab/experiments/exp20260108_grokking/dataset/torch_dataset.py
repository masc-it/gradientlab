"""
PyTorch dataset wrapper and collate function for Date to ISO dataset.
"""

import torch
from datasets import Dataset


class DateToISODataset(torch.utils.data.Dataset):
    """
    PyTorch dataset wrapper for Date to ISO conversion.
    """

    def __init__(self, ds: Dataset):
        self.ds = ds

    def __getitem__(self, index: int) -> str:
        sample = self.ds[index]
        # Format: <|im_start|>input > target<|im_end|>
        # The ">" token marks the boundary between prompt and target
        return f"<|im_start|>{sample['input']} > {sample['target']}<|im_end|>"

    def __len__(self) -> int:
        return len(self.ds)


class DateCollate:
    """
    Collate function for Date to ISO dataset.
    Performs tokenization with left-padding and creates labels.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        # Get the token ID for ">" separator
        self.sep_token_id = tokenizer(">", add_special_tokens=False)["input_ids"][0]

    def __call__(self, batch: list[str]) -> dict[str, torch.Tensor]:
        """
        Tokenize batch with left padding and create labels.

        Args:
            batch: List of formatted strings

        Returns:
            Dictionary with input_ids, attention_mask, and labels
        """
        # Tokenize with LEFT padding (standard for decoder-only models)
        encoded = self.tokenizer(
            batch,
            padding="longest",
            padding_side="left",  # LEFT padding for decoder-only
            return_tensors="pt",
            add_special_tokens=False,
        )

        # Create labels: mask everything before ">" separator
        labels = self._create_labels(encoded["input_ids"])

        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": labels,
        }

    def _create_labels(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Create labels by masking the prompt (everything before ">").

        Only the target portion (after ">") should be predicted.

        Args:
            input_ids: Tokenized input (batch_size, seq_len)

        Returns:
            Labels tensor with prompt masked as pad_token_id
        """
        labels = input_ids.clone()

        # Find positions where separator token appears (B, L)
        sep_mask = (input_ids == self.sep_token_id)

        # Find first occurrence index for each batch item (B,)
        # argmax returns 0 if all False, so we need to handle that case
        sep_indices = torch.argmax(sep_mask.int(), dim=1)

        # Check which rows actually have the separator (B,)
        has_sep = sep_mask.any(dim=1)

        # Create position indices for masking (L,)
        positions = torch.arange(input_ids.size(1), device=input_ids.device)

        # Mask positions <= sep_index for each batch item
        # Broadcasting: (B, 1) <= (L,) -> (B, L)
        mask = positions[None, :] <= sep_indices[:, None]

        # Only apply mask where separator exists
        mask = mask & has_sep[:, None]

        # Apply masking - these positions will be ignored in loss computation
        labels[mask] = self.pad_token_id

        return labels

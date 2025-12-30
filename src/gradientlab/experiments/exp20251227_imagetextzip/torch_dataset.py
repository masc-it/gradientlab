import torch
from datasets import Dataset
from torchvision.transforms import v2


class Collate:
    def __init__(
        self,
        tok,
    ) -> None:
        self.tokenizer = tok
        self.pad_token_id = tok.pad_token_id

    def __call__(self, batch: list):
        images = torch.stack([el["pixel_values"] for el in batch])
        encoded = self.tokenizer(
            [el["text"] for el in batch],
            add_special_tokens=False,
            padding="longest",
            padding_side="right",
            return_tensors="pt",
            pad_to_multiple_of=8,
        )

        return {
            "pixel_values": images,
            **encoded,
            "labels": encoded["input_ids"].clone(),
        }


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, ds: Dataset) -> None:
        super().__init__()

        self.ds = ds
        self.transforms = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize((0.1533,), (0.3034,)),
            ]
        )

    def __getitem__(self, index):
        sample = self.ds[index]
        return {
            "pixel_values": self.transforms(sample["pixel_values"]),
            "text": f"<|im_start|>{sample['text']}<|im_end|>",
        }

    def __len__(self):
        return len(self.ds)

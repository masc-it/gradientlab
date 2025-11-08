from typing import Any
import torch
from torch.utils.data import Dataset as TDataset
from datasets import Dataset
from torchvision import transforms



class NucleotidesDataset(TDataset):
    
    def __init__(self, ds: Dataset) -> None:
        super().__init__()

        self.ds = ds

        self.t = transforms.Compose([
            transforms.ToTensor()
        ])
    
    def __getitem__(self, index):
        
        sample = self.ds[index]["sequence"]
        return f"<|im_start|>{sample}<|im_end|>"
    
    def __len__(self):
        return len(self.ds)



class VLMCollate:

    def __init__(self, tok,) -> None:
        self.tok = tok
    
    def __call__(self, labels: list) -> Any:
        encoded = self.tok(labels, padding="longest", return_tensors="pt", add_special_tokens=False)
        return {**encoded}
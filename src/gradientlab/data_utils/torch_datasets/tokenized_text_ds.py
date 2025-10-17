import torch
from torch.utils.data import Dataset


class PreTokenizedTextDataset(Dataset):
    def __init__(self, dataset) -> None:
        self.dataset = dataset

    def __getitem__(self, idx):
        return torch.tensor(self.dataset[idx]["input_ids"], dtype=torch.long)

    def __len__(self):
        return len(self.dataset)

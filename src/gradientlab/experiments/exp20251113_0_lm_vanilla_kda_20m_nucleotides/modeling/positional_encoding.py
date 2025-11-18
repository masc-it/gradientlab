import math
import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """Default positional encoding."""

    def __init__(self, d_model: int, max_len: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(pos * div)
        pe[0, :, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor, start_pos: int) -> torch.Tensor:
        return self.dropout(x + self.pe[:, start_pos : start_pos + x.size(1)])  # type: ignore

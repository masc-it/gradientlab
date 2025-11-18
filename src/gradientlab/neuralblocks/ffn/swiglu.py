import torch
from torch import nn
import torch.nn.functional as F

class SwiGLUFeedForward(nn.Module):
    def __init__(
        self, d_model: int, mult: float = 4, dropout: float = 0.0, use_bias: bool = False
    ):
        super().__init__()
        hidden = int(d_model * mult * 2)  # ×2 for the gate
        self.w12 = nn.Linear(d_model, hidden, bias=use_bias)
        self.w3 = nn.Linear(hidden // 2, d_model, bias=use_bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.w12(x).chunk(2, dim=-1)  # Project and split in two halves
        x = x1 * F.silu(x2)  # SwiGLU gate
        return self.dropout(self.w3(x))
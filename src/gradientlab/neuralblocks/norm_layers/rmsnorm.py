import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, hidden_dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.w = nn.Parameter(torch.ones(hidden_dim))
        self.eps = torch.tensor(eps)

    def forward(
        self,
        x: torch.Tensor,
    ):
        x = (
            x
            * torch.rsqrt(
                torch.mean(x * x, dim=-1, keepdim=True, dtype=x.dtype) + self.eps
            )
            * self.w
        ).to(x.dtype)
        return x

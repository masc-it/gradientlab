import torch
from torch import nn


class PolyNormFeedForward(nn.Module):
    def __init__(
        self,
        d_model: int,
        mult: float = 4,
        dropout: float = 0.0,
        use_bias: bool = False,
    ):
        super().__init__()
        hidden = int(d_model * mult)
        self.in_proj = nn.Linear(d_model, hidden, bias=use_bias)
        self.act = PolyNorm()
        self.out_proj = nn.Linear(hidden, d_model, bias=use_bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(x)
        x = self.act(x)
        return self.dropout(self.out_proj(x))


class PolyNorm(torch.nn.Module):
    def __init__(self, eps=1e-6):
        super(PolyNorm, self).__init__()
        self.weight = torch.nn.Parameter(torch.ones(3) / 3)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        self.eps = eps

    def _norm(self, x: torch.Tensor):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return (
            self.weight[0] * self._norm(x**3)
            + self.weight[1] * self._norm(x**2)
            + self.weight[2] * self._norm(x)
            + self.bias
        )

import torch
from torch import nn

from gradientlab.neuralblocks.norm_layers.rmsnorm import RMSNorm


class ImagePatchifier(nn.Module):
    def __init__(self, hidden_dim: int, patch_size: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            3,
            hidden_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )
        self.norm = RMSNorm(hidden_dim)

    def forward(self, images: torch.Tensor):
        images = self.proj(images)  # [B, D, H, W]
        patches = images.flatten(2)  # [B, D, L_IMG]
        patches = patches.transpose(1, 2)  # [B, L_IMG, D]

        patches = self.norm(patches)
        return patches

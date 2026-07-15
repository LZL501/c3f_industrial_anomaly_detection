from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def nonlinearity(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def normalize(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6, affine=True)


def dissimilarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return 1 - F.cosine_similarity(a, b, dim=1).unsqueeze(1)


class PositionEmbeddingSine(nn.Module):
    def __init__(
        self,
        num_pos_feats: int,
        temperature: int = 10000,
        normalize_pos: bool = False,
        scale: float | None = None,
    ):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize_pos = normalize_pos
        self.scale = scale or 2 * math.pi

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        if self.num_pos_feats == 0:
            return x.new_empty((b, 0, h, w))
        y_embed = (
            torch.arange(1, h + 1, dtype=torch.float32, device=x.device)
            .view(1, h, 1)
            .repeat(b, 1, w)
        )
        x_embed = (
            torch.arange(1, w + 1, dtype=torch.float32, device=x.device)
            .view(1, 1, w)
            .repeat(b, h, 1)
        )
        if self.normalize_pos:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        return torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)

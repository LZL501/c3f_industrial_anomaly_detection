from __future__ import annotations

import torch
from einops import rearrange
from torch import nn


class VectorQuantizer(nn.Module):
    def __init__(
        self, n_embed: int, embed_dim: int, beta: float = 0.25, legacy: bool = True
    ) -> None:
        super().__init__()
        self.n_embed = n_embed
        self.embed_dim = embed_dim
        self.beta = beta
        self.legacy = legacy
        self.embedding = nn.Embedding(n_embed, embed_dim)

    def forward(self, z: torch.Tensor):
        z_bhwc = rearrange(z, "b c h w -> b h w c").contiguous()
        flat = z_bhwc.flatten(0, 2)
        indices = self._nearest_indices(flat, self.embedding.weight)
        quant = self.embedding(indices).view(z_bhwc.shape)
        loss = self._loss(z_bhwc, quant)
        quant = z_bhwc + (quant - z_bhwc).detach()
        quant = rearrange(quant, "b h w c -> b c h w").contiguous()
        return quant, loss, indices

    def get_code_scale(self, z: torch.Tensor, scale: int):
        z_bhwc = rearrange(z, "b c h w -> b h w c").contiguous()
        flat = z_bhwc.flatten(0, 2)
        codebook = rearrange(self.embedding.weight, "n (s c) -> (n s) c", s=scale)
        indices = self._nearest_indices(flat, codebook)
        quant = codebook[indices].view(z_bhwc.shape)
        loss = self._loss(z_bhwc, quant)
        quant = z_bhwc + (quant - z_bhwc).detach()
        quant = rearrange(quant, "b h w c -> b c h w").contiguous()
        return quant, loss, indices

    def set_codebook(self, features: torch.Tensor, freeze: bool = True) -> None:
        if features.shape != self.embedding.weight.shape:
            raise ValueError(
                f"Expected codebook {tuple(self.embedding.weight.shape)}, got {tuple(features.shape)}"
            )
        self.embedding = nn.Embedding.from_pretrained(
            features.detach().clone(), freeze=freeze
        )

    def _nearest_indices(
        self, flat: torch.Tensor, codebook: torch.Tensor
    ) -> torch.Tensor:
        distances = (
            torch.sum(flat**2, dim=1, keepdim=True)
            + torch.sum(codebook**2, dim=1)
            - 2 * torch.einsum("bd,dn->bn", flat, rearrange(codebook, "n d -> d n"))
        )
        return torch.argmin(distances, dim=1)

    def _loss(self, z: torch.Tensor, quant: torch.Tensor) -> torch.Tensor:
        if self.legacy:
            return torch.mean((quant.detach() - z) ** 2) + self.beta * torch.mean(
                (quant - z.detach()) ** 2
            )
        return self.beta * torch.mean((quant.detach() - z) ** 2) + torch.mean(
            (quant - z.detach()) ** 2
        )

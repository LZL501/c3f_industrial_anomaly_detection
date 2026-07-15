from __future__ import annotations

import torch
from torch import nn

from .blocks import normalize, nonlinearity


class Upsample(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        with_conv: bool = True,
        scale: float = 2.0,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.conv = (
            nn.Conv2d(
                in_channels,
                out_channels or in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            )
            if with_conv
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.interpolate(x, scale_factor=self.scale, mode="nearest")
        return self.conv(x) if self.conv is not None else x


class Bottleneck(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        mid_channels: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        mid_channels = out_channels // 2 if mid_channels is None else mid_channels
        self.norm1 = normalize(in_channels)
        self.conv1 = nn.Conv2d(
            in_channels, mid_channels, kernel_size=1, stride=1, padding=0
        )
        self.norm2 = normalize(mid_channels)
        self.conv2 = nn.Conv2d(
            mid_channels, mid_channels, kernel_size=3, stride=1, padding=1
        )
        self.norm3 = normalize(mid_channels)
        self.conv3 = nn.Conv2d(
            mid_channels, out_channels, kernel_size=1, stride=1, padding=0
        )
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
            if in_channels != out_channels
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.norm1(x))
        h = self.conv2(self.norm2(h))
        h = nonlinearity(self.conv3(self.norm3(h)))
        return (self.shortcut(x) if self.shortcut is not None else x) + h


class Decoder(nn.Module):
    def __init__(
        self,
        ch: int,
        out_ch: int,
        ch_mult: list[int],
        down_samples: list[int],
        num_res_blocks: list[int],
        attn_resolutions: list[int] | None = None,
        dropout: float = 0.0,
        resamp_with_conv: bool = True,
        in_channels: int = 3,
        resolution: int = 256,
        z_channels: int = 2048,
        **_: object,
    ) -> None:
        super().__init__()
        if attn_resolutions:
            raise NotImplementedError(
                "Clean C3F decoder currently expects no attention resolutions."
            )
        self.num_resolutions = len(ch_mult)
        block_in = ch * ch_mult[-1]
        self.up = nn.ModuleList()
        for level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            block_out = ch * ch_mult[level]
            for _ in range(num_res_blocks[level]):
                block.append(
                    Bottleneck(
                        block_in,
                        out_channels=block_out,
                        mid_channels=block_in // 2,
                        dropout=dropout,
                    )
                )
                block_in = block_out
            up = nn.Module()
            up.block = block
            if level != 0:
                out_channels = block_in if level == 1 else block_in // 2
                up.upsample = Upsample(
                    block_in, out_channels, resamp_with_conv, scale=down_samples[level]
                )
            self.up.insert(0, up)
        self.norm_out = normalize(block_in)
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(
        self, zs: list[torch.Tensor]
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        h = zs[-1]
        features = []
        for level in reversed(range(self.num_resolutions)):
            for block in self.up[level].block:
                h = block(h)
            if level != 0:
                features.append(h)
                h = self.up[level].upsample(h)
            if level > 1:
                h = torch.cat((h, zs[level - 2]), dim=1)
        h = self.conv_out(nonlinearity(self.norm_out(h)))
        return features[::-1], h

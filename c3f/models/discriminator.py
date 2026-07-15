from __future__ import annotations

import torch
from torch import nn


class SegmentationHead(nn.Module):
    def __init__(
        self, in_channels: int = 6, out_channels: int = 2, base_channels: int = 64
    ) -> None:
        super().__init__()
        self.encoder = EncoderDiscriminative(in_channels, base_channels)
        self.decoder = DecoderDiscriminative(base_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1, b2, b3, b4 = self.encoder(x)
        return self.decoder(b1, b2, b3, b4)


class EncoderDiscriminative(nn.Module):
    def __init__(self, in_channels: int, base_width: int) -> None:
        super().__init__()
        self.block1 = _block(in_channels, base_width)
        self.mp1 = nn.MaxPool2d(4)
        self.block2 = _block(base_width, base_width * 2)
        self.mp2 = nn.MaxPool2d(2)
        self.block3 = _block(base_width * 2, base_width * 4)
        self.mp3 = nn.MaxPool2d(2)
        self.block4 = _block(base_width * 4, base_width * 8)
        self.mp4 = nn.MaxPool2d(2)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        b1 = self.block1(x)
        mp1 = self.mp1(b1)
        b2 = self.block2(mp1)
        mp2 = self.mp2(b2)
        b3 = self.block3(mp2)
        mp3 = self.mp3(b3)
        b4 = self.block4(mp3)
        mp4 = self.mp4(b4)
        return mp1, mp2, mp3, mp4


class DecoderDiscriminative(nn.Module):
    def __init__(self, base_width: int, out_channels: int = 2) -> None:
        super().__init__()
        self.up2 = _up(base_width * 8, base_width * 4)
        self.db2 = _block(base_width * 8, base_width * 4, biases=(True, True))
        self.up3 = _up(base_width * 4, base_width * 2)
        self.db3 = _block(base_width * 4, base_width * 2)
        self.up4 = _up(base_width * 2, base_width)
        self.db4 = _block(base_width * 2, base_width, biases=(False, True))
        self.out = nn.Sequential(
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, out_channels, kernel_size=3, padding=1, bias=False),
        )

    def forward(
        self, b1: torch.Tensor, b2: torch.Tensor, b3: torch.Tensor, b4: torch.Tensor
    ) -> torch.Tensor:
        x = self.db2(torch.cat((self.up2(b4), b3), dim=1))
        x = self.db3(torch.cat((self.up3(x), b2), dim=1))
        x = self.db4(torch.cat((self.up4(x), b1), dim=1))
        return self.out(x)


class PatchDiscriminator(nn.Module):
    def __init__(
        self, in_channels: int = 3, base_channels: int = 64, num_layers: int = 3
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        nf = base_channels
        for _ in range(1, num_layers):
            next_nf = min(nf * 2, 512)
            layers += [
                nn.Conv2d(nf, next_nf, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(next_nf),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            nf = next_nf
        next_nf = min(nf * 2, 512)
        layers += [
            nn.Conv2d(nf, next_nf, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(next_nf),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        nf = next_nf
        layers.append(nn.Conv2d(nf, 1, kernel_size=4, stride=1, padding=1))
        self.model = nn.Sequential(*layers)
        self.apply(weights_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def weights_init(module: nn.Module) -> None:
    if isinstance(module, nn.Conv2d):
        nn.init.normal_(module.weight, 0.0, 0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight, 1.0, 0.02)
        nn.init.zeros_(module.bias)


def _block(
    in_channels: int,
    out_channels: int,
    biases: tuple[bool, bool] = (False, False),
) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=biases[0]),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=biases[1]),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def _up(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )

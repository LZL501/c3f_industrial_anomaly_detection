from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torchvision import models


class LPIPS(nn.Module):
    """LPIPS variant used by the released ``model_for_visa2.py`` loss."""

    def __init__(
        self, weights_path: str | Path | None = None, use_dropout: bool = True
    ) -> None:
        super().__init__()
        channels = (64, 128, 256, 512, 512)
        self.features = VGG16Features()
        self.linear = nn.ModuleList(
            [NetLinLayer(channel, use_dropout) for channel in channels]
        )
        if weights_path is None:
            weights_path = (
                Path(__file__).resolve().parents[2] / "assets" / "lpips" / "vgg.pth"
            )
        self._load_linear_weights(Path(weights_path))
        self.requires_grad_(False)

    def forward(
        self, input_image: torch.Tensor, target_image: torch.Tensor
    ) -> torch.Tensor:
        input_features = self.features(input_image)
        target_features = self.features(target_image)
        values = []
        for layer, input_feature, target_feature in zip(
            self.linear, input_features, target_features
        ):
            difference = (_normalize(input_feature) - _normalize(target_feature)) ** 2
            values.append(layer(difference).mean(dim=(2, 3), keepdim=True))
        return torch.stack(values, dim=0).sum(dim=0)

    def _load_linear_weights(self, path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"LPIPS weights are missing: {path}")
        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:  # PyTorch < 2.0
            state = torch.load(path, map_location="cpu")
        remapped = {}
        for index in range(5):
            old_key = f"lin{index}.model.1.weight"
            new_key = f"linear.{index}.model.1.weight"
            if old_key in state:
                remapped[new_key] = state[old_key]
        missing, unexpected = self.load_state_dict(remapped, strict=False)
        missing_linear = [key for key in missing if key.startswith("linear.")]
        if missing_linear or unexpected:
            raise RuntimeError(
                f"Invalid LPIPS weights in {path}: missing={missing_linear}, unexpected={unexpected}"
            )


class VGG16Features(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        try:
            features = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        except (AttributeError, TypeError):
            features = models.vgg16(pretrained=True).features
        boundaries = ((0, 4), (4, 9), (9, 16), (16, 23), (23, 30))
        self.slices = nn.ModuleList(
            [nn.Sequential(*features[start:end]) for start, end in boundaries]
        )
        self.requires_grad_(False)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
        outputs = []
        for layer in self.slices:
            image = layer(image)
            outputs.append(image)
        return tuple(outputs)


class NetLinLayer(nn.Module):
    def __init__(self, in_channels: int, use_dropout: bool) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Dropout()] if use_dropout else []
        layers.append(nn.Conv2d(in_channels, 1, kernel_size=1, bias=False))
        self.model = nn.Sequential(*layers)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.model(feature)


def _normalize(feature: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    norm = torch.sqrt(torch.sum(feature**2, dim=1, keepdim=True))
    return feature / (norm + eps)

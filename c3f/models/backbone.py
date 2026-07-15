from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
import torchvision


class ResNetBackbone(nn.Module):
    def __init__(
        self,
        name: str = "wide_resnet50_2",
        pretrained: bool = True,
        weights_name: str = "IMAGENET1K_V1",
        train_backbone: bool = False,
        return_interm_layers: bool = True,
        dilation: bool = False,
        freeze_running_stats: bool = True,
    ) -> None:
        super().__init__()
        factory = getattr(torchvision.models, name)
        try:
            weights = _resolve_weights(name, weights_name) if pretrained else None
            backbone = factory(
                weights=weights, replace_stride_with_dilation=[False, False, dilation]
            )
        except TypeError:
            backbone = factory(
                pretrained=pretrained,
                replace_stride_with_dilation=[False, False, dilation],
            )
        for param_name, parameter in backbone.named_parameters():
            train_layer = (
                "layer2" in param_name
                or "layer3" in param_name
                or "layer4" in param_name
            )
            if not train_backbone or not train_layer:
                parameter.requires_grad_(False)
        return_layers = (
            {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
            if return_interm_layers
            else {"layer3": "0"}
        )
        self.train_backbone = train_backbone
        self.freeze_running_stats = freeze_running_stats
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        if not self.train_backbone and self.freeze_running_stats:
            self.body.eval()

    def train(self, mode: bool = True) -> ResNetBackbone:
        super().train(mode)
        if not self.train_backbone and self.freeze_running_stats:
            self.body.eval()
        return self

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        outputs: OrderedDict[str, torch.Tensor] = self.body(x)
        return list(outputs.values())


def _resolve_weights(model_name: str, weights_name: str):
    get_model_weights = getattr(torchvision.models, "get_model_weights", None)
    if get_model_weights is None:
        return weights_name
    weights_enum = get_model_weights(model_name)
    try:
        return getattr(weights_enum, weights_name)
    except AttributeError as exc:
        choices = ", ".join(weight.name for weight in weights_enum)
        raise ValueError(
            f"Unknown weights '{weights_name}' for {model_name}; choose one of: {choices}, DEFAULT"
        ) from exc

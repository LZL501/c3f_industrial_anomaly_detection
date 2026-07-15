from __future__ import annotations

from enum import Enum

import pytest
import torch
from torch.nn import functional as F

from c3f.models import backbone as backbone_module
from c3f.models.c3f import C3FModel
from c3f.models.discriminator import PatchDiscriminator, SegmentationHead
from c3f.models.losses import FocalLoss
from c3f.models.quantizer import VectorQuantizer


def test_backbone_resolves_legacy_imagenet_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeWeights(Enum):
        IMAGENET1K_V1 = "v1"
        IMAGENET1K_V2 = "v2"

    monkeypatch.setattr(
        backbone_module.torchvision.models, "get_model_weights", lambda _: FakeWeights
    )

    assert (
        backbone_module._resolve_weights("wide_resnet50_2", "IMAGENET1K_V1")
        is FakeWeights.IMAGENET1K_V1
    )


def test_c3f_cosine_fusion_clamps_negative_similarity() -> None:
    source = torch.tensor([[[[1.0]], [[0.0]]]])

    same = C3FModel._mix_by_cosine(source, source.clone())
    opposite = C3FModel._mix_by_cosine(source, -source)
    orthogonal = C3FModel._mix_by_cosine(source, torch.tensor([[[[0.0]], [[1.0]]]]))

    torch.testing.assert_close(same, source)
    torch.testing.assert_close(opposite, -source)
    torch.testing.assert_close(orthogonal, torch.tensor([[[[0.0]], [[1.0]]]]))


def test_rough_map_averages_three_feature_stages() -> None:
    encoder = [torch.ones(1, 2, size, size) for size in (8, 4, 2)]
    decoder = [encoder[0], -encoder[1], encoder[2]]

    rough = C3FModel.rough_anomaly_map(encoder, decoder, (16, 16))

    torch.testing.assert_close(rough, torch.full_like(rough, 2.0 / 3.0))


def test_focal_loss_matches_released_probability_formulation() -> None:
    logits = torch.tensor([[[[2.0, -1.0]], [[-2.0, 1.0]]]])
    target = torch.tensor([[[0, 1]]])
    criterion = FocalLoss(gamma=2.0, smooth=1e-5)

    probabilities = torch.softmax(logits, dim=1).movedim(1, -1).reshape(-1, 2)
    one_hot = F.one_hot(target.reshape(-1), num_classes=2).float().clamp(1e-5, 1 - 1e-5)
    pt = (one_hot * probabilities).sum(dim=1) + 1e-5
    expected = (-((1 - pt) ** 2) * pt.log()).mean()

    torch.testing.assert_close(criterion(logits, target), expected)


def test_discriminators_match_released_layer_structure() -> None:
    patch = PatchDiscriminator()
    convolutions = [
        module for module in patch.modules() if isinstance(module, torch.nn.Conv2d)
    ]
    assert [module.out_channels for module in convolutions] == [64, 128, 256, 512, 1]
    assert [module.stride for module in convolutions] == [
        (2, 2),
        (2, 2),
        (2, 2),
        (1, 1),
        (1, 1),
    ]

    segmentation = SegmentationHead()
    assert segmentation.decoder.db2[0].bias is not None
    assert segmentation.decoder.db2[3].bias is not None
    assert segmentation.decoder.db3[0].bias is None
    assert segmentation.decoder.db4[0].bias is None
    assert segmentation.decoder.db4[3].bias is not None


def test_initialized_codebook_can_be_frozen_or_trainable() -> None:
    features = torch.randn(4, 8)
    quantizer = VectorQuantizer(4, 8)
    quantizer.set_codebook(features, freeze=True)
    assert not quantizer.embedding.weight.requires_grad

    quantizer.set_codebook(features, freeze=False)
    assert quantizer.embedding.weight.requires_grad

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .backbone import ResNetBackbone
from .blocks import PositionEmbeddingSine, dissimilarity
from .decoder import Decoder
from .discriminator import SegmentationHead
from .quantizer import VectorQuantizer


class C3FModel(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = config
        self.feature_dims = config["feature_dims"]
        self.pos_dims = config.get("pos_dims", [0, 0, 0, 0])
        self.folds = config["folds"]
        n_embeds = config["n_embeds"]
        self.encoder = ResNetBackbone(
            name=config.get("backbone", "wide_resnet50_2"),
            pretrained=config.get("pretrained_backbone", True),
            weights_name=config.get("backbone_weights", "IMAGENET1K_V1"),
            train_backbone=False,
            return_interm_layers=True,
            dilation=False,
            freeze_running_stats=config.get("freeze_backbone_stats", True),
        )
        self.position = nn.ModuleList(
            [
                PositionEmbeddingSine(pos_dim // 2, normalize_pos=True, temperature=10)
                for pos_dim in self.pos_dims
            ]
        )
        embed_dims = [
            (fd + pd) * fold**2
            for fd, pd, fold in zip(self.feature_dims, self.pos_dims, self.folds)
        ]
        self.quantizers = nn.ModuleList(
            [VectorQuantizer(n, d, beta=0.25) for n, d in zip(n_embeds, embed_dims)]
        )
        self.decoder = Decoder(**config["decoder"])
        self.segmentation_head = SegmentationHead(
            in_channels=6, out_channels=2, base_channels=64
        )
        self.guided_segmentation = bool(config.get("guided_segmentation", True))

    def forward(self, x: torch.Tensor, normal_x: torch.Tensor | None = None) -> dict:
        encoder_features, quants, quant_losses = self.encode(x)
        decoder_features, reconstruction = self.decode(quants)
        rough_map = self.rough_anomaly_map(
            encoder_features, decoder_features, x.shape[-2:]
        )
        segmentation_input = torch.cat((x, reconstruction), dim=1)
        if self.guided_segmentation:
            segmentation_input = segmentation_input * rough_map
        segmentation_logits = self.segmentation_head(segmentation_input)
        outputs = {
            "reconstruction": reconstruction,
            "rough_anomaly_map": rough_map,
            "quant_losses": quant_losses,
            "encoder_features": encoder_features,
            "decoder_features": decoder_features,
            "quants": quants,
            "segmentation_input": segmentation_input,
            "segmentation_logits": segmentation_logits,
        }
        if normal_x is not None:
            with torch.no_grad():
                outputs["normal_features"] = self.encoder(normal_x)
        else:
            outputs["normal_features"] = encoder_features
        return outputs

    def encode(
        self, x: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        encoder_features = self.encoder(x)
        features_with_pos = []
        for feature, pos_embed in zip(encoder_features, self.position):
            pos = pos_embed(feature)
            features_with_pos.append(torch.cat((feature, pos), dim=1))

        quants = []
        losses = []
        for i, feature in enumerate(features_with_pos):
            folded = self._fold_down(feature, self.folds[i])
            quant, loss, _ = self.quantizers[i](folded)
            mixed = self._mix_by_cosine(folded, quant)
            total_loss = loss
            for j in range(int(math.log2(self.folds[i]))):
                h, w = mixed.shape[2:]
                mixed = F.fold(
                    mixed.flatten(2),
                    output_size=(h * 2, w * 2),
                    kernel_size=2,
                    stride=2,
                )
                quant, loss, _ = self.quantizers[i].get_code_scale(
                    mixed, (2 ** (j + 1)) ** 2
                )
                mixed = self._mix_by_cosine(mixed, quant)
                total_loss = total_loss + loss
            quants.append(mixed[:, : self.feature_dims[i], :, :])
            losses.append(total_loss)
        return encoder_features, quants, losses

    def decode(
        self, quants: list[torch.Tensor]
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        return self.decoder(quants)

    @torch.no_grad()
    def codebook_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        encoder_features = self.encoder(x)
        features = []
        for i, feature in enumerate(encoder_features):
            pos = self.position[i](feature)
            folded = self._fold_down(torch.cat((feature, pos), dim=1), self.folds[i])
            features.append(folded.flatten(2).transpose(1, 2).flatten(0, 1))
        return features

    @torch.no_grad()
    def anomaly_map(
        self, x: torch.Tensor, score_mode: str = "segmentation"
    ) -> torch.Tensor:
        outputs = self.forward(x)
        if score_mode == "rough":
            return outputs["rough_anomaly_map"].squeeze(1)
        if score_mode != "segmentation":
            raise ValueError(f"Unknown score mode: {score_mode}")
        return torch.softmax(outputs["segmentation_logits"], dim=1)[:, 1]

    @staticmethod
    def rough_anomaly_map(
        encoder_features: list[torch.Tensor],
        decoder_features: list[torch.Tensor],
        image_size: tuple[int, int],
    ) -> torch.Tensor:
        maps = []
        for enc, dec in zip(encoder_features[:3], decoder_features[:3]):
            score = dissimilarity(dec.contiguous(), enc.contiguous())
            maps.append(
                F.interpolate(
                    score, size=image_size, mode="bilinear", align_corners=False
                )
            )
        return torch.stack(maps, dim=0).mean(dim=0)

    def _fold_down(self, feature: torch.Tensor, fold: int) -> torch.Tensor:
        out = feature
        for _ in range(int(math.log2(fold))):
            b, c, h, w = out.shape
            out = F.unfold(out, kernel_size=2, stride=2).reshape(b, -1, h // 2, w // 2)
        return out

    @staticmethod
    def _mix_by_cosine(source: torch.Tensor, quant: torch.Tensor) -> torch.Tensor:
        weight = torch.cosine_similarity(source, quant, dim=1).unsqueeze(1).clamp_min(0)
        return weight * source + (1 - weight) * quant

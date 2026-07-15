from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .blocks import dissimilarity
from .discriminator import PatchDiscriminator
from .perceptual import LPIPS


class FocalLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float | list[float] | None = None,
        balance_index: int = 0,
        smooth: float = 1e-5,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.balance_index = balance_index
        if not 0 <= smooth <= 1:
            raise ValueError("smooth must be between 0 and 1")
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.ndim == logits.ndim:
            target = target.squeeze(1)
        target = target.long().reshape(-1)
        probabilities = (
            torch.softmax(logits, dim=1).movedim(1, -1).reshape(-1, logits.shape[1])
        )
        one_hot = F.one_hot(target, num_classes=logits.shape[1]).to(probabilities.dtype)
        if self.smooth:
            one_hot = one_hot.clamp(
                self.smooth / (logits.shape[1] - 1), 1.0 - self.smooth
            )
        pt = (one_hot * probabilities).sum(dim=1) + self.smooth
        alpha = self._alpha(logits.shape[1], probabilities)
        alpha_t = alpha[target]
        return (-alpha_t * (1 - pt) ** self.gamma * pt.log()).mean()

    def _alpha(self, num_classes: int, reference: torch.Tensor) -> torch.Tensor:
        if self.alpha is None:
            return reference.new_ones(num_classes)
        if isinstance(self.alpha, float):
            alpha = reference.new_full((num_classes,), 1 - self.alpha)
            alpha[self.balance_index] = self.alpha
            return alpha
        if len(self.alpha) != num_classes:
            raise ValueError(
                f"Expected {num_classes} alpha values, got {len(self.alpha)}"
            )
        alpha = reference.new_tensor(self.alpha)
        return alpha / alpha.sum()


class C3FLoss(nn.Module):
    def __init__(
        self,
        codebook_weight: float = 1.0,
        feature_weight: float = 10.0,
        reconstruction_weight: float = 1.0,
        perceptual_weight: float = 1.0,
        segmentation_weight: float = 1.0,
        adversarial_weight: float = 0.75,
        discriminator_factor: float = 1.0,
        discriminator_start: int = 0,
        perceptual_weights_path: str | None = None,
    ) -> None:
        super().__init__()
        self.codebook_weight = codebook_weight
        self.feature_weight = feature_weight
        self.reconstruction_weight = reconstruction_weight
        self.perceptual_weight = perceptual_weight
        self.segmentation_weight = segmentation_weight
        self.adversarial_weight = adversarial_weight
        self.discriminator_factor = discriminator_factor
        self.discriminator_start = discriminator_start
        self.discriminator = PatchDiscriminator(in_channels=3)
        self.perceptual = (
            LPIPS(perceptual_weights_path) if perceptual_weight > 0 else None
        )
        self.focal = FocalLoss()

    def generator_loss(
        self,
        outputs: dict,
        normal_images: torch.Tensor,
        masks: torch.Tensor,
        global_step: int,
        last_layer: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        reconstruction = outputs["reconstruction"]
        pixel_loss = torch.abs(reconstruction - normal_images)
        if self.perceptual is None:
            perceptual_loss = reconstruction.new_tensor(0.0)
        else:
            perceptual_loss = self.perceptual(normal_images, reconstruction)
        rec_loss = torch.mean(
            self.reconstruction_weight * pixel_loss
            + self.perceptual_weight * perceptual_loss
        )
        feat_loss = sum(
            torch.mean(dissimilarity(enc.contiguous(), dec.contiguous()))
            for enc, dec in zip(outputs["normal_features"], outputs["decoder_features"])
        )
        reconstruction_loss = rec_loss + self.feature_weight * feat_loss
        q_loss = sum(torch.mean(loss) for loss in outputs["quant_losses"])
        if self.segmentation_weight > 0:
            seg_loss = self.focal(outputs["segmentation_logits"], masks)
        else:
            seg_loss = reconstruction.new_tensor(0.0)
        logits_fake = self.discriminator(reconstruction)
        g_loss = -torch.mean(logits_fake)
        disc_factor = (
            self.discriminator_factor
            if global_step >= self.discriminator_start
            else 0.0
        )
        adaptive_weight = (
            self._adaptive_weight(reconstruction_loss, g_loss, last_layer)
            if disc_factor
            else reconstruction.new_tensor(0.0)
        )
        total = (
            reconstruction_loss
            + self.codebook_weight * q_loss
            + self.segmentation_weight * seg_loss
            + adaptive_weight * disc_factor * g_loss
        )
        log = {
            "loss": float(total.detach().cpu()),
            "rec_loss": float(rec_loss.detach().cpu()),
            "perceptual_loss": float(torch.mean(perceptual_loss).detach().cpu()),
            "feature_loss": float(feat_loss.detach().cpu()),
            "quant_loss": float(q_loss.detach().cpu()),
            "seg_loss": float(seg_loss.detach().cpu()),
            "g_loss": float(g_loss.detach().cpu()),
            "adaptive_weight": float(adaptive_weight.detach().cpu()),
            "disc_factor": float(disc_factor),
        }
        return total, log

    def discriminator_loss(
        self, real: torch.Tensor, fake: torch.Tensor, global_step: int
    ) -> tuple[torch.Tensor, dict[str, float]]:
        disc_factor = (
            self.discriminator_factor
            if global_step >= self.discriminator_start
            else 0.0
        )
        if not disc_factor:
            zero = real.new_tensor(0.0, requires_grad=True)
            return zero, {"disc_loss": 0.0}
        logits_real = self.discriminator(real.detach())
        logits_fake = self.discriminator(fake.detach())
        loss_real = torch.mean(F.relu(1.0 - logits_real))
        loss_fake = torch.mean(F.relu(1.0 + logits_fake))
        loss = disc_factor * 0.5 * (loss_real + loss_fake)
        return loss, {"disc_loss": float(loss.detach().cpu())}

    def _adaptive_weight(
        self,
        reconstruction_loss: torch.Tensor,
        generator_loss: torch.Tensor,
        last_layer: torch.Tensor,
    ) -> torch.Tensor:
        reconstruction_grad = torch.autograd.grad(
            reconstruction_loss, last_layer, retain_graph=True
        )[0]
        generator_grad = torch.autograd.grad(
            generator_loss, last_layer, retain_graph=True
        )[0]
        weight = torch.norm(reconstruction_grad) / (torch.norm(generator_grad) + 1e-4)
        return weight.clamp(0.0, 1e4).detach() * self.adversarial_weight

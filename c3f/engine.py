from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torchvision.utils import make_grid
from tqdm import tqdm
from scipy import ndimage

from .data import build_dataloaders
from .metrics import compute_metrics
from .models import C3FModel
from .models.losses import C3FLoss
from .utils import AverageMeter, ensure_dir, write_jsonl

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover - optional dependency guard
    SummaryWriter = None


def build_model(config: dict, device: torch.device) -> C3FModel:
    return C3FModel(config["model"]).to(device)


def build_model_and_loss(
    config: dict, device: torch.device
) -> tuple[C3FModel, C3FLoss]:
    model = build_model(config, device)
    loss_config = config.get("loss", {})
    criterion = C3FLoss(
        codebook_weight=loss_config.get("codebook_weight", 1.0),
        feature_weight=loss_config.get(
            "feature_weight", config.get("train", {}).get("feature_weight", 10.0)
        ),
        reconstruction_weight=loss_config.get("reconstruction_weight", 1.0),
        perceptual_weight=loss_config.get("perceptual_weight", 1.0),
        segmentation_weight=loss_config.get(
            "segmentation_weight",
            config.get("train", {}).get("segmentation_weight", 1.0),
        ),
        adversarial_weight=loss_config.get(
            "adversarial_weight",
            config.get("model", {}).get("adversarial_weight", 0.75),
        ),
        discriminator_factor=loss_config.get("discriminator_factor", 1.0),
        discriminator_start=loss_config.get(
            "discriminator_start", config.get("model", {}).get("discriminator_start", 0)
        ),
        perceptual_weights_path=loss_config.get("perceptual_weights_path"),
    ).to(device)
    return model, criterion


def train(
    config: dict,
    device: torch.device,
    run_dir: Path,
    resume_path: str | Path | None = None,
) -> None:
    ensure_dir(run_dir)
    train_loader, val_loader = build_dataloaders(config)
    model, criterion = build_model_and_loss(config, device)
    resume_path = resume_path or config["train"].get("resume_from")
    if resume_path is None:
        initialize_codebooks(
            model,
            train_loader,
            device,
            max_samples=config["train"].get("support_samples", 500),
            freeze=config["train"].get("freeze_codebook", True),
        )
    else:
        train_codebook = not bool(config["train"].get("freeze_codebook", True))
        for quantizer in model.quantizers:
            quantizer.embedding.weight.requires_grad_(train_codebook)

    lr = config["train"]["lr"]
    model_params = [p for p in model.parameters() if p.requires_grad]
    opt_model = torch.optim.Adam(model_params, lr=lr, betas=(0.5, 0.9))
    opt_disc = torch.optim.Adam(
        criterion.discriminator.parameters(), lr=lr, betas=(0.5, 0.9)
    )
    scaler = GradScaler(enabled=bool(config["train"].get("amp", False)))
    best_metric = -float("inf")
    global_step = 0
    start_epoch = 1
    if resume_path is not None:
        checkpoint = load_training_checkpoint(
            resume_path,
            model,
            criterion,
            opt_model,
            opt_disc,
            scaler,
            map_location=device,
        )
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint.get("global_step", 0))
        best_metric = float(
            checkpoint.get(
                "best_metric",
                checkpoint.get("metrics", {}).get("metric", -float("inf")),
            )
        )
    log_path = run_dir / "metrics.jsonl"
    writer = _build_writer(config, run_dir)

    for epoch in range(start_epoch, config["train"]["epochs"] + 1):
        train_log, global_step = train_one_epoch(
            model,
            criterion,
            train_loader,
            opt_model,
            opt_disc,
            scaler,
            device,
            epoch,
            global_step,
            amp=bool(config["train"].get("amp", False)),
            log_every=config["train"].get("log_every", 20),
            image_log_every=config["train"].get("image_log_every", 200),
            writer=writer,
            synthetic_anomaly=bool(config["train"].get("synthetic_anomaly", True)),
        )
        eval_settings = _eval_settings(config)
        val_log = evaluate(
            model,
            val_loader,
            device,
            writer=writer,
            global_step=global_step,
            **eval_settings,
        )
        record = {
            "epoch": epoch,
            **{f"train/{k}": v for k, v in train_log.items()},
            **{f"val/{k}": v for k, v in val_log.items()},
        }
        write_jsonl(log_path, record)
        _write_scalars(writer, "epoch/train", train_log, epoch)
        _write_scalars(writer, "epoch/val", val_log, epoch)
        is_best = val_log["metric"] > best_metric
        if is_best:
            best_metric = val_log["metric"]
        save_checkpoint(
            run_dir / "last.pth",
            model,
            criterion,
            opt_model,
            opt_disc,
            scaler,
            epoch,
            global_step,
            best_metric,
            config,
            val_log,
        )
        if is_best:
            _atomic_copyfile(run_dir / "last.pth", run_dir / "best.pth")
        print(record)
    if writer is not None:
        writer.close()


def train_one_epoch(
    model: C3FModel,
    criterion: C3FLoss,
    loader,
    opt_model,
    opt_disc,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    global_step: int,
    amp: bool,
    log_every: int,
    image_log_every: int,
    writer,
    synthetic_anomaly: bool,
) -> tuple[dict[str, float], int]:
    model.train()
    criterion.train()
    meters: dict[str, AverageMeter] = {}
    progress = tqdm(loader, desc=f"epoch {epoch}", leave=False)
    for step, batch in enumerate(progress, start=1):
        normal, anomaly, mask, _ = _move_train_batch(batch, device)
        model_input = anomaly if synthetic_anomaly else normal
        normal_x = normal if synthetic_anomaly else None
        if not synthetic_anomaly:
            mask = torch.zeros_like(mask)
        with autocast(enabled=amp):
            outputs = model(model_input, normal_x=normal_x)
            gen_loss, gen_log = criterion.generator_loss(
                outputs,
                normal,
                mask,
                global_step,
                last_layer=model.decoder.conv_out.weight,
            )
        opt_model.zero_grad(set_to_none=True)
        scaler.scale(gen_loss).backward()
        scaler.step(opt_model)

        with autocast(enabled=amp):
            disc_loss, disc_log = criterion.discriminator_loss(
                normal, outputs["reconstruction"], global_step
            )
        opt_disc.zero_grad(set_to_none=True)
        scaler.scale(disc_loss).backward()
        scaler.step(opt_disc)
        scaler.update()

        global_step += 1
        for key, value in {**gen_log, **disc_log}.items():
            meters.setdefault(key, AverageMeter()).update(value, normal.size(0))
        if step % log_every == 0:
            progress.set_postfix(
                {
                    k: f"{m.avg:.4f}"
                    for k, m in meters.items()
                    if k in {"loss", "seg_loss", "disc_loss"}
                }
            )
            _write_scalars(
                writer, "step/train", {k: m.avg for k, m in meters.items()}, global_step
            )
        if (
            writer is not None
            and image_log_every > 0
            and global_step % image_log_every == 0
        ):
            _write_image_panel(
                writer, "train/images", normal, model_input, outputs, mask, global_step
            )
    return {key: meter.avg for key, meter in meters.items()}, global_step


@torch.no_grad()
def evaluate(
    model: C3FModel,
    loader,
    device: torch.device,
    writer=None,
    global_step: int | None = None,
    score_mode: str = "segmentation",
    smooth_sigma: float = 0.0,
    localization_metric: str = "aupro",
    aupro_mode: str = "per_image",
) -> dict[str, float]:
    model.eval()
    preds = []
    masks = []
    labels = []
    for batch_idx, batch in enumerate(tqdm(loader, desc="eval", leave=False)):
        image, mask, label = _move_eval_batch(batch, device)
        outputs = model(image)
        pred = _prediction_from_outputs(outputs, score_mode)
        if batch_idx == 0 and writer is not None and global_step is not None:
            _write_eval_image_panel(
                writer, "val/images", image, outputs, mask, global_step, score_mode
            )
        pred_np = pred.cpu().numpy()
        if smooth_sigma > 0:
            pred_np = np.stack(
                [ndimage.gaussian_filter(p, sigma=smooth_sigma) for p in pred_np],
                axis=0,
            )
        preds.append(pred_np)
        masks.append(mask.cpu().numpy())
        labels.append(label.cpu().numpy())
    return compute_metrics(
        np.concatenate(preds),
        np.concatenate(masks),
        np.concatenate(labels),
        localization_metric=localization_metric,
        aupro_mode=aupro_mode,
    )


@torch.no_grad()
def initialize_codebooks(
    model: C3FModel,
    loader,
    device: torch.device,
    max_samples: int = 500,
    freeze: bool = True,
) -> None:
    model.eval()
    banks: list[list[torch.Tensor]] | None = None
    seen = 0
    for batch in tqdm(loader, desc="support features", leave=False):
        normal = batch[0].to(device, non_blocking=True)
        features = model.codebook_features(normal)
        if banks is None:
            banks = [[] for _ in features]
        for bank, feature in zip(banks, features):
            bank.append(feature.cpu())
        seen += normal.size(0)
        if seen >= max_samples:
            break
    if banks is None:
        raise RuntimeError("No support features collected.")
    for quantizer, chunks in zip(model.quantizers, banks):
        features = torch.cat(chunks, dim=0)
        selected = coreset_or_repeat(features, quantizer.n_embed)
        quantizer.set_codebook(selected.to(device), freeze=freeze)


def coreset_or_repeat(
    features: torch.Tensor, keep: int, starting_points: int = 10
) -> torch.Tensor:
    if features.size(0) == 0:
        raise ValueError("Cannot build codebook from empty features.")
    if features.size(0) < keep:
        indices = torch.randint(0, features.size(0), (keep,))
        return features[indices]
    if features.size(0) == keep:
        return features
    features = features.float()
    starts = torch.randperm(features.size(0))[: min(starting_points, features.size(0))]
    distances = torch.cdist(features, features[starts]).mean(dim=1, keepdim=True)
    selected = []
    for _ in tqdm(range(keep), desc="coreset", leave=False):
        idx = torch.argmax(distances).item()
        selected.append(idx)
        new_dist = torch.cdist(features, features[idx : idx + 1])
        distances = torch.minimum(distances, new_dist)
    return features[selected]


def save_checkpoint(
    path: Path,
    model: nn.Module,
    criterion: C3FLoss,
    opt_model,
    opt_disc,
    scaler: GradScaler,
    epoch: int,
    global_step: int,
    best_metric: float,
    config: dict,
    metrics: dict,
) -> None:
    checkpoint = {
        "format_version": 2,
        "model": model.state_dict(),
        "discriminator": criterion.discriminator.state_dict(),
        "opt_model": opt_model.state_dict(),
        "opt_disc": opt_disc.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_metric": best_metric,
        "config": config,
        "metrics": metrics,
    }
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        torch.save(checkpoint, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_copyfile(source: Path, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        try:
            os.link(source, temporary)
        except OSError:
            shutil.copyfile(source, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    criterion: C3FLoss | None = None,
    map_location: str | torch.device = "cpu",
) -> dict:
    checkpoint = _torch_load(path, map_location)
    model.load_state_dict(checkpoint["model"])
    if criterion is not None:
        if "discriminator" in checkpoint:
            criterion.discriminator.load_state_dict(checkpoint["discriminator"])
        elif "criterion" in checkpoint:
            criterion.load_state_dict(checkpoint["criterion"])
    return checkpoint


def load_training_checkpoint(
    path: str | Path,
    model: nn.Module,
    criterion: C3FLoss,
    opt_model,
    opt_disc,
    scaler: GradScaler,
    map_location: str | torch.device = "cpu",
) -> dict:
    checkpoint = load_checkpoint(path, model, criterion, map_location)
    opt_model.load_state_dict(checkpoint["opt_model"])
    opt_disc.load_state_dict(checkpoint["opt_disc"])
    if "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
    return checkpoint


def _torch_load(path: str | Path, map_location: str | torch.device) -> dict:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # PyTorch < 2.0
        return torch.load(path, map_location=map_location)


def _build_writer(config: dict, run_dir: Path):
    if not config.get("train", {}).get("tensorboard", True):
        return None
    if SummaryWriter is None:
        print("TensorBoard is not installed; continuing with JSONL logs only.")
        return None
    return SummaryWriter(log_dir=str(run_dir / "tensorboard"))


def _write_scalars(writer, prefix: str, values: dict[str, float], step: int) -> None:
    if writer is None:
        return
    for key, value in values.items():
        if value == value:
            writer.add_scalar(f"{prefix}/{key}", value, step)


def _write_image_panel(
    writer,
    tag: str,
    normal: torch.Tensor,
    anomaly: torch.Tensor,
    outputs: dict,
    mask: torch.Tensor,
    step: int,
) -> None:
    if writer is None:
        return
    with torch.no_grad():
        normal_img = _denormalize(normal[:4].detach().cpu())
        anomaly_img = _denormalize(anomaly[:4].detach().cpu())
        recon_img = _denormalize(outputs["reconstruction"][:4].detach().cpu())
        rough = outputs["rough_anomaly_map"][:4].detach().cpu()
        pred = torch.softmax(outputs["segmentation_logits"][:4].detach().cpu(), dim=1)[
            :, 1:2
        ]
        rough_heat = _heatmap(_normalize_map(rough))
        heat = _heatmap(pred)
        mask_img = mask[:4].detach().cpu().float().unsqueeze(1).repeat(1, 3, 1, 1)
        panel = torch.cat(
            (normal_img, anomaly_img, recon_img, rough_heat, heat, mask_img), dim=0
        )
        writer.add_image(tag, make_grid(panel, nrow=4), step)


def _write_eval_image_panel(
    writer,
    tag: str,
    image: torch.Tensor,
    outputs: dict,
    mask: torch.Tensor,
    step: int,
    score_mode: str,
) -> None:
    if writer is None:
        return
    with torch.no_grad():
        image_img = _denormalize(image[:4].detach().cpu())
        recon_img = _denormalize(outputs["reconstruction"][:4].detach().cpu())
        rough = outputs["rough_anomaly_map"][:4].detach().cpu()
        pred = _prediction_from_outputs(
            {k: v[:4].detach().cpu() for k, v in outputs.items() if torch.is_tensor(v)},
            score_mode,
        ).unsqueeze(1)
        rough_heat = _heatmap(_normalize_map(rough))
        heat = _heatmap(pred)
        mask_img = mask[:4].detach().cpu().float().unsqueeze(1).repeat(1, 3, 1, 1)
        panel = torch.cat((image_img, recon_img, rough_heat, heat, mask_img), dim=0)
        writer.add_image(tag, make_grid(panel, nrow=4), step)


def _denormalize(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (x * std + mean).clamp(0, 1)


def _heatmap(pred: torch.Tensor) -> torch.Tensor:
    pred = pred.clamp(0, 1)
    red = pred
    green = 1 - (pred - 0.5).abs() * 2
    blue = 1 - pred
    return torch.cat((red, green.clamp(0, 1), blue), dim=1)


def _normalize_map(pred: torch.Tensor) -> torch.Tensor:
    flat = pred.flatten(1)
    min_v = flat.min(dim=1).values.view(-1, 1, 1, 1)
    max_v = flat.max(dim=1).values.view(-1, 1, 1, 1)
    return ((pred - min_v) / (max_v - min_v).clamp_min(1e-6)).clamp(0, 1)


def _prediction_from_outputs(outputs: dict, score_mode: str) -> torch.Tensor:
    if score_mode == "rough":
        return outputs["rough_anomaly_map"].squeeze(1)
    if score_mode == "segmentation":
        return torch.softmax(outputs["segmentation_logits"], dim=1)[:, 1]
    raise ValueError(f"Unknown score mode: {score_mode}")


def _eval_settings(config: dict) -> dict[str, object]:
    eval_cfg = config.get("eval", {})
    dataset = config.get("data", {}).get("dataset", "").lower()
    default_localization = "pixel_auroc" if dataset == "mvtec" else "aupro"
    default_score_mode = "rough" if dataset == "mvtec" else "segmentation"
    default_sigma = 4.0 if default_score_mode == "rough" else 0.0
    return {
        "score_mode": eval_cfg.get("score_mode", default_score_mode),
        "smooth_sigma": float(eval_cfg.get("smooth_sigma", default_sigma)),
        "localization_metric": eval_cfg.get(
            "localization_metric", default_localization
        ),
        "aupro_mode": eval_cfg.get("aupro_mode", "per_image"),
    }


def _move_train_batch(batch, device: torch.device):
    normal, anomaly, mask, target = batch
    return (
        normal.to(device, non_blocking=True),
        anomaly.to(device, non_blocking=True),
        mask.to(device, non_blocking=True),
        target.to(device, non_blocking=True),
    )


def _move_eval_batch(batch, device: torch.device):
    image, mask, label = batch
    return (
        image.to(device, non_blocking=True),
        mask.to(device, non_blocking=True),
        label.to(device, non_blocking=True),
    )

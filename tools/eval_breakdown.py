from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy import ndimage
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from c3f.config import load_config, merge_overrides, parse_dotlist
from c3f.data import build_eval_dataloader
from c3f.engine import _eval_settings, build_model, load_checkpoint
from c3f.metrics import compute_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate C3F by MVTec defect type.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--category")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    config = load_config(args.config)
    cli_overrides = parse_dotlist(args.overrides)
    if args.data_root:
        cli_overrides["data.root"] = args.data_root
    if args.category:
        cli_overrides["data.category"] = args.category
    config = merge_overrides(config, cli_overrides)
    if config["data"]["dataset"].lower() != "mvtec":
        raise ValueError("Per-defect breakdown is currently defined for MVTec only.")

    device = torch.device(args.device)
    loader = build_eval_dataloader(config)
    model = build_model(config, device)
    checkpoint = load_checkpoint(args.checkpoint, model, map_location=device)
    settings = _eval_settings(config)
    preds, masks, labels = collect_predictions(
        model,
        loader,
        device,
        score_mode=str(settings["score_mode"]),
        smooth_sigma=float(settings["smooth_sigma"]),
    )
    defect_types = np.asarray([path.parent.name for path in loader.dataset.image_paths])
    results = build_breakdown(
        preds,
        masks,
        labels,
        defect_types,
        localization_metric=str(settings["localization_metric"]),
        aupro_mode=str(settings["aupro_mode"]),
    )
    payload = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "category": config["data"]["category"],
        "score_mode": settings["score_mode"],
        "smooth_sigma": settings["smooth_sigma"],
        "aupro_mode": settings["aupro_mode"],
        "results": results,
    }
    print_table(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


@torch.no_grad()
def collect_predictions(
    model,
    loader,
    device: torch.device,
    score_mode: str,
    smooth_sigma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_preds = []
    all_masks = []
    all_labels = []
    for image, mask, label in tqdm(loader, desc="eval", leave=False):
        image = image.to(device, non_blocking=True)
        outputs = model(image)
        if score_mode == "rough":
            pred = outputs["rough_anomaly_map"].squeeze(1)
        elif score_mode == "segmentation":
            pred = torch.softmax(outputs["segmentation_logits"], dim=1)[:, 1]
        else:
            raise ValueError(f"Unknown score mode: {score_mode}")
        pred_np = pred.cpu().numpy()
        if smooth_sigma > 0:
            pred_np = np.stack(
                [ndimage.gaussian_filter(item, sigma=smooth_sigma) for item in pred_np]
            )
        all_preds.append(pred_np)
        all_masks.append(mask.numpy())
        all_labels.append(label.numpy())
    return (
        np.concatenate(all_preds),
        np.concatenate(all_masks),
        np.concatenate(all_labels),
    )


def build_breakdown(
    preds: np.ndarray,
    masks: np.ndarray,
    labels: np.ndarray,
    defect_types: np.ndarray,
    localization_metric: str,
    aupro_mode: str,
) -> dict[str, dict[str, float | int]]:
    results = {
        "all": with_count(
            compute_metrics(
                preds,
                masks,
                labels,
                localization_metric,
                aupro_mode=aupro_mode,
            ),
            len(labels),
        )
    }
    good = defect_types == "good"
    for defect_type in sorted(set(defect_types) - {"good"}):
        selected = good | (defect_types == defect_type)
        metrics = compute_metrics(
            preds[selected],
            masks[selected],
            labels[selected],
            localization_metric,
            aupro_mode=aupro_mode,
        )
        results[defect_type] = with_count(metrics, int(selected.sum()))
        results[defect_type]["anomaly_count"] = int((defect_types == defect_type).sum())
    return results


def with_count(metrics: dict[str, float], count: int) -> dict[str, float | int]:
    return {"count": count, **metrics}


def print_table(payload: dict) -> None:
    print(
        f"category={payload['category']} checkpoint_epoch={payload['checkpoint_epoch']} "
        f"score_mode={payload['score_mode']} smooth_sigma={payload['smooth_sigma']}"
    )
    print("defect_type\tcount\tanomaly_count\timage_auroc\tpixel_auroc\taupro\tmetric")
    for defect_type, metrics in payload["results"].items():
        values = [
            defect_type,
            str(metrics["count"]),
            str(metrics.get("anomaly_count", "-")),
            percent(metrics["image_auroc"]),
            percent(metrics["pixel_auroc"]),
            percent(metrics["aupro"]),
            f"{metrics['metric']:.6f}",
        ]
        print("\t".join(values))


def percent(value: float) -> str:
    return "nan" if np.isnan(value) else f"{100 * value:.2f}"


if __name__ == "__main__":
    main()

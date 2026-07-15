from __future__ import annotations

import math

import numpy as np
from scipy import ndimage


def image_scores(preds: np.ndarray) -> np.ndarray:
    return np.max(preds.reshape(preds.shape[0], -1), axis=1)


def safe_roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(np.int64).reshape(-1)
    if len(np.unique(labels)) < 2:
        return math.nan
    return float(binary_roc_auc(labels, scores.reshape(-1)))


def compute_metrics(
    preds: np.ndarray,
    masks: np.ndarray,
    labels: np.ndarray,
    localization_metric: str = "aupro",
    aupro_mode: str = "per_image",
) -> dict[str, float]:
    preds = np.asarray(preds)
    masks = np.asarray(masks)
    labels = np.asarray(labels).reshape(-1)
    pixel_auroc = safe_roc_auc(masks.reshape(-1), preds.reshape(-1))
    image_auroc = safe_roc_auc(labels, image_scores(preds))
    anomaly = labels > 0
    if not np.any(anomaly):
        aupro = math.nan
    elif aupro_mode == "per_image":
        per_image = [
            compute_aupro(mask[None], pred[None])
            for mask, pred in zip(masks[anomaly], preds[anomaly])
        ]
        aupro = float(np.mean(per_image))
    elif aupro_mode == "dataset":
        aupro = compute_aupro(masks[anomaly], preds[anomaly])
    else:
        raise ValueError(f"Unknown AUPRO mode: {aupro_mode}")
    if localization_metric == "pixel_auroc":
        selected_localization = pixel_auroc
    elif localization_metric == "aupro":
        selected_localization = aupro
    else:
        raise ValueError(f"Unknown localization metric: {localization_metric}")
    metric = _nan_to_zero(image_auroc) + _nan_to_zero(selected_localization)
    return {
        "image_auroc": image_auroc,
        "pixel_auroc": pixel_auroc,
        "aupro": aupro,
        "metric": metric,
    }


def compute_aupro(
    masks: np.ndarray,
    amaps: np.ndarray,
    num_thresholds: int = 200,
    max_fpr: float = 0.3,
) -> float:
    if masks.size == 0:
        return math.nan
    masks = (masks > 0).astype(np.uint8)
    min_th = float(amaps.min())
    max_th = float(amaps.max())
    if max_th <= min_th:
        return math.nan
    thresholds = np.arange(min_th, max_th, (max_th - min_th) / num_thresholds)
    pros = []
    fprs = []
    inverse_masks = 1 - masks
    inverse_area = max(int(inverse_masks.sum()), 1)
    for th in thresholds:
        binary = (amaps > th).astype(np.uint8)
        per_region = []
        for binary_amap, mask in zip(binary, masks):
            labeled, num_regions = ndimage.label(
                mask, structure=np.ones((3, 3), dtype=np.uint8)
            )
            for region_id in range(1, num_regions + 1):
                region = labeled == region_id
                area = int(region.sum())
                if area == 0:
                    continue
                tp = int(binary_amap[region].sum())
                per_region.append(tp / area)
        if not per_region:
            continue
        fp = np.logical_and(inverse_masks, binary).sum()
        fpr = fp / inverse_area
        if fpr < max_fpr:
            pros.append(float(np.mean(per_region)))
            fprs.append(float(fpr))
    if len(fprs) < 2:
        return math.nan
    fprs = np.asarray(fprs)
    pros = np.asarray(pros)
    order = np.argsort(fprs)
    fprs = fprs[order]
    pros = pros[order]
    fprs = fprs / max(float(fprs.max()), 1e-12)
    trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(trapezoid(pros, fprs))


def binary_roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(np.int64)
    scores = scores.astype(np.float64)
    n_pos = int(labels.sum())
    n_neg = int(labels.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return math.nan
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    sorted_scores = scores[order]
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    pos_rank_sum = ranks[labels == 1].sum()
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _nan_to_zero(value: float) -> float:
    return 0.0 if math.isnan(value) else value

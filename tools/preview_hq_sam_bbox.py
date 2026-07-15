#!/usr/bin/env python3
"""Preview HQ-SAM foreground masks from category-specific box prompts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--categories", nargs="+", default=["screw", "transistor"])
    parser.add_argument("--count", type=int, default=18)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-type", default="vit_b")
    parser.add_argument(
        "--transistor-box-variant",
        choices=["wide", "tight", "extra_tight", "body", "fixed_mid", "fixed_wide"],
        default="wide",
    )
    parser.add_argument("--postprocess-largest", action="store_true")
    parser.add_argument(
        "--transistor-points", choices=["none", "fixed"], default="none"
    )
    parser.add_argument("--hq-token-only", action="store_true", default=True)
    parser.add_argument(
        "--no-hq-token-only", dest="hq_token_only", action="store_false"
    )
    return parser.parse_args()


def largest_component(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return mask.astype(bool)
    areas = stats[1:, cv2.CC_STAT_AREA]
    label = int(areas.argmax()) + 1
    return labels == label


def mask_bbox(mask: np.ndarray, pad: int = 12) -> list[int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        h, w = mask.shape
        return [0, 0, w - 1, h - 1]
    h, w = mask.shape
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return [
        max(0, x0 - pad),
        max(0, y0 - pad),
        min(w - 1, x1 + pad),
        min(h - 1, y1 + pad),
    ]


def screw_box(rgb: np.ndarray) -> list[int]:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    return mask_bbox(largest_component(mask > 0), pad=28)


def transistor_box(rgb: np.ndarray, variant: str = "wide") -> list[int]:
    if variant == "fixed_mid":
        h, w = rgb.shape[:2]
        return [int(0.24 * w), int(0.12 * h), int(0.76 * w), int(0.80 * h)]
    if variant == "fixed_wide":
        h, w = rgb.shape[:2]
        return [int(0.18 * w), int(0.10 * h), int(0.82 * w), int(0.82 * h)]

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    dark = gray < 75
    dark[:80, :] = False
    dark[-80:, :] = False
    dark[:, :80] = False
    dark[:, -80:] = False
    body = largest_component(dark)
    x0, y0, x1, y1 = mask_bbox(body, pad=0)
    h, w = gray.shape
    bw = max(1, x1 - x0 + 1)
    bh = max(1, y1 - y0 + 1)
    settings = {
        "wide": (0.45, 0.20, 0.45, 1.75),
        "tight": (0.22, 0.16, 0.22, 1.55),
        "extra_tight": (0.10, 0.12, 0.10, 1.45),
        "body": (0.08, 0.10, 0.08, 0.18),
    }
    left, top, right, bottom = settings[variant]
    return [
        max(0, int(x0 - left * bw)),
        max(0, int(y0 - top * bh)),
        min(w - 1, int(x1 + right * bw)),
        min(h - 1, int(y1 + bottom * bh)),
    ]


def prompt_box(category: str, rgb: np.ndarray, args: argparse.Namespace) -> list[int]:
    if category == "screw":
        return screw_box(rgb)
    if category == "transistor":
        return transistor_box(rgb, args.transistor_box_variant)
    if category == "zipper":
        return zipper_box(rgb)
    h, w = rgb.shape[:2]
    return [0, 0, w - 1, h - 1]


def zipper_box(rgb: np.ndarray) -> list[int]:
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    mask = (mask > 0).astype(np.uint8)
    mask = keep_center_components_preview(
        mask, min_area_ratio=0.002, max_components=8, x_range=(0.10, 0.90)
    )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, np.ones((17, 9), np.uint8), iterations=2
    )
    return mask_bbox(mask > 0, pad=max(6, int(0.02 * max(h, w))))


def keep_center_components_preview(
    mask: np.ndarray,
    min_area_ratio: float,
    max_components: int | None = None,
    x_range: tuple[float, float] = (0.0, 1.0),
    y_range: tuple[float, float] = (0.0, 1.0),
) -> np.ndarray:
    mask = mask.astype(np.uint8)
    h, w = mask.shape
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    image_area = h * w
    min_area = max(1, int(image_area * min_area_ratio))
    comps = []
    for label in range(1, n):
        area = int(stats[label, cv2.CC_STAT_AREA])
        cx, cy = centroids[label]
        if area < min_area:
            continue
        if not (
            x_range[0] * w <= cx <= x_range[1] * w
            and y_range[0] * h <= cy <= y_range[1] * h
        ):
            continue
        comps.append((area, label))
    comps.sort(reverse=True)
    if max_components is not None:
        comps = comps[:max_components]
    out = np.zeros_like(mask)
    for _, label in comps:
        out[labels == label] = 1
    return out


def overlay_mask(rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    out = rgb.copy()
    color = np.zeros_like(out)
    color[..., 0] = 220
    color[..., 1] = 35
    masked = mask.astype(bool)
    out[masked] = (out[masked] * (1 - alpha) + color[masked] * alpha).astype(np.uint8)
    return out


def draw_box(rgb: np.ndarray, box: Iterable[int]) -> np.ndarray:
    out = rgb.copy()
    x0, y0, x1, y1 = [int(v) for v in box]
    cv2.rectangle(out, (x0, y0), (x1, y1), (30, 220, 70), 6)
    return out


def draw_points(
    rgb: np.ndarray, points: np.ndarray | None, labels: np.ndarray | None
) -> np.ndarray:
    out = rgb.copy()
    if points is None or labels is None:
        return out
    for (x, y), label in zip(points.astype(int), labels.astype(int)):
        color = (30, 220, 70) if label == 1 else (230, 35, 35)
        cv2.circle(out, (int(x), int(y)), 12, color, -1)
        cv2.circle(out, (int(x), int(y)), 14, (255, 255, 255), 2)
    return out


def prompt_points(
    category: str,
    rgb: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if category == "zipper":
        h, w = rgb.shape[:2]
        x0, y0, x1, y1 = prompt_box(category, rgb, args)
        bw = max(1, x1 - x0)
        bh = max(1, y1 - y0)
        positives = [
            (x0 + 0.25 * bw, y0 + 0.18 * bh),
            (x0 + 0.50 * bw, y0 + 0.18 * bh),
            (x0 + 0.75 * bw, y0 + 0.18 * bh),
            (x0 + 0.25 * bw, y0 + 0.50 * bh),
            (x0 + 0.50 * bw, y0 + 0.50 * bh),
            (x0 + 0.75 * bw, y0 + 0.50 * bh),
            (x0 + 0.25 * bw, y0 + 0.82 * bh),
            (x0 + 0.50 * bw, y0 + 0.82 * bh),
            (x0 + 0.75 * bw, y0 + 0.82 * bh),
        ]
        negatives = [
            (max(0, x0 - 0.20 * bw), y0 + 0.25 * bh),
            (min(w - 1, x1 + 0.20 * bw), y0 + 0.25 * bh),
            (max(0, x0 - 0.20 * bw), y0 + 0.75 * bh),
            (min(w - 1, x1 + 0.20 * bw), y0 + 0.75 * bh),
        ]
        points = np.asarray(positives + negatives, dtype=np.float32)
        labels = np.asarray([1] * len(positives) + [0] * len(negatives), dtype=np.int32)
        return points, labels
    if category != "transistor" or args.transistor_points == "none":
        return None, None
    h, w = rgb.shape[:2]
    positives = [
        (0.50 * w, 0.30 * h),
        (0.38 * w, 0.62 * h),
        (0.50 * w, 0.62 * h),
        (0.62 * w, 0.62 * h),
    ]
    negatives = [
        (0.22 * w, 0.23 * h),
        (0.22 * w, 0.45 * h),
        (0.22 * w, 0.74 * h),
        (0.78 * w, 0.23 * h),
        (0.78 * w, 0.45 * h),
        (0.78 * w, 0.74 * h),
    ]
    points = np.asarray(positives + negatives, dtype=np.float32)
    labels = np.asarray([1] * len(positives) + [0] * len(negatives), dtype=np.int32)
    return points, labels


def panel(image: np.ndarray, label: str, size: int = 160) -> Image.Image:
    img = Image.fromarray(image).resize((size, size), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size + 24), "white")
    canvas.paste(img, (0, 24))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 5), label[:32], fill=(0, 0, 0))
    return canvas


def make_sheet(rows: list[list[Image.Image]], out_path: Path) -> None:
    if not rows:
        return
    row_width = max(sum(cell.width for cell in row) for row in rows)
    row_height = max(cell.height for row in rows for cell in row)
    sheet = Image.new("RGB", (row_width, row_height * len(rows)), "white")
    y = 0
    for row in rows:
        x = 0
        for cell in row:
            sheet.paste(cell, (x, y))
            x += cell.width
        y += row_height
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=95)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    from segment_anything import SamPredictor, sam_model_registry

    sam = sam_model_registry[args.model_type](checkpoint=str(args.checkpoint))
    sam.to(device=args.device)
    predictor = SamPredictor(sam)

    all_stats = {}
    for category in args.categories:
        image_dir = args.data_root / category / "train" / "good"
        image_paths = sorted(image_dir.glob("*.png"))[: args.count]
        rows = []
        category_stats = []
        for path in image_paths:
            rgb = np.asarray(Image.open(path).convert("RGB"))
            box = prompt_box(category, rgb, args)
            points, point_labels = prompt_points(category, rgb, args)
            predictor.set_image(rgb)
            masks, scores, _ = predictor.predict(
                point_coords=points,
                point_labels=point_labels,
                box=np.asarray(box, dtype=np.float32),
                multimask_output=False,
                hq_token_only=args.hq_token_only,
            )
            mask = masks[0].astype(bool)
            if args.postprocess_largest:
                mask = largest_component(mask)
            area = float(mask.mean())
            score = float(scores[0]) if len(scores) else 0.0

            mask_rgb = np.repeat(mask[:, :, None], 3, axis=2).astype(np.uint8) * 255
            label = f"{path.name} a={area:.3f} s={score:.3f}"
            rows.append(
                [
                    panel(rgb, path.name),
                    panel(
                        draw_points(draw_box(rgb, box), points, point_labels), "prompt"
                    ),
                    panel(mask_rgb, f"mask a={area:.3f}"),
                    panel(overlay_mask(rgb, mask), f"overlay s={score:.3f}"),
                ]
            )
            category_stats.append(
                {
                    "file": str(path),
                    "box": [int(v) for v in box],
                    "area": area,
                    "score": score,
                    "label": label,
                }
            )

        make_sheet(rows, args.output_dir / f"{category}_hq_sam_bbox.jpg")
        all_stats[category] = category_stats

    with (args.output_dir / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()

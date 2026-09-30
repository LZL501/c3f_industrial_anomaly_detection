"""Refine reviewed MVTec foreground defects in a separate output directory.

Metal-nut voids stay background; zipper fabric follows its visible silhouette;
cable masks use individually traced polygons including the outer sheath. Other
categories retain existing masks, with whole-image masks for texture classes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from refine_cable_foregrounds import load_annotations, read_manual_mask

TEXTURES = {"carpet", "grid", "leather", "tile", "wood"}
CATEGORIES = (
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "wood",
    "zipper",
)


def largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep the main object, rejecting empty segmentations."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    if count < 2:
        raise ValueError("Empty foreground")
    return (labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)


def fill_interior(mask: np.ndarray) -> np.ndarray:
    """Fill enclosed holes even when the object touches an image border."""
    padded = np.pad((mask > 0).astype(np.uint8), 1)
    flood = padded.copy()
    cv2.floodFill(flood, None, (0, 0), 1)
    return (padded | (1 - flood))[1:-1, 1:-1]


def metal_nut_foreground(
    image: np.ndarray, baseline: np.ndarray
) -> tuple[np.ndarray, dict]:
    height, width = baseline.shape
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    yy, xx = np.indices(gray.shape)
    ys, xs = np.where(baseline > 0)
    cx, cy = float(xs.mean()), float(ys.mean())
    radius = 0.22 * min(height, width)
    central = (xx - cx) ** 2 + (yy - cy) ** 2 < radius**2
    border = np.r_[
        gray[:10].ravel(),
        gray[-10:].ravel(),
        gray[:, :10].ravel(),
        gray[:, -10:].ravel(),
    ]
    threshold = float(np.clip(np.quantile(border, 0.95) + 3, 20, 55))
    dark = ((gray <= threshold) & central).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    void = fill_interior(largest_component(dark))
    contours, _ = cv2.findContours(void, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 5:
        raise ValueError("Cannot identify metal-nut aperture")
    ellipse = cv2.fitEllipse(contour)
    (hx, hy), (a, b), _ = ellipse
    if not (
        abs(hx - cx) < 0.08 * width
        and abs(hy - cy) < 0.08 * height
        and 0.12 * min(height, width) < min(a, b)
        and max(a, b) < 0.35 * min(height, width)
        and max(a, b) / min(a, b) < 1.3
    ):
        raise ValueError("Unexpected metal-nut aperture geometry; inspect manually")
    hole = np.zeros_like(baseline, dtype=np.uint8)
    cv2.ellipse(hole, ellipse, 1, thickness=-1)
    result = (baseline > 0).astype(np.uint8)
    result[hole > 0] = 0
    return result, {
        "hole_center": [hx, hy],
        "hole_axes": [a, b],
        "background_threshold": threshold,
    }


def zipper_foreground(image: np.ndarray) -> tuple[np.ndarray, dict]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    threshold, raw = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    mask = fill_interior(largest_component(raw))
    # Fill cloth interior row by row, without expanding to a rectangular box.
    for row in mask:
        xs = np.flatnonzero(row)
        if not len(xs):
            raise ValueError("Zipper does not span every row; inspect manually")
        row[xs[0] : xs[-1] + 1] = 1
    return mask, {"intensity_threshold": threshold}


def refine(
    category: str, image: np.ndarray, baseline: np.ndarray
) -> tuple[np.ndarray, dict]:
    """Return a binary foreground and category-specific provenance."""
    if image.shape[:2] != baseline.shape:
        raise ValueError("Image and foreground dimensions differ")
    if category not in CATEGORIES:
        raise ValueError(f"Unsupported category: {category}")
    details = {}
    if category in TEXTURES:
        mask = np.ones(baseline.shape, np.uint8)
    elif category == "metal_nut":
        mask, details = metal_nut_foreground(image, baseline)
    elif category == "zipper":
        mask, details = zipper_foreground(image)
    elif category == "cable":
        raise ValueError(
            "Cable requires direct manual annotations; use the CLI or read_manual_mask"
        )
    else:
        mask = (baseline > 0).astype(np.uint8)
    if not mask.any():
        raise ValueError("Empty foreground")
    return mask, details


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--categories", nargs="+", choices=CATEGORIES, default=list(CATEGORIES)
    )
    parser.add_argument(
        "--cable-annotations",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "assets/foreground/cable_manual.json",
    )
    args = parser.parse_args()
    cable_records = (
        load_annotations(args.cable_annotations) if "cable" in args.categories else {}
    )
    if args.data_root.resolve() == args.output_root.resolve():
        raise ValueError("Use a separate output root to preserve experiment inputs")
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"Output root must be empty: {args.output_root}")
    cv2.setNumThreads(2)
    records = []
    args.output_root.mkdir(parents=True, exist_ok=True)
    for category in args.categories:
        paths = sorted((args.data_root / category / "train/good").glob("*.png"))
        if not paths:
            raise FileNotFoundError(f"No training images for {category}")
        output = args.output_root / category / "train/foreground"
        output.mkdir(parents=True, exist_ok=True)
        for path in paths:
            image = cv2.imread(str(path))
            mask_path = args.data_root / category / "train/foreground" / path.name
            baseline = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if image is None or baseline is None:
                raise FileNotFoundError(
                    f"Unreadable image or mask: {category}/{path.name}"
                )
            try:
                if category == "cable":
                    if path.name not in cable_records:
                        raise ValueError(
                            "Missing direct manual annotation; no automatic fallback"
                        )
                    mask = read_manual_mask(path, cable_records[path.name])
                    details = {
                        "method": "direct_manual_polygon",
                        "annotations_sha256": hashlib.sha256(
                            args.cable_annotations.read_bytes()
                        ).hexdigest(),
                    }
                else:
                    mask, details = refine(category, image, baseline)
            except ValueError as exc:
                raise ValueError(f"{category}/{path.name}: {exc}") from exc
            changed = int(np.count_nonzero(mask != (baseline > 0)))
            target = output / path.name
            if changed == 0:
                target.write_bytes(mask_path.read_bytes())
            elif not cv2.imwrite(str(target), mask * 255):
                raise OSError(f"Cannot write mask: {target}")
            records.append(
                {
                    "category": category,
                    "image": path.name,
                    "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "baseline_sha256": hashlib.sha256(
                        mask_path.read_bytes()
                    ).hexdigest(),
                    "mask_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                    "changed_pixels": changed,
                    "foreground_fraction": float(mask.mean()),
                    "method": "direct_manual_polygon"
                    if category == "cable"
                    else ("retained" if changed == 0 else category + "_refined"),
                    "details": details,
                    "visual_review": "pending",
                }
            )
        print(category, len(paths), "masks", flush=True)
        # A per-category checkpoint keeps provenance if a later category needs review.
        (args.output_root / "manifest.json").write_text(
            json.dumps({"images": records}, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()

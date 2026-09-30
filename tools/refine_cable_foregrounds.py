"""Rasterize directly traced cable polygons; never infer a foreground boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def rasterize_cable(record: dict) -> np.ndarray:
    """Fill exactly the supplied vertices without fitting or color segmentation."""
    width, height = record["size"]
    points = np.asarray(record["polygon"])
    if (
        width <= 0
        or height <= 0
        or points.ndim != 2
        or points.shape[1] != 2
        or len(points) < 3
        or not np.isfinite(points).all()
        or np.any(points < 0)
        or np.any(points[:, 0] >= width)
        or np.any(points[:, 1] >= height)
    ):
        raise ValueError("Invalid cable polygon")
    mask = np.zeros((height, width), np.uint8)
    cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 1)
    if cv2.connectedComponents(mask)[0] != 2:
        raise ValueError("Cable polygon must describe one connected object")
    return mask


def load_annotations(path: Path) -> dict[str, dict]:
    data = json.loads(path.read_text())
    if data.get("version") != 1 or data.get("category") != "cable":
        raise ValueError("Expected version 1 cable annotations")
    if str(data.get("quality_status", "")).startswith("rejected"):
        raise ValueError(
            "Withdrawn cable annotations include cast shadow; do not regenerate"
        )
    records = {}
    for record in data["images"]:
        name = record["image"]
        if Path(name).name != name or not name.endswith(".png") or name in records:
            raise ValueError(f"Invalid or duplicate reference name: {name}")
        rasterize_cable(record)
        records[name] = record
    return records


def read_manual_mask(path: Path, record: dict) -> np.ndarray:
    if (
        path.name != record["image"]
        or hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]
    ):
        raise ValueError(f"Source does not match manual annotation: {path.name}")
    image = cv2.imread(str(path))
    if image is None or list(image.shape[1::-1]) != record["size"]:
        raise ValueError(f"Image dimensions do not match annotation: {path.name}")
    return rasterize_cable(record)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "assets/foreground/cable_manual.json",
    )
    args = parser.parse_args()
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError("Output root must be empty")
    paths = sorted((args.data_root / "cable/train/good").glob("*.png"))
    if not paths:
        raise FileNotFoundError("No cable training images")
    records = load_annotations(args.annotations)
    missing = [p.name for p in paths if p.name not in records]
    if missing:
        raise ValueError(
            f"Missing direct manual annotations: {missing}; no automatic fallback"
        )
    # Validate the entire batch before writing any output.
    masks = [(p, read_manual_mask(p, records[p.name])) for p in paths]
    output = args.output_root / "cable/train/foreground"
    output.mkdir(parents=True)
    rows = []
    for path, mask in masks:
        target = output / path.name
        if not cv2.imwrite(str(target), mask * 255):
            raise OSError(f"Cannot write mask: {target}")
        rows.append(
            {
                "category": "cable",
                "image": path.name,
                "source_sha256": records[path.name]["sha256"],
                "mask_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                "method": "direct_manual_polygon",
                "foreground_fraction": float(mask.mean()),
                "visual_review": "pending final overlay review",
            }
        )
    manifest = {
        "scope": "whole visible cable including outer sheath; direct manual polygons only",
        "annotations_sha256": hashlib.sha256(args.annotations.read_bytes()).hexdigest(),
        "images": rows,
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print("Rasterized", len(rows), "directly traced cable polygons")


if __name__ == "__main__":
    main()

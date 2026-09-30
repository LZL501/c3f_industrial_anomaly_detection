"""Save the original VisA Otsu foreground rule as an unrefined PNG baseline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

BRIGHT_FOREGROUND = {
    "candle",
    "cashew",
    "chewinggum",
    "fryum",
    "macaroni1",
    "macaroni2",
    "pipe_fryum",
    "pcb1",
    "pcb2",
    "pcb3",
    "pcb4",
}
CATEGORIES = BRIGHT_FOREGROUND | {"capsules"}


def legacy_foreground(image_rgb: np.ndarray, category: str) -> np.ndarray:
    """Same threshold and polarity as ldm/data/visa2.py, without removed NumPy aliases."""
    if category not in CATEGORIES:
        raise ValueError(f"Unknown VisA category: {category}")
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    _, bright = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return bright if category in BRIGHT_FOREGROUND else 255 - bright


def mask_relative_path(image_path: str, category: str) -> Path:
    relative = Path(image_path)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or len(relative.parts) != 5
        or relative.parts[:3] != (category, "Data", "Images")
    ):
        raise ValueError(f"Unexpected VisA image path: {image_path}")
    return (
        Path(category)
        / "Data"
        / "Foreground"
        / relative.parts[3]
        / relative.with_suffix(".png").name
    )


def generate(row: dict, root: Path, output: Path) -> dict:
    category = row["object"]
    relative = mask_relative_path(row["image"], category)
    source = root / row["image"]
    source_bytes = source.read_bytes()
    image = cv2.imdecode(np.frombuffer(source_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unreadable source: {source}")
    mask = legacy_foreground(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), category)
    target = output / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(target), mask):
        raise OSError(target)
    saved = cv2.imread(str(target), cv2.IMREAD_GRAYSCALE)
    if saved is None or not np.array_equal(saved, mask):
        raise ValueError(f"PNG roundtrip mismatch: {target}")
    foreground = (mask > 0).astype(np.uint8)
    fraction = float(foreground.mean())
    flags = []
    if fraction == 0:
        flags.append("empty")
    if fraction > 0.95:
        flags.append("almost_full_image")
    if 0 < fraction < 0.005:
        flags.append("very_small_foreground")
    _, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    holes = int((hierarchy[0, :, 3] >= 0).sum()) if hierarchy is not None else 0
    return {
        "category": category,
        "image": row["image"],
        "mask": str(relative),
        "method": "original VisA grayscale Otsu with category polarity",
        "resolution": "native source dimensions; no image resizing before threshold",
        "review_status": "automatic_baseline_pending_refinement",
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "mask_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "shape": list(mask.shape),
        "foreground_fraction": fraction,
        "components": cv2.connectedComponents(foreground)[0] - 1,
        "holes": holes,
        "flags": flags,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--report-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.output_root.resolve().is_relative_to(args.root.resolve()):
        parser.error("Keep foreground output outside the original image dataset")
    args.output_root.mkdir(parents=True, exist_ok=False)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    with (args.root / "split_csv/1cls.csv").open() as stream:
        rows = sorted(
            (r for r in csv.DictReader(stream) if r["split"] == "train"),
            key=lambda r: (r["object"], r["image"]),
        )
    assert rows and all(r["label"] == "normal" for r in rows)
    outputs = [mask_relative_path(r["image"], r["object"]) for r in rows]
    assert len(set(outputs)) == len(rows)
    results = []
    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for record in pool.map(
            lambda r: generate(r, args.root, args.output_root), rows
        ):
            results.append(record)
            if len(results) % 500 == 0 or len(results) == len(rows):
                state = {
                    "generated": len(results),
                    "total": len(rows),
                    "elapsed_seconds": round(time.monotonic() - start),
                    "phase": "generated" if len(results) == len(rows) else "generating",
                    "quality": "automatic baseline pending refinement",
                }
                (args.report_dir / "STATUS.json").write_text(
                    json.dumps(state, indent=2) + "\n"
                )
                print(json.dumps(state), flush=True)
    manifest = {
        "method_reference": "ldm/data/visa2.py:MemSegDataset.generate_target_foreground_mask",
        "runtime": {
            "python": platform.python_version(),
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "torch": "not used",
            "cuda": "not used",
        },
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "images": results,
    }
    (args.report_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()

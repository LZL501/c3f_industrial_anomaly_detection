"""Produce an unreviewed VisA baseline with the existing HQ-SAM pipeline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from generate_mvtec_foreground_sam import build_predictor, predict_foreground


def initialize(options: dict) -> None:
    import torch

    global predictor, args
    args = argparse.Namespace(**options)
    if args.devices:
        devices = args.devices.split(",")
        rank = multiprocessing.current_process()._identity[-1] - 1
        args.device = devices[rank % len(devices)]
    if args.device.startswith("cuda"):
        torch.cuda.set_device(args.device)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    predictor = build_predictor(args)
    if args.low_memory:
        from hq_sam_low_memory import enable_low_memory_attention

        enable_low_memory_attention(predictor.model)
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()


def generate(row: dict) -> dict:
    relative = Path(row["image"])
    source = Path(args.root) / relative
    parts = list(relative.parts)
    assert parts[0] == row["object"] and "Images" in parts
    parts[parts.index("Images")] = "Foreground"
    target_relative = Path(*parts).with_suffix(".png")
    output = Path(args.output_root) / target_relative
    record_path = (
        Path(args.report_dir)
        / "records"
        / row["object"]
        / output.with_suffix(".json").name
    )
    if output.exists() and record_path.exists():
        record = json.loads(record_path.read_text())
        if (
            record.get("source_sha256")
            == hashlib.sha256(source.read_bytes()).hexdigest()
            and record.get("mask_sha256")
            == hashlib.sha256(output.read_bytes()).hexdigest()
        ):
            return record
        raise ValueError(f"Existing output differs from its recorded source: {output}")
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unreadable source: {source}")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    start = time.monotonic()
    mask = (
        predict_foreground(predictor, rgb, row["object"], args).astype(np.uint8) * 255
    )
    assert mask.shape == image.shape[:2]
    assert set(np.unique(mask)) <= {0, 255}
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), mask):
        raise OSError(f"Cannot save {output}")
    flags = []
    fraction = float(np.mean(mask > 0))
    if fraction == 0:
        flags.append("empty")
    if fraction > 0.95:
        flags.append("almost_full_image")
    if 0 < fraction < 0.005:
        flags.append("very_small_foreground")
    record = {
        "category": row["object"],
        "image": row["image"],
        "mask": str(target_relative),
        "method": "HQ-SAM ViT-B, existing v2 full-image box pipeline",
        "review_status": "automatic_baseline_pending_refinement",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "mask_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "shape": list(mask.shape),
        "foreground_fraction": fraction,
        "components": cv2.connectedComponents((mask > 0).astype(np.uint8))[0] - 1,
        "flags": flags,
        "seconds": round(time.monotonic() - start, 3),
        "low_memory_attention": args.low_memory,
        "device": args.device,
    }
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--devices", help="Comma-separated worker devices")
    parser.add_argument("--low-memory", action="store_true")
    parser.add_argument("--limit-per-category", type=int)
    args = parser.parse_args()
    args.backend = "sam_hq"
    args.model_type = "vit_b"
    args.strategy = "v2"
    args.prompt_mode = "full_image"
    args.box_margin = 0.03
    args.hq_token_only = True
    rows = list(csv.DictReader((Path(args.root) / "split_csv/1cls.csv").open()))
    rows = sorted(
        (r for r in rows if r["split"] == "train"),
        key=lambda r: (r["object"], r["image"]),
    )
    if args.limit_per_category:
        counts = {}
        selected = []
        for row in rows:
            category = row["object"]
            counts[category] = counts.get(category, 0) + 1
            if counts[category] <= args.limit_per_category:
                selected.append(row)
        rows = selected
    report = Path(args.report_dir)
    report.mkdir(parents=True, exist_ok=True)
    options = vars(args)
    (report / "generation_options.json").write_text(
        json.dumps(options, indent=2) + "\n"
    )
    results = []
    start = time.monotonic()
    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=initialize,
        initargs=(options,),
    ) as pool:
        for result in pool.map(generate, rows, chunksize=1):
            results.append(result)
            if len(results) % 50 == 0 or len(results) == len(rows):
                state = {
                    "generated": len(results),
                    "total": len(rows),
                    "elapsed_seconds": round(time.monotonic() - start),
                    "phase": "complete" if len(results) == len(rows) else "generating",
                    "quality": "automatic baseline; not manually refined",
                }
                (report / "STATUS.json").write_text(json.dumps(state, indent=2) + "\n")
                print(json.dumps(state), flush=True)
    (report / "manifest.json").write_text(
        json.dumps({"images": results}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()

"""Build reviewable transistor masks from manually traced training exemplars.

This is an offline, category-specific annotation tool, not model inference.
Unannotated images receive a three-exemplar optical-flow vote followed by a
narrow GrabCut boundary adjustment. Every output still needs visual review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from generate_mvtec_foreground_sam import transistor_body_geometry

REGIONS = {"body", "left_lead", "center_lead", "right_lead"}


def rasterize(record: dict) -> np.ndarray:
    """Rasterize separate regions as a union, including overlapping lead roots."""
    width, height = record["size"]
    if width <= 0 or height <= 0 or set(record["regions"]) != REGIONS:
        raise ValueError("Expected positive image dimensions and body plus three leads")
    mask = np.zeros((height, width), np.uint8)
    for polygon in record["regions"].values():
        points = np.asarray(polygon)
        if (
            points.ndim != 2
            or points.shape[1] != 2
            or len(points) < 3
            or not np.isfinite(points).all()
            or np.any(points < 0)
            or np.any(points[:, 0] >= width)
            or np.any(points[:, 1] >= height)
        ):
            raise ValueError("Invalid polygon coordinates")
        cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 1)
    if cv2.connectedComponents(mask)[0] != 2:
        raise ValueError("Body and all three leads must form one connected object")
    return mask


def read_reference(image_dir: Path, record: dict) -> tuple[np.ndarray, np.ndarray]:
    name = record["image"]
    if Path(name).name != name or not name.endswith(".png"):
        raise ValueError(f"Expected a PNG basename: {name}")
    path = image_dir / name
    if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
        raise ValueError(f"Source image hash does not match annotation: {name}")
    image = cv2.imread(str(path))
    if image is None or list(image.shape[1::-1]) != record["size"]:
        raise ValueError(f"Source image dimensions do not match annotation: {name}")
    return image, rasterize(record)


class AnnotationTransfer:
    """Transfer visible silhouettes; require references from this dataset version."""

    def __init__(self, image_dir: Path, annotations: dict) -> None:
        if (
            annotations.get("version") != 1
            or annotations.get("category") != "transistor"
        ):
            raise ValueError("Only version 1 transistor annotations are supported")
        self.records = {}
        self.references = []
        for record in annotations["images"]:
            name = record["image"]
            if name in self.records:
                raise ValueError(f"Duplicate reference: {name}")
            image, mask = read_reference(image_dir, record)
            image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
            self.records[name] = record
            if not record.get("override_only", False):
                self.references.append(
                    (name, image, cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), mask)
                )
        if len(self.references) < 4:
            raise ValueError(
                "At least four references are required for leave-one-out checks"
            )

    def predict(self, image: np.ndarray, exclude: str) -> tuple[np.ndarray, dict]:
        height, width = image.shape[:2]
        small = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        # Restrict ranking to the object neighborhood rather than distant board texture.
        crop = small[30:470, 110:400].astype(np.float32)
        references = sorted(
            (ref for ref in self.references if ref[0] != exclude),
            key=lambda ref: float(
                np.mean(np.abs(crop - ref[1][30:470, 110:400].astype(np.float32)))
            ),
        )[:3]
        yy, xx = np.indices(gray.shape, dtype=np.float32)
        flow_model = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        flow_model.setUseSpatialPropagation(True)
        warped = []
        for _, _, source_gray, source_mask in references:
            # Backward flow maps target pixels to exemplar coordinates.
            flow = flow_model.calc(gray, source_gray, None)
            warped.append(
                cv2.remap(
                    source_mask,
                    xx + flow[:, :, 0],
                    yy + flow[:, :, 1],
                    cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                )
                > 0
            )
        votes = np.sum(warped, axis=0)
        prior = (votes >= 2).astype(np.uint8)
        if not prior.any():
            raise ValueError("Reference transfer produced an empty mask")
        seeds = np.zeros(gray.shape, np.uint8)
        seeds[cv2.dilate(prior, np.ones((7, 7), np.uint8)) > 0] = cv2.GC_PR_BGD
        seeds[prior > 0] = cv2.GC_PR_FGD
        # Keep dark terminal cores; color alone confuses them with the copper board.
        seeds[cv2.erode(prior, np.ones((3, 3), np.uint8)) > 0] = cv2.GC_FGD
        cv2.setRNGSeed(1999)
        cv2.grabCut(
            small,
            seeds,
            None,
            np.zeros((1, 65)),
            np.zeros((1, 65)),
            3,
            cv2.GC_INIT_WITH_MASK,
        )
        result = ((seeds == cv2.GC_FGD) | (seeds == cv2.GC_PR_FGD)).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(result)
        if count < 2:
            raise ValueError("Boundary refinement produced an empty mask")
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        discarded = int(result.sum() - stats[largest, cv2.CC_STAT_AREA])
        result = (labels == largest).astype(np.uint8)
        metadata = {
            "method": "reference_transfer",
            "references": [ref[0] for ref in references],
            "vote_disagreement": float(
                ((votes > 0) & (votes < 3)).sum() / (votes > 0).sum()
            ),
            "discarded_pixels_512": discarded,
        }
        return cv2.resize(
            result, (width, height), interpolation=cv2.INTER_NEAREST
        ), metadata


def fit_edge(gray, positions, expected, radius, axis, sign):
    # axis=0: y=f(x), axis=1: x=f(y)
    grad = (
        cv2.Sobel(
            cv2.GaussianBlur(gray, (5, 5), 0),
            cv2.CV_32F,
            int(axis == 1),
            int(axis == 0),
            ksize=3,
        )
        * sign
    )
    lo = max(1, expected - radius)
    hi = min(gray.shape[axis] - 1, expected + radius + 1)
    points = []
    strength = []
    for pos in positions:
        values = grad[lo:hi, pos] if axis == 0 else grad[pos, lo:hi]
        j = int(np.argmax(values))
        points.append((pos, lo + j))
        strength.append(float(values[j]))
    p = np.array(points, float)
    w = np.array(strength)
    keep = w > max(15, np.quantile(w, 0.25))
    p = p[keep]
    w = w[keep]
    if len(p) < 4:
        return np.array([0.0, float(expected)])
    rng = np.random.default_rng(1999)
    best = None
    best_score = -1
    for _ in range(150):
        a, b = p[rng.choice(len(p), 2, replace=False)]
        if abs(a[0] - b[0]) < 20:
            continue
        slope = (a[1] - b[1]) / (a[0] - b[0])
        if abs(slope) > 0.15:
            continue
        intercept = a[1] - slope * a[0]
        selected = np.abs(p[:, 1] - (slope * p[:, 0] + intercept)) < 4
        score = selected.sum()
        if score > best_score:
            best_score = score
            best = selected
    if best is None:
        return np.array([0.0, float(expected)])
    return np.polyfit(p[best, 0], p[best, 1], 1)


def body_polygon(image, baseline):
    x0, x1, t, b = transistor_body_geometry((baseline > 0).astype(np.uint8))
    bw = x1 - x0
    bh = b - t
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    xs = np.arange(x0 + int(0.1 * bw), x1 - int(0.1 * bw), 3)
    ys = np.arange(t + int(0.1 * bh), b - int(0.1 * bh), 3)
    top = fit_edge(gray, xs, t, 50, 0, -1)
    bottom = fit_edge(gray, xs, b, 25, 0, 1)
    left = fit_edge(gray, ys, x0, 25, 1, -1)
    right = fit_edge(gray, ys, x1, 25, 1, 1)

    def intersect(horizontal, vertical):
        a, c = horizontal
        d, e = vertical
        y = (a * e + c) / (1 - a * d)
        x = d * y + e
        return round(x), round(y)

    return np.array(
        [
            intersect(top, left),
            intersect(top, right),
            intersect(bottom, right),
            intersect(bottom, left),
        ],
        np.int32,
    )


def restore_package_body(
    mask: np.ndarray, baseline: np.ndarray, image: np.ndarray
) -> np.ndarray:
    """Fit the four package edges using image gradients near the baseline body.

    A robust line fit rejects board-hole edges. Restrict this shape prior to
    the package so it never fills the background between the three leads.
    """
    if baseline.shape != mask.shape or image.shape[:2] != mask.shape:
        raise ValueError("Baseline mask dimensions do not match source image")
    if transistor_body_geometry((baseline > 0).astype(np.uint8)) is None:
        raise ValueError("Cannot identify package body in baseline mask")
    polygon = body_polygon(image, baseline)
    left, right = polygon[3], polygon[2]
    slope = float(right[1] - left[1]) / max(1, int(right[0] - left[0]))
    yy, xx = np.indices(mask.shape)
    result = mask.copy()
    result[yy <= left[1] + slope * (xx - left[0])] = 0
    cv2.fillConvexPoly(result, polygon, 1)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(result)
    if count < 2:
        raise ValueError("Empty mask after package repair")
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if int(result.sum() - stats[largest, cv2.CC_STAT_AREA]) > 100:
        raise ValueError(
            "Package repair disconnected a significant region; trace this image manually"
        )
    return (labels == largest).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "assets/foreground/transistor_manual.json",
    )
    parser.add_argument("--leave-one-out", action="store_true")
    args = parser.parse_args()
    cv2.setNumThreads(2)
    image_dir = args.data_root / "transistor/train/good"
    output_dir = args.output_root / "transistor/train/foreground"
    paths = sorted(image_dir.glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"No training images in {image_dir}")
    if (
        output_dir.resolve()
        == (args.data_root / "transistor/train/foreground").resolve()
    ):
        raise ValueError(
            "Use a separate output root to preserve existing experiment inputs"
        )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    annotations = json.loads(args.annotations.read_text())
    transfer = AnnotationTransfer(image_dir, annotations)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for path in paths:
        if args.leave_one_out and path.name not in transfer.records:
            continue
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Cannot read source image: {path}")
        if path.name in transfer.records and not args.leave_one_out:
            mask = rasterize(transfer.records[path.name])
            metadata = {"method": "manual_polygon"}
        else:
            mask, metadata = transfer.predict(image, exclude=path.name)
            baseline_path = args.data_root / "transistor/train/foreground" / path.name
            baseline = cv2.imread(str(baseline_path), cv2.IMREAD_GRAYSCALE)
            if baseline is None:
                raise FileNotFoundError(baseline_path)
            mask = restore_package_body(mask, baseline, image)
            metadata["baseline_sha256"] = hashlib.sha256(
                baseline_path.read_bytes()
            ).hexdigest()
        output_path = output_dir / path.name
        if not cv2.imwrite(str(output_path), mask * 255):
            raise OSError(f"Cannot write mask: {output_path}")
        metadata.update(
            image=path.name,
            source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            mask_sha256=hashlib.sha256(output_path.read_bytes()).hexdigest(),
            foreground_fraction=float(mask.mean()),
            visual_review="pending",
        )
        records.append(metadata)
        if len(records) % 25 == 0:
            print(f"Wrote {len(records)} masks", flush=True)
    manifest = {
        "annotations_sha256": hashlib.sha256(args.annotations.read_bytes()).hexdigest(),
        "leave_one_out": args.leave_one_out,
        "images": records,
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(
        f"Wrote {len(records)} masks; visual review is still required: {args.output_root}"
    )


if __name__ == "__main__":
    main()

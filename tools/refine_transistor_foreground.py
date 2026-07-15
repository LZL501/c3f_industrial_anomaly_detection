from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from generate_mvtec_foreground_sam import complete_transistor_leads, overlay_mask


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    mask_root = Path(args.mask_root) if args.mask_root else root
    input_dir = mask_root / "transistor" / "train" / "foreground"
    image_dir = root / "transistor" / "train" / "good"
    output_dir = Path(args.output_root) / "transistor" / "train" / "foreground"
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = Path(args.preview_dir) if args.preview_dir else None
    if preview_dir is not None:
        preview_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(image_dir.glob("*.png"))
    if args.limit is not None:
        paths = paths[: args.limit]
    for path in paths:
        image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(input_dir / path.name), cv2.IMREAD_GRAYSCALE)
        if image_bgr is None:
            raise FileNotFoundError(path)
        if mask is None:
            raise FileNotFoundError(input_dir / path.name)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        refined = complete_transistor_leads(mask, image_rgb)
        cv2.imwrite(str(output_dir / path.name), refined * 255)
        if preview_dir is not None:
            cv2.imwrite(str(preview_dir / path.name), overlay_mask(image_bgr, refined))

    print(f"transistor: wrote {len(paths)} refined masks to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Complete transistor leads in existing foreground masks."
    )
    parser.add_argument(
        "--root", required=True, help="MVTec AD root containing the source images."
    )
    parser.add_argument(
        "--mask-root", default=None, help="Input mask root. Defaults to --root."
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Separate output root; existing masks are not overwritten.",
    )
    parser.add_argument("--preview-dir", default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main()

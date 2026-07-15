from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from c3f.data.anomaly import SyntheticAnomalyGenerator
from c3f.data.datasets import _mvtec_foreground, _read_rgb


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for category in args.categories:
        write_category_preview(args, category, output_dir / f"{category}.jpg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview training-time MVTec pseudo anomalies."
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--texture-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--categories", nargs="+", default=["cable", "screw", "transistor"]
    )
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1999)
    return parser.parse_args()


def write_category_preview(
    args: argparse.Namespace, category: str, output_path: Path
) -> None:
    resize = (args.image_size, args.image_size)
    image_paths = sorted(
        (Path(args.data_root) / category / "train" / "good").glob("*.png")
    )
    if not image_paths:
        raise FileNotFoundError(f"No training images found for {category}")

    rng = random.Random(f"{args.seed}:{category}")
    selected = rng.sample(image_paths, min(args.samples, len(image_paths)))
    generator = SyntheticAnomalyGenerator(resize=resize, texture_root=args.texture_root)

    columns = ("original", "foreground", "pseudo anomaly", "pseudo mask")
    tile = args.image_size
    header_h = 34
    label_h = 28
    sheet = Image.new(
        "RGB",
        (tile * len(columns), header_h + (tile + label_h) * len(selected)),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    for column, title in enumerate(columns):
        draw.text((column * tile + 8, 10), title, fill=(15, 15, 15))

    for row, image_path in enumerate(selected):
        image = _read_rgb(image_path, resize)
        foreground = _mvtec_foreground(
            image_path, image, resize, category, require=True
        ).astype(np.uint8)
        pseudo, mask = generator.generate(image, foreground)
        outside = int(np.logical_and(mask > 0, foreground == 0).sum())
        area = 100.0 * float(mask.mean())

        cells = (
            image,
            overlay(image, foreground, (0, 180, 0)),
            pseudo,
            overlay(pseudo, mask.astype(np.uint8), (235, 30, 30)),
        )
        y = header_h + row * (tile + label_h)
        for column, cell in enumerate(cells):
            sheet.paste(Image.fromarray(cell), (column * tile, y))
        draw.text(
            (8, y + tile + 7),
            f"{image_path.name}  mask={area:.1f}%  outside_fg={outside}",
            fill=(20, 20, 20),
        )

    sheet.save(output_path, quality=94)
    print(output_path)


def overlay(
    image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]
) -> np.ndarray:
    result = image.copy()
    active = mask > 0
    tint = np.empty_like(result)
    tint[:, :] = color
    result[active] = (0.58 * result[active] + 0.42 * tint[active]).astype(np.uint8)
    return result


if __name__ == "__main__":
    main()

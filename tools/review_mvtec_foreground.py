from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


MVTEC_CATEGORIES = (
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
    "transistor",
    "wood",
    "zipper",
)
FULL_IMAGE_CATEGORIES = {"carpet", "grid", "leather", "tile", "wood"}


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    mask_root = Path(args.mask_root)
    review_dir = Path(args.review_dir)
    review_dir.mkdir(parents=True, exist_ok=True)

    records = {
        category: collect_category(
            data_root,
            mask_root,
            category,
            args.overview_count if args.sample_only else None,
        )
        for category in args.categories
    }
    if args.sample_only:
        make_overview(
            records, "overlay", review_dir / "overview_overlay.jpg", args.overview_count
        )
        make_overview(
            records, "mask", review_dir / "overview_mask.jpg", args.overview_count
        )
        print(review_dir)
        return
    write_stats(review_dir / "stats.tsv", records)
    make_overview(
        records, "overlay", review_dir / "overview_overlay.jpg", args.overview_count
    )
    make_overview(
        records, "mask", review_dir / "overview_mask.jpg", args.overview_count
    )
    make_outliers(records, "overlay", review_dir / "outliers_overlay.jpg")
    make_outliers(records, "mask", review_dir / "outliers_mask.jpg")
    if args.category_sheets:
        for category, rows in records.items():
            make_category_sheet(
                rows, "overlay", review_dir / f"{category}_all_overlay.jpg"
            )
            make_category_sheet(rows, "mask", review_dir / f"{category}_all_mask.jpg")
    print(review_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Review generated MVTec foreground masks."
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--mask-root", required=True)
    parser.add_argument("--review-dir", required=True)
    parser.add_argument("--overview-count", type=int, default=12)
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=MVTEC_CATEGORIES,
        default=list(MVTEC_CATEGORIES),
    )
    parser.add_argument("--category-sheets", action="store_true")
    parser.add_argument("--sample-only", action="store_true")
    return parser.parse_args()


def collect_category(
    data_root: Path, mask_root: Path, category: str, limit: int | None = None
) -> list[dict]:
    records = []
    image_paths = sorted((data_root / category / "train" / "good").glob("*.png"))
    if limit is not None:
        image_paths = image_paths[:limit]
    for image_path in image_paths:
        mask_path = mask_root / category / "train" / "foreground" / image_path.name
        mask_image = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask_image is None:
            raise FileNotFoundError(mask_path)
        mask = (mask_image > 0).astype(np.uint8)
        area = float(mask.mean())
        border = float(np.r_[mask[0, :], mask[-1, :], mask[:, 0], mask[:, -1]].mean())
        components = max(0, cv2.connectedComponents(mask, connectivity=8)[0] - 1)
        records.append(
            {
                "image": image_path,
                "mask": mask_path,
                "area": area,
                "border": border,
                "components": int(components),
            }
        )
    return records


def write_stats(path: Path, records: dict[str, list[dict]]) -> None:
    with path.open("w") as handle:
        handle.write(
            "category\tcount\tarea_mean\tarea_min\tarea_p05\tarea_median\t"
            "area_p95\tarea_max\tborder_mean\tcomponents_mean\tzero_count\tfull_count\n"
        )
        for category, rows in records.items():
            areas = np.array([row["area"] for row in rows], dtype=float)
            borders = np.array([row["border"] for row in rows], dtype=float)
            components = np.array([row["components"] for row in rows], dtype=float)
            handle.write(
                f"{category}\t{len(rows)}\t{areas.mean():.6f}\t{areas.min():.6f}\t"
                f"{np.quantile(areas, 0.05):.6f}\t{np.median(areas):.6f}\t"
                f"{np.quantile(areas, 0.95):.6f}\t{areas.max():.6f}\t"
                f"{borders.mean():.6f}\t{components.mean():.3f}\t"
                f"{int((areas == 0).sum())}\t{int((areas == 1).sum())}\n"
            )


def read_pair(record: dict) -> tuple[Image.Image, Image.Image]:
    image_bgr = cv2.imread(str(record["image"]), cv2.IMREAD_COLOR)
    mask = (cv2.imread(str(record["mask"]), cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)
    color = np.zeros_like(image_bgr)
    color[:, :, 2] = 255
    overlay = np.where(
        mask[:, :, None] > 0,
        (0.55 * image_bgr + 0.45 * color).astype(np.uint8),
        image_bgr,
    )
    overlay_rgb = Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    mask_rgb = Image.fromarray((mask * 255).astype(np.uint8)).convert("RGB")
    return overlay_rgb, mask_rgb


def make_overview(
    records: dict[str, list[dict]], kind: str, output_path: Path, sample_count: int
) -> None:
    thumb = 116
    label_h = 22
    cols = sample_count
    categories = list(records)
    sheet = Image.new(
        "RGB", (cols * thumb, len(categories) * (thumb + label_h)), "white"
    )
    draw = ImageDraw.Draw(sheet)
    for row_index, category in enumerate(categories):
        y0 = row_index * (thumb + label_h)
        draw.rectangle([0, y0, cols * thumb, y0 + label_h], fill=(236, 236, 236))
        draw.text((5, y0 + 5), category, fill=(0, 0, 0))
        for col_index, record in enumerate(records[category][:sample_count]):
            overlay, mask = read_pair(record)
            image = overlay if kind == "overlay" else mask
            resample = Image.LANCZOS if kind == "overlay" else Image.NEAREST
            image.thumbnail((thumb - 4, thumb - 24), resample)
            x = col_index * thumb + 2
            y = y0 + label_h + 18
            sheet.paste(image, (x, y))
            draw.text((x, y0 + label_h + 2), record["image"].name, fill=(35, 35, 35))
    sheet.save(output_path, quality=92)


def pick_outliers(rows: list[dict]) -> list[dict]:
    selected = []
    selected_names = set()

    def add(items: list[dict]) -> None:
        for item in items:
            name = item["image"].name
            if name not in selected_names:
                selected.append(item)
                selected_names.add(name)

    add(sorted(rows, key=lambda row: row["area"])[:2])
    add(sorted(rows, key=lambda row: row["area"], reverse=True)[:2])
    add(sorted(rows, key=lambda row: row["border"], reverse=True)[:2])
    add(sorted(rows, key=lambda row: row["components"], reverse=True)[:2])
    if len(selected) < 8 and rows:
        add([rows[len(rows) // 2]])
    return selected[:8]


def make_outliers(records: dict[str, list[dict]], kind: str, output_path: Path) -> None:
    thumb = 180
    label_h = 42
    cols = 8
    categories = list(records)
    sheet = Image.new(
        "RGB", (cols * thumb, len(categories) * (thumb + label_h)), "white"
    )
    draw = ImageDraw.Draw(sheet)
    for row_index, category in enumerate(categories):
        y0 = row_index * (thumb + label_h)
        draw.rectangle([0, y0, cols * thumb, y0 + label_h], fill=(236, 236, 236))
        draw.text((5, y0 + 5), category, fill=(0, 0, 0))
        for col_index, record in enumerate(pick_outliers(records[category])):
            overlay, mask = read_pair(record)
            image = overlay if kind == "overlay" else mask
            resample = Image.LANCZOS if kind == "overlay" else Image.NEAREST
            image.thumbnail((thumb - 4, thumb - 44), resample)
            x = col_index * thumb + 2
            y = y0 + label_h + 22
            sheet.paste(image, (x, y))
            label = (
                f"{record['image'].name} a={record['area']:.2f} "
                f"b={record['border']:.2f} c={record['components']}"
            )
            draw.text((x, y0 + label_h + 2), label, fill=(20, 20, 20))
    sheet.save(output_path, quality=92)


def make_category_sheet(rows: list[dict], kind: str, output_path: Path) -> None:
    thumb = 160
    label_h = 34
    cols = 10
    row_h = thumb + label_h
    sheet_rows = (len(rows) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb, sheet_rows * row_h), "white")
    draw = ImageDraw.Draw(sheet)
    for index, record in enumerate(rows):
        overlay, mask = read_pair(record)
        image = overlay if kind == "overlay" else mask
        resample = Image.LANCZOS if kind == "overlay" else Image.NEAREST
        image.thumbnail((thumb - 4, thumb - 4), resample)
        x = (index % cols) * thumb + 2
        y = (index // cols) * row_h + label_h
        sheet.paste(image, (x, y))
        label = (
            f"{record['image'].stem} a={record['area']:.2f} c={record['components']}"
        )
        draw.text((x, y - label_h + 8), label, fill=(20, 20, 20))
    sheet.save(output_path, quality=92)


if __name__ == "__main__":
    main()

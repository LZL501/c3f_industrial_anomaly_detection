from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path

import cv2
import numpy as np

try:
    import torch
except ModuleNotFoundError:
    torch = None


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
TEXTURE_CATEGORIES = {"carpet", "grid", "leather", "tile", "wood"}


def main() -> None:
    args = parse_args()
    categories = parse_categories(args.categories)
    full_image_categories = set(parse_categories(args.full_image_categories))
    predictor = None
    if any(
        needs_predictor(category, args, full_image_categories)
        for category in categories
    ):
        predictor = build_predictor(args)

    for category in categories:
        image_dir = Path(args.root) / category / "train" / "good"
        output_root = Path(args.output_root) if args.output_root else Path(args.root)
        output_dir = output_root / category / "train" / "foreground"
        preview_dir = Path(args.preview_dir) / category if args.preview_dir else None
        output_dir.mkdir(parents=True, exist_ok=True)
        if preview_dir is not None:
            preview_dir.mkdir(parents=True, exist_ok=True)

        paths = sorted(image_dir.glob("*.png"))
        if args.limit is not None:
            paths = paths[: args.limit]
        for index, path in enumerate(paths):
            image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise FileNotFoundError(path)
            if category in full_image_categories:
                mask = np.ones(image_bgr.shape[:2], dtype=np.uint8)
            else:
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                if needs_predictor(category, args, full_image_categories):
                    assert predictor is not None
                mask = predict_foreground(predictor, image_rgb, category, args)

            output_path = output_dir / path.name
            cv2.imwrite(str(output_path), mask * 255)
            if preview_dir is not None and index < args.preview_count:
                preview_path = preview_dir / path.name
                cv2.imwrite(str(preview_path), overlay_mask(image_bgr, mask))

        print(f"{category}: wrote {len(paths)} foreground masks to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate MVTec foreground masks with SAM."
    )
    parser.add_argument("--root", required=True, help="MVTec AD root directory.")
    parser.add_argument("--checkpoint", required=True, help="SAM checkpoint path.")
    parser.add_argument("--backend", default="sam2", choices=("sam2", "sam1", "sam_hq"))
    parser.add_argument("--sam2-config", default="configs/sam2.1/sam2.1_hiera_b+.yaml")
    parser.add_argument(
        "--model-type", default="vit_b", choices=("vit_b", "vit_l", "vit_h")
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch is not None and torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--categories", default="all", help="Comma-separated categories or 'all'."
    )
    parser.add_argument(
        "--output-root", default=None, help="Output root. Defaults to --root."
    )
    parser.add_argument(
        "--full-image-categories",
        default=",".join(sorted(TEXTURE_CATEGORIES)),
        help="Categories whose foreground is the whole image.",
    )
    parser.add_argument(
        "--box-margin", type=float, default=0.03, help="Prompt box inset ratio."
    )
    parser.add_argument(
        "--prompt-mode",
        default="heuristic_box",
        choices=("heuristic_box", "full_image"),
    )
    parser.add_argument("--strategy", default="default", choices=("default", "v2"))
    parser.add_argument("--hq-token-only", action="store_true", default=True)
    parser.add_argument(
        "--no-hq-token-only", dest="hq_token_only", action="store_false"
    )
    parser.add_argument("--min-area-ratio", type=float, default=0.01)
    parser.add_argument("--max-area-ratio", type=float, default=0.95)
    parser.add_argument("--preview-dir", default=None)
    parser.add_argument("--preview-count", type=int, default=8)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N images per category.",
    )
    return parser.parse_args()


def parse_categories(value: str) -> tuple[str, ...]:
    if value == "all":
        return MVTEC_CATEGORIES
    categories = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = sorted(set(categories) - set(MVTEC_CATEGORIES))
    if unknown:
        raise ValueError(f"Unknown MVTec categories: {unknown}")
    return categories


def build_predictor(args: argparse.Namespace):
    require_torch()
    if args.backend == "sam2":
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        model = build_sam2(args.sam2_config, args.checkpoint, device=args.device)
        return SAM2ImagePredictor(model)

    from segment_anything import SamPredictor, sam_model_registry

    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    sam.to(device=args.device)
    sam.eval()
    return SamPredictor(sam)


def needs_predictor(
    category: str, args: argparse.Namespace, full_image_categories: set[str]
) -> bool:
    if category in full_image_categories:
        return False
    if args.strategy == "v2" and args.backend == "sam_hq":
        return True
    if args.strategy == "v2" and category in {
        "bottle",
        "cable",
        "capsule",
        "screw",
        "transistor",
        "zipper",
    }:
        return False
    return True


def predict_foreground(
    predictor,
    image_rgb: np.ndarray,
    category: str,
    args: argparse.Namespace,
) -> np.ndarray:
    if args.strategy == "v2":
        if args.backend == "sam_hq":
            return foreground_from_hq_sam(predictor, image_rgb, category, args)
        if category == "bottle":
            return bottle_foreground_from_image(image_rgb)
        if category == "cable":
            return cable_foreground_from_color(image_rgb)
        if category == "capsule":
            return capsule_foreground_from_image(image_rgb)
        if category == "screw":
            return screw_foreground_from_image(image_rgb)
        if category == "transistor":
            return transistor_foreground_from_color(image_rgb)
        if category == "zipper":
            return zipper_foreground_from_image(image_rgb)

    height, width = image_rgb.shape[:2]
    if args.prompt_mode == "heuristic_box":
        prompt_mask = heuristic_foreground_from_image(image_rgb, category)
        expected_area = float(prompt_mask.mean())
        box = mask_to_box(prompt_mask, width, height, args.box_margin)
    else:
        expected_area = None
        margin_x = int(round(width * args.box_margin))
        margin_y = int(round(height * args.box_margin))
        box = np.array(
            [margin_x, margin_y, width - 1 - margin_x, height - 1 - margin_y],
            dtype=np.float32,
        )

    torch_module = require_torch()
    with torch_module.inference_mode(), autocast_context(args.device):
        predictor.set_image(image_rgb)
        masks, scores, _ = predictor.predict(box=box, multimask_output=True)
    masks = to_numpy(masks)
    scores = to_numpy(scores)
    mask = choose_mask(masks, scores, args, expected_area)
    return refine_foreground_mask(category, mask)


def cable_foreground_from_color(image_rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    mask = (mask > 0).astype(np.uint8)
    h, w = mask.shape
    border = np.zeros_like(mask, dtype=bool)
    pad_y = max(1, int(round(h * 0.08)))
    pad_x = max(1, int(round(w * 0.08)))
    border[:pad_y, :] = True
    border[-pad_y:, :] = True
    border[:, :pad_x] = True
    border[:, -pad_x:] = True
    center = np.zeros_like(mask, dtype=bool)
    center[int(h * 0.25) : int(h * 0.75), int(w * 0.25) : int(w * 0.75)] = True
    if float(mask[center].mean()) < float(mask[border].mean()):
        mask = 1 - mask
    mask = remove_border_components(mask)
    mask = keep_components(mask, min_area_ratio=0.02, max_components=1)
    kernel = np.ones((13, 13), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = fill_holes(mask)
    mask = keep_components(mask, min_area_ratio=0.01, max_components=1)
    return cleanup_mask(mask)


def bottle_foreground_from_image(image_rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    mask = (mask > 0).astype(np.uint8)
    mask = remove_border_components(mask)
    mask = keep_center_components(
        mask,
        min_area_ratio=0.01,
        max_components=1,
        x_range=(0.12, 0.88),
        y_range=(0.08, 0.92),
    )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8), iterations=2
    )
    mask = fill_holes(mask)
    return cleanup_mask(mask)


def capsule_foreground_from_image(image_rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    image = image_rgb.astype(np.int16)
    red = image[:, :, 0]
    green = image[:, :, 1]
    blue = image[:, :, 2]
    red_region = (saturation > 36) & (red > green + 8) & (red > blue + 8)
    dark_region = gray < 115
    mask = (red_region | dark_region).astype(np.uint8)
    mask = keep_center_components(
        mask,
        min_area_ratio=0.004,
        max_components=3,
        x_range=(0.02, 0.98),
        y_range=(0.12, 0.75),
    )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, np.ones((29, 13), np.uint8), iterations=2
    )
    mask = keep_components(mask, min_area_ratio=0.006, max_components=1)
    mask = fill_holes(mask)
    return cleanup_mask(mask)


def screw_foreground_from_image(image_rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    image = image_rgb.astype(np.int16)
    red = image[:, :, 0]
    green = image[:, :, 1]
    blue = image[:, :, 2]
    red_region = (saturation > 30) & (red > green + 6) & (red > blue + 6)
    dark_region = gray < 105
    mask = (red_region | dark_region).astype(np.uint8)
    mask = keep_center_components(
        mask,
        min_area_ratio=0.0015,
        max_components=4,
        x_range=(0.02, 0.98),
        y_range=(0.02, 0.98),
    )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=1
    )
    mask = keep_components(mask, min_area_ratio=0.002, max_components=1)
    mask = fill_holes(mask)
    return cleanup_mask(mask)


def foreground_from_hq_sam(
    predictor,
    image_rgb: np.ndarray,
    category: str,
    args: argparse.Namespace,
) -> np.ndarray:
    box, points, labels = hq_sam_prompt(image_rgb, category, args)
    torch_module = require_torch()
    with torch_module.inference_mode():
        predictor.set_image(image_rgb)
        masks, _, _ = predictor.predict(
            point_coords=points,
            point_labels=labels,
            box=box,
            multimask_output=False,
            hq_token_only=args.hq_token_only,
        )
    mask = to_numpy(masks)[0].astype(np.uint8)
    return refine_hq_sam_foreground(category, mask, image_rgb)


def hq_sam_prompt(
    image_rgb: np.ndarray,
    category: str,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    if category == "transistor":
        points, labels = transistor_prompt_points(image_rgb)
        return transistor_prompt_box(image_rgb), points, labels
    if category == "screw":
        return screw_prompt_box(image_rgb), None, None
    if category == "zipper":
        height, width = image_rgb.shape[:2]
        prompt_mask = hq_sam_prompt_mask(image_rgb, category)
        box = mask_to_box(prompt_mask, width, height, margin_ratio=0.05)
        points, labels = zipper_prompt_points(image_rgb, box)
        return box, points, labels

    height, width = image_rgb.shape[:2]
    if args.prompt_mode == "full_image":
        margin_x = int(round(width * args.box_margin))
        margin_y = int(round(height * args.box_margin))
        box = np.asarray(
            [margin_x, margin_y, width - 1 - margin_x, height - 1 - margin_y],
            dtype=np.float32,
        )
        return box, None, None

    prompt_mask = hq_sam_prompt_mask(image_rgb, category)
    margin = 0.05 if category in {"cable", "zipper", "toothbrush"} else 0.04
    return mask_to_box(prompt_mask, width, height, margin), None, None


def hq_sam_prompt_mask(image_rgb: np.ndarray, category: str) -> np.ndarray:
    if category == "bottle":
        return bottle_foreground_from_image(image_rgb)
    if category == "cable":
        return cable_foreground_from_color(image_rgb)
    if category == "capsule":
        return capsule_foreground_from_image(image_rgb)
    if category == "zipper":
        return zipper_foreground_from_image(image_rgb)
    return heuristic_foreground_from_image(image_rgb, category)


def refine_hq_sam_foreground(
    category: str, mask: np.ndarray, image_rgb: np.ndarray | None = None
) -> np.ndarray:
    mask = cleanup_mask(mask)
    if category == "screw":
        mask = keep_components(mask, min_area_ratio=0.002, max_components=1)
        mask = fill_holes(mask)
        return cleanup_mask(mask)
    if category == "transistor":
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, np.ones((5, 3), np.uint8), iterations=1
        )
        return refine_transistor_hq_mask(mask, image_rgb)
    if category in {"bottle", "capsule", "hazelnut", "metal_nut", "pill", "toothbrush"}:
        mask = keep_center_components(mask, min_area_ratio=0.003, max_components=1)
        mask = fill_holes(mask)
        return cleanup_mask(mask)
    if category == "zipper":
        mask = fill_holes(mask)
        return filled_box_from_mask(mask)
    if category == "cable":
        if image_rgb is not None:
            return cable_disk_foreground(image_rgb)
        return keep_components(mask, min_area_ratio=0.002, max_components=2)
    return mask


def cable_disk_foreground(image_rgb: np.ndarray) -> np.ndarray:
    height, width = image_rgb.shape[:2]
    detection_size = 256
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(
        gray, (detection_size, detection_size), interpolation=cv2.INTER_AREA
    )
    gray = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=64,
        param1=100,
        param2=25,
        minRadius=96,
        maxRadius=113,
    )
    if circles is None:
        center_x = width / 2.0
        center_y = height / 2.0
    else:
        target_radius = 0.415 * detection_size
        image_center = detection_size / 2.0
        candidates = circles[0]
        circle = min(
            candidates,
            key=lambda value: abs(float(value[2]) - target_radius)
            + 0.25
            * np.hypot(float(value[0]) - image_center, float(value[1]) - image_center),
        )
        detected_x = float(circle[0]) * width / detection_size
        detected_y = float(circle[1]) * height / detection_size
        center_x = 0.65 * detected_x + 0.35 * width / 2.0
        center_y = 0.65 * detected_y + 0.35 * height / 2.0
    radius = 0.455 * min(height, width)

    foreground = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(
        foreground,
        (int(round(center_x)), int(round(center_y))),
        int(round(radius)),
        1,
        thickness=-1,
    )
    return foreground


def screw_prompt_box(image_rgb: np.ndarray) -> np.ndarray:
    height, width = image_rgb.shape[:2]
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8), iterations=1
    )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1
    )
    mask = keep_components(
        (mask > 0).astype(np.uint8), min_area_ratio=0.01, max_components=1
    )
    return mask_to_box(mask, width, height, margin_ratio=0.06)


def screw_foreground_from_hq_sam(
    predictor, image_rgb: np.ndarray, args: argparse.Namespace
) -> np.ndarray:
    return foreground_from_hq_sam(predictor, image_rgb, "screw", args)


def transistor_prompt_box(image_rgb: np.ndarray) -> np.ndarray:
    height, width = image_rgb.shape[:2]
    return np.asarray(
        [
            int(0.18 * width),
            int(0.03 * height),
            int(0.82 * width),
            int(0.92 * height),
        ],
        dtype=np.float32,
    )


def transistor_prompt_points(image_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height, width = image_rgb.shape[:2]
    positives = [
        (0.50 * width, 0.30 * height),
        (0.38 * width, 0.62 * height),
        (0.50 * width, 0.62 * height),
        (0.62 * width, 0.62 * height),
    ]
    negatives = [
        (0.22 * width, 0.23 * height),
        (0.22 * width, 0.45 * height),
        (0.22 * width, 0.74 * height),
        (0.78 * width, 0.23 * height),
        (0.78 * width, 0.45 * height),
        (0.78 * width, 0.74 * height),
    ]
    points = np.asarray(positives + negatives, dtype=np.float32)
    labels = np.asarray([1] * len(positives) + [0] * len(negatives), dtype=np.int32)
    return points, labels


def refine_transistor_hq_mask(
    mask: np.ndarray, image_rgb: np.ndarray | None = None
) -> np.ndarray:
    mask = cleanup_mask(mask)
    height, width = mask.shape[:2]
    yy, xx = np.mgrid[:height, :width]
    body_region = (
        (xx >= 0.27 * width)
        & (xx <= 0.73 * width)
        & (yy >= 0.03 * height)
        & (yy <= 0.58 * height)
    )
    lead_region = (
        (xx >= 0.23 * width)
        & (xx <= 0.77 * width)
        & (yy >= 0.42 * height)
        & (yy <= 0.88 * height)
    )
    mask = (mask > 0).astype(np.uint8)
    mask[~(body_region | lead_region)] = 0
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, np.ones((5, 3), np.uint8), iterations=1
    )
    mask = keep_transistor_components(mask)
    mask = cleanup_mask(mask)
    if image_rgb is not None:
        color_mask = transistor_foreground_from_color(image_rgb).astype(np.uint8)
        constrained = keep_components(
            (mask & color_mask).astype(np.uint8),
            min_area_ratio=0.0008,
            max_components=5,
        )
        if float(constrained.mean()) >= 0.12:
            return complete_transistor_leads(constrained, image_rgb)
        return complete_transistor_leads(mask, image_rgb)
    return mask


def complete_transistor_leads(mask: np.ndarray, image_rgb: np.ndarray) -> np.ndarray:
    """Complete the three leads using the package body and terminal holes as anchors."""
    mask = (mask > 0).astype(np.uint8)
    height, width = mask.shape
    geometry = transistor_body_geometry(mask)
    if geometry is None:
        return mask
    body_x0, body_x1, body_top, body_bottom = geometry
    body_width = body_x1 - body_x0 + 1
    body_center = 0.5 * (body_x0 + body_x1)

    roots = detect_transistor_lead_roots(
        image_rgb,
        body_x0,
        body_x1,
        body_bottom,
    )
    terminals = select_transistor_terminal_holes(
        detect_transistor_board_holes(image_rgb),
        body_center,
        body_bottom,
        width,
        height,
    )

    lead_mask = np.zeros_like(mask)
    thickness = max(9, int(round(min(height, width) * 0.030)))
    upper_thickness = max(thickness, int(round(min(height, width) * 0.042)))
    start_y = int(round(body_bottom - 0.01 * height))
    bend_y1 = int(round(body_bottom + 0.08 * height))
    bend_y2 = int(round(body_bottom + 0.20 * height))
    for index, (root_x, (terminal_x, terminal_y)) in enumerate(zip(roots, terminals)):
        terminal_y = max(bend_y2 + 1, terminal_y)
        if index == 1:
            points = np.asarray(
                [(root_x, start_y), (root_x, bend_y2), (terminal_x, terminal_y)],
                dtype=np.int32,
            )
        else:
            direction = -1 if index == 0 else 1
            shoulder_x = root_x + direction * 0.02 * body_width
            points = np.asarray(
                [
                    (root_x, start_y),
                    (shoulder_x, bend_y1),
                    (terminal_x, bend_y2),
                    (terminal_x, terminal_y),
                ],
                dtype=np.int32,
            )
        upper_points = points[:-1]
        lower_points = points[-2:]
        cv2.polylines(
            lead_mask,
            [upper_points],
            isClosed=False,
            color=1,
            thickness=upper_thickness,
            lineType=cv2.LINE_AA,
        )
        cv2.polylines(
            lead_mask,
            [lower_points],
            isClosed=False,
            color=1,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )
        for point in upper_points:
            cv2.circle(lead_mask, tuple(point), upper_thickness // 2, 1, thickness=-1)
        cv2.circle(lead_mask, tuple(points[-1]), thickness // 2, 1, thickness=-1)

    body_mask = mask.copy()
    body_pad_x = int(round(0.02 * width))
    body_mask[:body_top, :] = 0
    body_mask[body_bottom + 1 :, :] = 0
    body_mask[:, : max(0, body_x0 - body_pad_x)] = 0
    body_mask[:, min(width, body_x1 + body_pad_x + 1) :] = 0

    lead_support_width = max(9, int(round(min(height, width) * 0.035)))
    lead_support = cv2.dilate(
        lead_mask,
        np.ones((lead_support_width, lead_support_width), np.uint8),
        iterations=1,
    )
    metal_leads = transistor_metal_pixels(image_rgb) & lead_support
    metal_leads[: max(0, body_bottom - int(round(0.01 * height))), :] = 0
    metal_leads = cv2.morphologyEx(
        metal_leads.astype(np.uint8),
        cv2.MORPH_OPEN,
        np.ones((3, 3), np.uint8),
        iterations=1,
    )
    metal_leads = cv2.morphologyEx(
        metal_leads,
        cv2.MORPH_CLOSE,
        np.ones((5, 3), np.uint8),
        iterations=1,
    )

    completed = np.maximum(body_mask, lead_mask)
    completed = np.maximum(completed, metal_leads)
    completed = cv2.morphologyEx(
        completed, cv2.MORPH_CLOSE, np.ones((11, 7), np.uint8), iterations=1
    )
    return keep_components(completed, min_area_ratio=0.001, max_components=1)


def transistor_body_geometry(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    height, width = mask.shape
    row_widths = np.count_nonzero(mask, axis=1)
    body_rows = np.flatnonzero(row_widths >= 0.25 * width)
    body_rows = body_rows[(body_rows >= 0.05 * height) & (body_rows <= 0.68 * height)]
    if body_rows.size == 0:
        return None

    body_top = int(body_rows.min())
    body_bottom = int(body_rows.max())
    stable_rows = body_rows[
        body_rows <= body_bottom - max(2, int(round(0.02 * height)))
    ]
    if stable_rows.size == 0:
        stable_rows = body_rows
    left_edges = []
    right_edges = []
    for y in stable_rows:
        xs = np.flatnonzero(mask[y] > 0)
        if xs.size:
            left_edges.append(int(xs.min()))
            right_edges.append(int(xs.max()))
    if not left_edges:
        return None
    body_x0 = int(round(float(np.median(left_edges))))
    body_x1 = int(round(float(np.median(right_edges))))
    if body_x1 - body_x0 < 0.20 * width:
        return None
    return body_x0, body_x1, body_top, body_bottom


def detect_transistor_lead_roots(
    image_rgb: np.ndarray,
    body_x0: int,
    body_x1: int,
    body_bottom: int,
) -> tuple[int, int, int]:
    height, width = image_rgb.shape[:2]
    body_width = body_x1 - body_x0 + 1
    metal = transistor_metal_pixels(image_rgb) > 0
    y0 = max(0, body_bottom - int(round(0.01 * height)))
    y1 = min(height, body_bottom + int(round(0.13 * height)))
    scores = np.count_nonzero(metal[y0:y1], axis=0).astype(np.float32)
    smooth_width = max(5, int(round(0.025 * width)))
    scores = cv2.blur(scores.reshape(1, -1), (smooth_width, 1)).ravel()

    roots = []
    for fraction in (0.24, 0.50, 0.76):
        expected = body_x0 + fraction * body_width
        radius = int(round(0.09 * body_width))
        x0 = max(0, int(round(expected)) - radius)
        x1 = min(width, int(round(expected)) + radius + 1)
        if x1 <= x0 or float(scores[x0:x1].max(initial=0.0)) <= 0:
            roots.append(int(round(expected)))
        else:
            roots.append(x0 + int(np.argmax(scores[x0:x1])))
    return tuple(roots)


def transistor_metal_pixels(image_rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    saturation = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)[:, :, 1]
    return ((gray >= 80) & (saturation <= 75) & (~red_dominant(image_rgb))).astype(
        np.uint8
    )


def detect_transistor_board_holes(
    image_rgb: np.ndarray,
) -> list[tuple[float, float, float]]:
    height, width = image_rgb.shape[:2]
    detection_size = 256
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(
        gray, (detection_size, detection_size), interpolation=cv2.INTER_AREA
    )
    gray = cv2.GaussianBlur(gray, (7, 7), 1.5)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.1,
        minDist=18,
        param1=80,
        param2=18,
        minRadius=5,
        maxRadius=14,
    )
    if circles is None:
        return []
    return [
        (
            float(center_x) * width / detection_size,
            float(center_y) * height / detection_size,
            float(radius) * min(height, width) / detection_size,
        )
        for center_x, center_y, radius in circles[0]
    ]


def select_transistor_terminal_holes(
    holes: list[tuple[float, float, float]],
    body_center: float,
    body_bottom: int,
    width: int,
    height: int,
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    expected_y = max(0.84 * height, body_bottom + 0.28 * height)
    expected_xs = (body_center - 0.23 * width, body_center, body_center + 0.23 * width)
    candidates = [
        hole
        for hole in holes
        if body_bottom + 0.18 * height <= hole[1] <= 0.95 * height
        and 0.12 * width <= hole[0] <= 0.88 * width
    ]
    selected = []
    used: set[int] = set()
    for expected_x in expected_xs:
        choices = [
            (
                abs(x - expected_x) / width + 0.35 * abs(y - expected_y) / height,
                index,
                x,
                y,
            )
            for index, (x, y, _) in enumerate(candidates)
            if index not in used and abs(x - expected_x) <= 0.12 * width
        ]
        if choices:
            _, index, x, y = min(choices)
            used.add(index)
            selected.append((int(round(x)), int(round(y))))
        else:
            selected.append((int(round(expected_x)), int(round(expected_y))))
    return tuple(selected)


def keep_transistor_components(mask: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    if num_labels <= 1:
        return mask.astype(np.uint8)
    height, width = mask.shape[:2]
    cleaned = np.zeros_like(mask, dtype=np.uint8)
    min_area = max(80, int(height * width * 0.0015))
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        component_width = stats[label, cv2.CC_STAT_WIDTH]
        component_height = stats[label, cv2.CC_STAT_HEIGHT]
        cx, cy = centroids[label]
        overlaps_body = (
            x < 0.73 * width
            and x + component_width > 0.27 * width
            and y < 0.60 * height
            and y + component_height > 0.12 * height
        )
        looks_like_lead = (
            0.23 * width <= cx <= 0.77 * width
            and cy >= 0.45 * height
            and component_height >= 0.08 * height
        )
        if area >= min_area and (overlaps_body or looks_like_lead):
            cleaned[labels == label] = 1
    return cleaned


def zipper_prompt_points(
    image_rgb: np.ndarray, box: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    height, width = image_rgb.shape[:2]
    x0, y0, x1, y1 = [float(value) for value in box]
    box_width = max(1.0, x1 - x0)
    box_height = max(1.0, y1 - y0)
    positives = [
        (x0 + 0.25 * box_width, y0 + 0.18 * box_height),
        (x0 + 0.50 * box_width, y0 + 0.18 * box_height),
        (x0 + 0.75 * box_width, y0 + 0.18 * box_height),
        (x0 + 0.25 * box_width, y0 + 0.50 * box_height),
        (x0 + 0.50 * box_width, y0 + 0.50 * box_height),
        (x0 + 0.75 * box_width, y0 + 0.50 * box_height),
        (x0 + 0.25 * box_width, y0 + 0.82 * box_height),
        (x0 + 0.50 * box_width, y0 + 0.82 * box_height),
        (x0 + 0.75 * box_width, y0 + 0.82 * box_height),
    ]
    negatives = [
        (max(0.0, x0 - 0.20 * box_width), y0 + 0.25 * box_height),
        (min(width - 1.0, x1 + 0.20 * box_width), y0 + 0.25 * box_height),
        (max(0.0, x0 - 0.20 * box_width), y0 + 0.75 * box_height),
        (min(width - 1.0, x1 + 0.20 * box_width), y0 + 0.75 * box_height),
    ]
    points = np.asarray(positives + negatives, dtype=np.float32)
    labels = np.asarray([1] * len(positives) + [0] * len(negatives), dtype=np.int32)
    return points, labels


def transistor_foreground_from_hq_sam(
    predictor, image_rgb: np.ndarray, args: argparse.Namespace
) -> np.ndarray:
    return foreground_from_hq_sam(predictor, image_rgb, "transistor", args)


def transistor_foreground_from_color(image_rgb: np.ndarray) -> np.ndarray:
    height, width = image_rgb.shape[:2]
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    red = red_dominant(image_rgb)
    yy, xx = np.mgrid[:height, :width]

    central_body = (
        (xx >= 0.16 * width)
        & (xx <= 0.84 * width)
        & (yy >= 0.04 * height)
        & (yy <= 0.72 * height)
    )
    dark = ((gray < 82) & (~red) & central_body).astype(np.uint8)
    dark = cv2.morphologyEx(
        dark, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2
    )
    dark = cv2.morphologyEx(
        dark, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1
    )
    body = keep_components(dark, min_area_ratio=0.01, max_components=1)
    body = fill_holes(body)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        body, connectivity=8
    )
    if num_labels <= 1:
        return keep_components(dark, min_area_ratio=0.0002)
    body_label = max(
        range(1, num_labels), key=lambda label: stats[label, cv2.CC_STAT_AREA]
    )
    body = (labels == body_label).astype(np.uint8)
    x, y, box_width, box_height, _ = stats[body_label]

    x0 = max(0, x - int(0.08 * width))
    x1 = min(width, x + box_width + int(0.08 * width))
    y1 = min(height, y + box_height)
    lead_roi = (
        (xx >= x0) & (xx < x1) & (yy >= y1 - int(0.03 * height)) & (yy <= 0.98 * height)
    )
    neutral = saturation < 90
    lead = ((gray >= 65) & (gray <= 245) & neutral & (~red) & lead_roi).astype(np.uint8)
    lead = cv2.morphologyEx(
        lead, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1
    )
    lead = cv2.morphologyEx(
        lead, cv2.MORPH_CLOSE, np.ones((7, 5), np.uint8), iterations=2
    )
    lead = cv2.dilate(lead, np.ones((3, 3), np.uint8), iterations=1)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        lead, connectivity=8
    )
    clean_lead = np.zeros_like(lead)
    min_area = max(30, int(height * width * 0.00025))
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        _, _, _, component_height, _ = stats[label]
        _, cy = centroids[label]
        if area >= min_area and cy > y + 0.4 * box_height and component_height >= 4:
            clean_lead[labels == label] = 1

    mask = np.maximum(body, clean_lead)
    mask = cv2.morphologyEx(
        mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1
    )
    mask = remove_transistor_board_holes(mask, image_rgb)
    mask = keep_components(mask, min_area_ratio=0.0002)
    return complete_transistor_leads(mask, image_rgb)


def remove_transistor_board_holes(
    mask: np.ndarray, image_rgb: np.ndarray
) -> np.ndarray:
    height, width = mask.shape
    circles = detect_transistor_board_holes(image_rgb)
    if not circles:
        return mask

    cleaned = mask.copy()
    for center_x, center_y, radius in circles:
        x = int(round(center_x))
        y = int(round(center_y))
        r = int(round(radius + min(height, width) / 256))
        cv2.circle(cleaned, (x, y), r, 0, thickness=-1)
    return cleaned


def zipper_foreground_from_image(image_rgb: np.ndarray) -> np.ndarray:
    height, width = image_rgb.shape[:2]
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    mask = (mask > 0).astype(np.uint8)
    mask = keep_center_components(
        mask, min_area_ratio=0.002, max_components=8, x_range=(0.10, 0.90)
    )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, np.ones((17, 9), np.uint8), iterations=2
    )
    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return mask
    pad_x = max(2, int(round(width * 0.015)))
    pad_y = max(2, int(round(height * 0.015)))
    x0 = max(0, int(xs.min()) - pad_x)
    x1 = min(width, int(xs.max()) + pad_x + 1)
    y0 = max(0, int(ys.min()) - pad_y)
    y1 = min(height, int(ys.max()) + pad_y + 1)
    filled = np.zeros_like(mask)
    filled[y0:y1, x0:x1] = 1
    return filled


def refine_foreground_mask(category: str, mask: np.ndarray) -> np.ndarray:
    mask = cleanup_mask(mask)
    if category == "bottle":
        mask = remove_border_components(mask)
        mask = keep_components(mask, min_area_ratio=0.01, max_components=1)
        mask = fill_holes(mask)
        return cleanup_mask(mask)
    if category == "capsule":
        mask = keep_center_components(
            mask, min_area_ratio=0.005, max_components=1, y_range=(0.02, 0.78)
        )
        mask = fill_holes(mask)
        return cleanup_mask(mask)
    if category == "screw":
        return keep_components(mask, min_area_ratio=0.002, max_components=1)
    return mask


def heuristic_foreground_from_image(image_rgb: np.ndarray, category: str) -> np.ndarray:
    if category == "transistor":
        height, width = image_rgb.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        x0, x1 = int(width * 0.22), int(width * 0.78)
        y0, y1 = int(height * 0.05), int(height * 0.96)
        mask[y0:y1, x0:x1] = 1
        return mask

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    _, background = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    background = (background > 0).astype(np.uint8)
    if category in {"carpet", "cable", "wood"}:
        return np.ones(image_rgb.shape[:2], dtype=np.uint8)
    if category in {"toothbrush", "hazelnut", "metal_nut", "pill"}:
        return background
    return 1 - background


def mask_to_box(
    mask: np.ndarray, width: int, height: int, margin_ratio: float
) -> np.ndarray:
    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        margin_x = int(round(width * margin_ratio))
        margin_y = int(round(height * margin_ratio))
        return np.array(
            [margin_x, margin_y, width - 1 - margin_x, height - 1 - margin_y],
            dtype=np.float32,
        )
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    pad_x = max(2, int(round((x1 - x0 + 1) * margin_ratio)))
    pad_y = max(2, int(round((y1 - y0 + 1) * margin_ratio)))
    x0 = max(0, x0 - pad_x)
    y0 = max(0, y0 - pad_y)
    x1 = min(width - 1, x1 + pad_x)
    y1 = min(height - 1, y1 + pad_y)
    return np.array([x0, y0, x1, y1], dtype=np.float32)


def filled_box_from_mask(mask: np.ndarray) -> np.ndarray:
    mask = (mask > 0).astype(np.uint8)
    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return mask
    filled = np.zeros_like(mask)
    filled[int(ys.min()) : int(ys.max()) + 1, int(xs.min()) : int(xs.max()) + 1] = 1
    return filled


def autocast_context(device: str):
    if str(device).startswith("cuda"):
        torch_module = require_torch()
        return torch_module.autocast("cuda", dtype=torch_module.bfloat16)
    return nullcontext()


def to_numpy(value):
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return value


def require_torch():
    if torch is None:
        raise RuntimeError("PyTorch is required for SAM foreground generation.")
    return torch


def choose_mask(
    masks: np.ndarray,
    scores: np.ndarray,
    args: argparse.Namespace,
    expected_area: float | None = None,
) -> np.ndarray:
    best = masks[int(np.argmax(scores))].astype(np.uint8)
    area_ratio = float(best.mean())
    if is_likely_background(best, area_ratio) or is_much_larger_than_prompt(
        area_ratio, expected_area
    ):
        best = 1 - best
    if args.min_area_ratio <= float(best.mean()) <= args.max_area_ratio:
        return best

    candidates = [
        (float(score), float(mask.mean()), mask) for mask, score in zip(masks, scores)
    ]
    candidates.sort(key=lambda item: (item[0], -abs(item[1] - 0.35)), reverse=True)
    return candidates[0][2].astype(np.uint8)


def red_dominant(image_rgb: np.ndarray) -> np.ndarray:
    image = image_rgb.astype(np.int16)
    r = image[:, :, 0]
    g = image[:, :, 1]
    b = image[:, :, 2]
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    return ((hue < 12) | (hue > 168)) & (sat > 55) & (r > g + 20) & (r > b + 20)


def is_much_larger_than_prompt(area_ratio: float, expected_area: float | None) -> bool:
    if expected_area is None or expected_area <= 0 or expected_area >= 0.5:
        return False
    return area_ratio > 0.65 and area_ratio > expected_area * 1.8


def is_likely_background(mask: np.ndarray, area_ratio: float) -> bool:
    if area_ratio < 0.15:
        return False
    top = float(mask[0, :].mean()) > 0.5
    bottom = float(mask[-1, :].mean()) > 0.5
    left = float(mask[:, 0].mean()) > 0.5
    right = float(mask[:, -1].mean()) > 0.5
    return sum((top, bottom, left, right)) >= 3


def cleanup_mask(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    if num_labels <= 1:
        return mask
    image_area = mask.shape[0] * mask.shape[1]
    min_area = max(64, int(image_area * 0.001))
    cleaned = np.zeros_like(mask)
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == label] = 1
    return cleaned


def keep_components(
    mask: np.ndarray, min_area_ratio: float, max_components: int | None = None
) -> np.ndarray:
    mask = mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    if num_labels <= 1:
        return mask
    image_area = mask.shape[0] * mask.shape[1]
    min_area = max(1, int(image_area * min_area_ratio))
    components = []
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_area:
            components.append((area, label))
    components.sort(reverse=True)
    if max_components is not None:
        components = components[:max_components]
    cleaned = np.zeros_like(mask)
    for _, label in components:
        cleaned[labels == label] = 1
    return cleaned


def keep_center_components(
    mask: np.ndarray,
    min_area_ratio: float,
    max_components: int | None = None,
    x_range: tuple[float, float] = (0.0, 1.0),
    y_range: tuple[float, float] = (0.0, 1.0),
) -> np.ndarray:
    mask = mask.astype(np.uint8)
    height, width = mask.shape
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    if num_labels <= 1:
        return mask
    image_area = height * width
    min_area = max(1, int(image_area * min_area_ratio))
    components = []
    x0, x1 = x_range
    y0, y1 = y_range
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        cx, cy = centroids[label]
        if area < min_area:
            continue
        if not (x0 * width <= cx <= x1 * width and y0 * height <= cy <= y1 * height):
            continue
        components.append((area, label))
    components.sort(reverse=True)
    if max_components is not None:
        components = components[:max_components]
    cleaned = np.zeros_like(mask)
    for _, label in components:
        cleaned[labels == label] = 1
    return cleaned


def remove_border_components(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(np.uint8)
    num_labels, labels, _, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask
    border_labels = set(np.unique(labels[0, :]))
    border_labels.update(np.unique(labels[-1, :]))
    border_labels.update(np.unique(labels[:, 0]))
    border_labels.update(np.unique(labels[:, -1]))
    cleaned = mask.copy()
    for label in border_labels:
        if label != 0:
            cleaned[labels == label] = 0
    return cleaned


def fill_holes(mask: np.ndarray) -> np.ndarray:
    mask = (mask > 0).astype(np.uint8)
    h, w = mask.shape
    flood = (mask * 255).copy()
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    filled = cv2.bitwise_or(mask * 255, holes)
    return (filled > 0).astype(np.uint8)


def overlay_mask(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = image_bgr.copy()
    color = np.zeros_like(image_bgr)
    color[:, :, 2] = 255
    overlay = np.where(
        mask[:, :, None] > 0,
        (0.55 * image_bgr + 0.45 * color).astype(np.uint8),
        overlay,
    )
    return overlay


if __name__ == "__main__":
    main()

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
SPEC = importlib.util.spec_from_file_location(
    "foreground_annotations", ROOT / "tools/refine_foreground_from_annotations.py"
)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def annotation() -> dict:
    return {
        "size": [32, 32],
        "regions": {
            "body": [[4, 2], [28, 2], [28, 14], [4, 14]],
            "left_lead": [[6, 12], [10, 12], [10, 28], [6, 28]],
            "center_lead": [[14, 12], [18, 12], [18, 28], [14, 28]],
            "right_lead": [[22, 12], [26, 12], [26, 28], [22, 28]],
        },
    }


def test_overlapping_roots_are_foreground_and_gaps_are_background() -> None:
    mask = module.rasterize(annotation())
    assert mask[13, 8] == 1  # Polygon union, not even-odd holes at overlaps.
    assert mask[25, 8] == mask[25, 16] == mask[25, 24] == 1
    assert mask[25, 12] == mask[25, 20] == 0
    assert set(np.unique(mask)) == {0, 1}


def test_disconnected_lead_is_rejected() -> None:
    record = annotation()
    record["regions"]["left_lead"] = [[6, 17], [10, 17], [10, 28], [6, 28]]
    with pytest.raises(ValueError, match="connected"):
        module.rasterize(record)


def test_reference_rejects_same_name_different_pixels(tmp_path: Path) -> None:
    path = tmp_path / "sample.png"
    cv2.imwrite(str(path), np.zeros((32, 32, 3), np.uint8))
    record = dict(
        annotation(),
        image=path.name,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    module.read_reference(tmp_path, record)
    cv2.imwrite(str(path), np.full((32, 32, 3), 255, np.uint8))
    with pytest.raises(ValueError, match="hash"):
        module.read_reference(tmp_path, record)


def test_traced_sample_retains_body_sides_and_dark_lead_tips() -> None:
    data = json.loads((ROOT / "assets/foreground/transistor_manual.json").read_text())
    record = next(row for row in data["images"] if row["image"] == "006.png")
    mask = module.rasterize(record)
    # The old circular-hole exclusion incorrectly removed these body pixels.
    assert mask[220, 300] == mask[443, 680] == 1
    # Dark copper terminal tips belong to the object, despite their color.
    assert mask[880, 272] == mask[880, 499] == mask[880, 728] == 1
    assert mask[220, 257] == mask[450, 740] == mask[900, 320] == 0
    for row in data["images"]:
        module.rasterize(row)


def test_package_repair_removes_board_hole_but_keeps_interlead_gaps() -> None:
    baseline = module.rasterize(annotation())
    baseline[5:9, 4:8] = 0  # Spurious side notch from board-hole subtraction.
    transferred = baseline.copy()
    transferred[0:3, 13:19] = 1  # Board hole spur above the package.
    image = np.repeat(
        np.where(module.rasterize(annotation()) > 0, 30, 150).astype(np.uint8)[
            :, :, None
        ],
        3,
        axis=2,
    )
    repaired = module.restore_package_body(transferred, baseline, image)
    assert repaired[6, 5] == 1
    assert repaired[0, 16] == 0
    assert repaired[25, 12] == repaired[25, 20] == 0

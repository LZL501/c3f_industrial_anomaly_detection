from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from refine_mvtec_foregrounds import fill_interior, refine


def test_metal_nut_keeps_actual_void_background() -> None:
    image = np.full((256, 256, 3), 20, np.uint8)
    baseline = np.zeros((256, 256), np.uint8)
    cv2.circle(baseline, (128, 128), 95, 1, -1)
    image[baseline > 0] = 160
    cv2.circle(image, (128, 128), 26, (20, 20, 20), -1)
    mask, _ = refine("metal_nut", image, baseline)
    assert mask[128, 128] == 0
    assert mask[128, 160] == 1
    assert mask[0, 0] == 0
    assert cv2.connectedComponents(mask)[0] == 2


def test_zipper_follows_slanted_edge_without_masking_white_margin() -> None:
    image = np.full((128, 128, 3), 255, np.uint8)
    expected = np.zeros((128, 128), np.uint8)
    for y in range(128):
        left, right = 15 + y // 8, 105 + y // 16
        image[y, left:right] = 60
        expected[y, left:right] = 1
    image[40:45, 55:60] = 255  # A light mark inside the cloth remains foreground.
    baseline = np.ones((128, 128), np.uint8)
    mask, _ = refine("zipper", image, baseline)
    np.testing.assert_array_equal(mask, expected)


def test_fill_interior_handles_object_touching_borders() -> None:
    mask = np.zeros((24, 24), np.uint8)
    mask[:, 0:20] = 1
    mask[10:14, 8:12] = 0
    filled = fill_interior(mask)
    assert filled[11, 10] == 1
    assert filled[11, 23] == 0


def test_cable_rejects_automatic_refinement() -> None:
    image = np.zeros((32, 32, 3), np.uint8)
    baseline = np.ones((32, 32), np.uint8)
    with pytest.raises(ValueError, match="direct manual annotations"):
        refine("cable", image, baseline)


def test_textures_use_full_image_and_reviewed_objects_retain_masks() -> None:
    image = np.zeros((32, 32, 3), np.uint8)
    baseline = np.zeros((32, 32), np.uint8)
    baseline[8:24, 10:22] = 255
    mask, _ = refine("wood", image, baseline)
    assert mask.all()
    mask, _ = refine("hazelnut", image, baseline)
    np.testing.assert_array_equal(mask, baseline > 0)

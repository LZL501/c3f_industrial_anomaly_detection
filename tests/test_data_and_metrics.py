from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from c3f.data.anomaly import SyntheticAnomalyGenerator
from c3f.data.datasets import _mvtec_foreground, _visa_foreground, build_eval_dataloader
from c3f.metrics import compute_aupro, compute_metrics


def test_pseudo_anomaly_is_nonempty_and_inside_foreground(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = SyntheticAnomalyGenerator(resize=(8, 8), texture_root=None)
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    foreground = np.zeros((8, 8), dtype=np.uint8)
    foreground[2:6, 1:7] = 1
    monkeypatch.setattr(
        generator, "_perlin_mask", lambda: np.ones((8, 8), dtype=np.float32)
    )
    monkeypatch.setattr(
        generator, "_source", lambda _: np.full((8, 8, 3), 255, dtype=np.float32)
    )

    _, mask = generator.generate(image, foreground)

    assert mask.any()
    assert np.all(mask <= foreground)


def test_pseudo_anomaly_rejects_empty_foreground() -> None:
    generator = SyntheticAnomalyGenerator(resize=(8, 8), texture_root=None)
    with pytest.raises(ValueError, match="Foreground mask is empty"):
        generator.generate(
            np.zeros((8, 8, 3), dtype=np.uint8), np.zeros((8, 8), dtype=np.uint8)
        )


def test_missing_object_foreground_fails_when_required(tmp_path: Path) -> None:
    path = tmp_path / "bottle" / "train" / "good" / "000.png"
    image = np.zeros((8, 8, 3), dtype=np.uint8)

    with pytest.raises(FileNotFoundError, match="Foreground mask is required"):
        _mvtec_foreground(path, image, (8, 8), "bottle", require=True)


def test_texture_foreground_is_full_image_without_mask(tmp_path: Path) -> None:
    path = tmp_path / "wood" / "train" / "good" / "000.png"
    foreground = _mvtec_foreground(
        path, np.zeros((8, 8, 3), dtype=np.uint8), (8, 8), "wood", require=True
    )
    assert np.all(foreground == 1)


def test_external_foreground_overrides_dataset_mask(tmp_path: Path) -> None:
    path = tmp_path / "good-dataset" / "cable" / "train" / "good" / "001.png"
    internal = path.parent.parent / "foreground" / path.name
    external_root = tmp_path / "mvtec-foreground"
    external = external_root / "cable" / "train" / "foreground" / path.name
    internal.parent.mkdir(parents=True)
    external.parent.mkdir(parents=True)
    cv2.imwrite(str(internal), np.full((8, 8), 255, dtype=np.uint8))
    expected = np.zeros((8, 8), dtype=np.uint8)
    expected[2:6, 3:5] = 255
    cv2.imwrite(str(external), expected)
    image = np.zeros((8, 8, 3), dtype=np.uint8)

    actual = _mvtec_foreground(
        path, image, (8, 8), "cable", require=True, foreground_root=external_root
    )
    np.testing.assert_array_equal(actual, expected > 0)

    external.unlink()
    with pytest.raises(FileNotFoundError, match="mvtec-foreground"):
        _mvtec_foreground(
            path, image, (8, 8), "cable", require=True, foreground_root=external_root
        )


def test_eval_loader_does_not_require_training_assets(tmp_path: Path) -> None:
    image_path = tmp_path / "bottle" / "test" / "good" / "000.png"
    image_path.parent.mkdir(parents=True)
    cv2.imwrite(str(image_path), np.zeros((8, 8, 3), dtype=np.uint8))
    config = {
        "data": {
            "dataset": "mvtec",
            "root": str(tmp_path),
            "texture_root": str(tmp_path / "missing-dtd"),
            "category": "bottle",
            "image_size": 8,
            "batch_size": 1,
            "num_workers": 0,
            "require_foreground": True,
            "require_texture": True,
        }
    }

    image, mask, label = next(iter(build_eval_dataloader(config)))

    assert image.shape == (1, 3, 8, 8)
    assert mask.sum() == 0
    assert label.item() == 0


def test_visa_external_png_is_used_for_jpg_source(tmp_path: Path) -> None:
    path = tmp_path / "VisA" / "capsules" / "Data" / "Images" / "Normal" / "001.JPG"
    mask_root = tmp_path / "visa-foreground"
    mask_path = mask_root / "capsules" / "Data" / "Foreground" / "Normal" / "001.png"
    mask_path.parent.mkdir(parents=True)
    expected = np.zeros((8, 8), dtype=np.uint8)
    expected[1:4, 2:5] = 255
    cv2.imwrite(str(mask_path), expected)
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    actual = _visa_foreground(
        path, image, (8, 8), "capsules", require=True, foreground_root=mask_root
    )
    np.testing.assert_array_equal(actual, expected > 0)
    mask_path.unlink()
    with pytest.raises(FileNotFoundError, match="visa-foreground"):
        _visa_foreground(
            path, image, (8, 8), "capsules", require=True, foreground_root=mask_root
        )


def test_per_image_aupro_mode_matches_released_evaluation() -> None:
    masks = np.zeros((3, 8, 8), dtype=np.uint8)
    masks[1, 1:3, 1:3] = 1
    masks[2, 3:7, 3:7] = 1
    predictions = np.linspace(0, 1, masks.size, dtype=np.float64).reshape(masks.shape)
    labels = np.array([0, 1, 1])

    result = compute_metrics(predictions, masks, labels, aupro_mode="per_image")
    expected = np.mean(
        [
            compute_aupro(masks[1:2], predictions[1:2]),
            compute_aupro(masks[2:3], predictions[2:3]),
        ]
    )

    assert result["aupro"] == pytest.approx(expected)

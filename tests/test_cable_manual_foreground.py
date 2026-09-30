from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from refine_cable_foregrounds import (
    load_annotations,
    main,
    read_manual_mask,
)

ANNOTATIONS = (
    Path(__file__).resolve().parents[1] / "assets/foreground/cable_manual.json"
)


def test_withdrawn_cable_annotations_cannot_be_regenerated() -> None:
    with pytest.raises(ValueError, match="Withdrawn cable annotations"):
        load_annotations(ANNOTATIONS)


def test_source_content_change_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "000.png"
    cv2.imwrite(str(path), np.zeros((16, 16, 3), np.uint8))
    record = {
        "image": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size": [16, 16],
        "polygon": [[2, 2], [13, 2], [13, 13], [2, 13]],
    }
    assert read_manual_mask(path, record)[8, 8] == 1
    cv2.imwrite(str(path), np.full((16, 16, 3), 255, np.uint8))
    with pytest.raises(ValueError, match="Source does not match"):
        read_manual_mask(path, record)


def test_missing_annotation_never_generates_a_mask(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source/cable/train/good"
    source.mkdir(parents=True)
    cv2.imwrite(str(source / "000.png"), np.zeros((16, 16, 3), np.uint8))
    annotations = tmp_path / "empty.json"
    annotations.write_text(
        json.dumps({"version": 1, "category": "cable", "images": []})
    )
    output = tmp_path / "output"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "refine_cable_foregrounds.py",
            "--data-root",
            str(tmp_path / "source"),
            "--output-root",
            str(output),
            "--annotations",
            str(annotations),
        ],
    )
    with pytest.raises(ValueError, match="no automatic fallback"):
        main()
    assert not output.exists()

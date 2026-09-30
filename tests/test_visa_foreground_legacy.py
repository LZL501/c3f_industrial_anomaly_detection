import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from generate_visa_foreground_legacy import legacy_foreground, mask_relative_path


def test_legacy_visa_polarity_keeps_multiple_objects() -> None:
    image = np.full((40, 50, 3), 20, dtype=np.uint8)
    image[4:14, 5:15] = 220
    image[22:32, 30:40] = 220
    expected = image[:, :, 0] == 220
    np.testing.assert_array_equal(legacy_foreground(image, "candle") > 0, expected)
    np.testing.assert_array_equal(legacy_foreground(image, "capsules") > 0, ~expected)


def test_foreground_paths_preserve_category_and_use_lossless_png() -> None:
    assert mask_relative_path("candle/Data/Images/Normal/0001.JPG", "candle") == Path(
        "candle/Data/Foreground/Normal/0001.png"
    )
    with pytest.raises(ValueError):
        mask_relative_path("../candle/Data/Images/Normal/0001.JPG", "candle")
    with pytest.raises(ValueError):
        mask_relative_path("pcb1/Data/Images/Normal/0001.JPG", "candle")

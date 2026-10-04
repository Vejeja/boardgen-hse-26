import importlib.util
from pathlib import Path

import numpy as np
import pytest

SOURCE = (
    Path(__file__).resolve().parents[1]
    / "src/boardgen/methods/repair/repair_border_mask_artifacts.py"
)
SPEC = importlib.util.spec_from_file_location("border_mask_repair", SOURCE)
REPAIR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPAIR)


def test_reviewed_background_speck_removal_preserves_thin_interior_parts():
    alpha = np.zeros((180, 180), np.uint8)
    alpha[50:140, 50:140] = 220
    alpha[45:75, 150:152] = 240
    alpha[60:62, :4] = 180
    expected = alpha.copy()
    expected[60:62, :4] = 0
    actual, edits = REPAIR.remove_verified_border_noise(alpha)
    np.testing.assert_array_equal(actual, expected)
    assert len(edits) == 1


def test_actual_foreground_clipping_requires_new_crop():
    alpha = np.zeros((180, 180), np.uint8)
    alpha[50:140, :100] = 220
    with pytest.raises(ValueError, match="Main foreground is clipped"):
        REPAIR.remove_verified_border_noise(alpha)

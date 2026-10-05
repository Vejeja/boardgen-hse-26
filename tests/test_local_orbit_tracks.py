import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from boardgen.camera.camera_audit import read_model

SOURCE = Path(__file__).resolve().parents[1] / "src/boardgen/methods/repair/fit_orbit_tracks.py"
SPEC = importlib.util.spec_from_file_location("local_orbit_tracks", SOURCE)
TRACKS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRACKS)


def dataset(tmp_path, textured):
    fit = [i for i in range(10) if i % 5 != 4]

    for section in ["images", "masks"]:
        (tmp_path / "reference" / section).mkdir(parents=True)
    rng = np.random.default_rng(312)
    pixels = (
        rng.integers(0, 256, (320, 320, 3), dtype=np.uint8)
        if textured
        else np.full((320, 320, 3), 160, np.uint8)
    )
    mask = np.zeros((320, 320), np.uint8)
    mask[35:285, 35:285] = 255
    for i in fit:
        image = cv2.warpAffine(pixels, np.float32([[1, 0, i], [0, 1, 0]]), (320, 320))
        cv2.imwrite(str(tmp_path / "reference/images" / f"view_{i:04d}.png"), image)
        cv2.imwrite(str(tmp_path / "reference/masks" / f"view_{i:04d}.png"), mask)
    (tmp_path / "dataset.json").write_text(json.dumps({"fit": fit, "heldout": [4, 9]}))
    return fit


def test_fit_only_correspondences_are_explicit_initialization(tmp_path):
    fit = dataset(tmp_path, textured=True)
    result = TRACKS.local_tracks(tmp_path, 500)
    _, images, _ = read_model(tmp_path / "train/sparse/0")
    assert {im["name"] for im in images} == {f"view_{i:04d}.png" for i in fit}
    assert all(len(im["observations"]) >= 3 for im in images)
    assert result["placeholder_poses"] is True
    assert "not training cameras" in json.loads((tmp_path / "dataset.json").read_text())["cameras"]
    for im in images:
        xy = np.rint(im["observations"]["xy"]).astype(int)
        assert ((xy >= 35) & (xy < 285)).all()


def test_featureless_data_cannot_fabricate_camera_evidence(tmp_path):
    dataset(tmp_path, textured=False)
    with pytest.raises(ValueError, match="local tracks"):
        TRACKS.local_tracks(tmp_path, 500)
    assert not (tmp_path / "train/sparse/0").exists()

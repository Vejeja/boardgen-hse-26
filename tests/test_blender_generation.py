import random
import subprocess

import numpy as np
import OpenEXR
import pytest

from boardgen.rendering import blender_render
from boardgen.studio.rendering import sample_objects


def test_failed_blender_render_cannot_reuse_stale_rgb_or_id_annotations(tmp_path, monkeypatch):
    rgb = tmp_path / "scene.png"
    ids = tmp_path / "scene_id0001.exr"
    labels = tmp_path / "scene.txt"
    for path in (rgb, ids, labels):
        path.write_bytes(b"stale")
    scene = {
        "schema": blender_render.SCHEMA,
        "flags": {"id_encoding": blender_render.ID_ENCODING},
        "output": {"rgb_path": str(rgb), "id_path": str(tmp_path / "scene_id.exr")},
    }

    def fail(*args):
        raise subprocess.CalledProcessError(1, "blender")

    monkeypatch.setattr(blender_render, "call_blender", fail)
    with pytest.raises(subprocess.CalledProcessError):
        blender_render.render_scene(scene, tmp_path / "scene.json", labels, blender_bin="blender")
    assert not any(path.exists() for path in (rgb, ids, labels))


def test_visible_instance_pass_preserves_selected_class_mapping(tmp_path, monkeypatch):
    rgb, raw, labels = tmp_path / "rgb.png", tmp_path / "id0001.exr", tmp_path / "rgb.txt"
    scene = {
        "schema": blender_render.SCHEMA,
        "flags": {"id_encoding": blender_render.ID_ENCODING},
        "image_size": [20, 10],
        "output": {"rgb_path": str(rgb), "id_path": str(tmp_path / "id.exr")},
        "objects": [{"pass_index": 1, "class_id": 0}, {"pass_index": 2, "class_id": 1}],
    }

    def simulate(*args):
        rgb.write_bytes(b"rendered")
        ids = np.zeros((10, 20), np.float32)
        ids[2:5, 1:5], ids[5:8, 12:16] = 1, 2
        with OpenEXR.File(
            {}, {"R": ids / 256, "G": (ids / 256) ** 2, "B": (ids > 0).astype(np.float32)}
        ) as image:
            image.write(str(raw))

    monkeypatch.setattr(blender_render, "call_blender", simulate)
    blender_render.render_scene(scene, tmp_path / "scene.json", labels, blender_bin="blender")
    assert [int(row.split()[0]) for row in labels.read_text().splitlines()] == [0, 1]


def test_mixed_size_figures_stay_on_board_and_do_not_overlap():
    assets = {
        0: {"path": "small.glb", "height_m": 0.04, "footprint_over_height": 0.5},
        1: {"path": "large.glb", "height_m": 0.12, "footprint_over_height": 1.5},
    }
    objects = sample_objects(assets, 8, 1.0, 1.0, random.Random(15))
    assert len(objects) == 8
    assert {obj["class_id"] for obj in objects} == {0, 1}
    radii = [
        max(0.5 * obj["height_m"] * assets[obj["class_id"]]["footprint_over_height"], 0.01)
        for obj in objects
    ]
    for i, obj in enumerate(objects):
        x, y = obj["xy"]
        assert abs(x) + radii[i] <= 0.5 - 0.05
        assert abs(y) + radii[i] <= 0.5 - 0.05
        for j in range(i):
            px, py = objects[j]["xy"]
            assert (x - px) ** 2 + (y - py) ** 2 >= (radii[i] + radii[j]) ** 2

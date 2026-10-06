import numpy as np
import OpenEXR
import pytest

from boardgen.rendering.labels import write_yolo_labels_from_id_pass


def save_pass(path, ids, coverage=None):
    coverage = (ids > 0).astype(np.float32) if coverage is None else coverage
    channels = {"R": ids / 256 * coverage, "G": (ids / 256) ** 2 * coverage, "B": coverage}
    with OpenEXR.File({}, channels) as image:
        image.write(str(path))


def test_colour_exr_preserves_multiple_instance_ids(tmp_path):
    ids = np.zeros((20, 40), dtype=np.float32)
    ids[2:8, 3:13] = 1
    ids[10:18, 22:38] = 7
    raw = tmp_path / "scene_id0001.exr"
    save_pass(raw, ids)
    output = tmp_path / "labels.txt"
    write_yolo_labels_from_id_pass(raw, {1: 0, 7: 6}, output, image_size=(40, 20))
    rows = [line.split() for line in output.read_text().splitlines()]
    assert [int(row[0]) for row in rows] == [0, 6]
    np.testing.assert_allclose([float(x) for x in rows[0][1:]], [0.2, 0.25, 0.25, 0.3])
    np.testing.assert_allclose([float(x) for x in rows[1][1:]], [0.75, 0.7, 0.4, 0.4])


def test_bbox_includes_disconnected_visible_parts_and_pixel_edges(tmp_path):
    ids = np.zeros((20, 40), dtype=np.float32)
    ids[0:5, 0:10] = 1
    ids[10:20, 30:40] = 1
    raw = tmp_path / "scene_id0001.exr"
    save_pass(raw, ids)
    output = tmp_path / "labels.txt"
    write_yolo_labels_from_id_pass(raw, {1: 0}, output, image_size=(40, 20))
    row = output.read_text().split()
    np.testing.assert_allclose([float(x) for x in row[1:]], [0.5, 0.5, 1, 1])


def test_antialiased_id_two_cannot_become_id_one(tmp_path):
    ids = np.zeros((20, 40), dtype=np.float32)
    coverage = np.zeros_like(ids)
    ids[2:8, 3:13] = 1
    coverage[2:8, 3:13] = 1
    ids[10:18, 22:38] = 2
    coverage[10:18, 22:38] = 0.5
    channels = {"R": ids / 256 * coverage, "G": (ids / 256) ** 2 * coverage, "B": coverage}

    channels["R"][19, 39] = 2 / 256
    channels["G"][19, 39] = 5 / 65536
    channels["B"][19, 39] = 1
    raw = tmp_path / "scene_id0001.exr"
    with OpenEXR.File({}, channels) as image:
        image.write(str(raw))
    output = tmp_path / "labels.txt"
    write_yolo_labels_from_id_pass(raw, {1: 0, 2: 1}, output, image_size=(40, 20))
    rows = [line.split() for line in output.read_text().splitlines()]
    np.testing.assert_allclose([float(x) for x in rows[0][1:]], [0.2, 0.25, 0.25, 0.3])
    np.testing.assert_allclose([float(x) for x in rows[1][1:]], [0.75, 0.7, 0.4, 0.4])


def test_wrong_resolution_or_unknown_ids_cannot_publish_annotations(tmp_path):
    raw = tmp_path / "scene.exr"
    save_pass(raw, np.full((20, 40), 7, dtype=np.float32))
    output = tmp_path / "labels.txt"
    with pytest.raises(ValueError, match="resolution"):
        write_yolo_labels_from_id_pass(raw, {7: 0}, output, image_size=(20, 40))
    with pytest.raises(ValueError, match="Unknown instance"):
        write_yolo_labels_from_id_pass(raw, {1: 0}, output, image_size=(40, 20))
    assert not output.exists()

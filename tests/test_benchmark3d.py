import json
import struct
import sys

import cv2
import numpy as np
import pytest
import trimesh
from PIL import Image
from scipy.spatial.transform import Rotation

from boardgen.common.image_utils import load_view_masks
from boardgen.evaluation.benchmark3d import (
    constrain_mesh,
    hull,
    inputs,
    render_textured,
    run_command,
    synthetic,
)
from boardgen.evaluation.benchmark_preview import read_colmap
from boardgen.reconstruction.recon3d import load_masks


@pytest.mark.parametrize("size,recover_warm", [(1024, False), (1536, False), (2048, True)])
def test_real_cli_passes_the_requested_mask_resolution(monkeypatch, tmp_path, size, recover_warm):
    from boardgen.evaluation import benchmark3d

    captured = {}
    monkeypatch.setattr(benchmark3d, "real", lambda *args, **kwargs: captured.update(kwargs))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark3d",
            "real",
            str(tmp_path / "dataset"),
            "video.mp4",
            "--mask-size",
            str(size),
            *(["--recover-warm"] if recover_warm else []),
        ],
    )
    benchmark3d.main()
    assert captured["mask_size"] == size
    assert captured["recover_warm"] is recover_warm


def test_loader_preserves_alpha_holes_and_ignores_qc(tmp_path):
    rgba = np.zeros((32, 32, 4), np.uint8)
    rgba[4:28, 4:28, 3] = 255
    rgba[14:18, 14:18, 3] = 0
    rgba[0, :, 3] = 10
    Image.fromarray(rgba).save(tmp_path / "view_0000.png")
    Image.fromarray(np.full((32, 32), 255, np.uint8)).save(tmp_path / "_plate.png")
    masks = load_masks(tmp_path)
    assert masks.shape == (1, 32, 32)
    assert not masks[0, 14:18, 14:18].any()
    assert not masks[0, 0].any()
    assert masks[0].sum() == 24 * 24 - 16


def test_fixture_exports_fit_only_and_roundtrips_camera_conventions(tmp_path):
    root = tmp_path / "fixture"
    synthetic(root, size=64, views=10)
    data, model, _ = inputs(root)
    transforms = json.loads((root / "train/transforms.json").read_text())
    names = {row["file_path"].split("/")[-1] for row in transforms["frames"]}
    assert names == {f"view_{i:04d}.png" for i in data["fit"]}
    assert not names & {f"view_{i:04d}.png" for i in data["heldout"]}
    assert not list((root / "train").rglob("truth*"))
    headers = (root / "train/sparse/0/images.txt").read_text().splitlines()[::2]
    for header, frame, i in zip(headers, transforms["frames"], data["fit"], strict=True):
        vals = header.split()
        q = np.array(vals[1:5], float)
        r = Rotation.from_quat([*q[1:], q[0]]).as_matrix()
        t = np.array(vals[5:8], float)
        w2c = np.vstack([np.column_stack([r, t]), [0, 0, 0, 1]])
        cv_c2w = np.array(frame["transform_matrix"])
        cv_c2w[:, 1:3] *= -1
        np.testing.assert_allclose(np.linalg.inv(cv_c2w), w2c, atol=1e-10)
        np.testing.assert_allclose(
            model.effective_intrinsics().K() @ w2c[:3], model.projection_matrices()[i], atol=1e-10
        )


def test_heldout_masks_do_not_change_reconstructed_geometry(tmp_path):
    root = tmp_path / "fixture"
    synthetic(root, size=64, views=10)
    before = hull(root, grid=24)
    data = json.loads((root / "dataset.json").read_text())
    for i in data["heldout"]:
        Image.fromarray(np.zeros((64, 64), np.uint8)).save(
            root / "reference/masks" / f"view_{i:04d}.png"
        )
    after = hull(root, grid=24)
    assert before["mesh_sha256"] == after["mesh_sha256"]
    assert after["heldout"]["iou_mean"] == 0
    assert before["heldout"]["iou_mean"] > 0.8


def test_external_exit_zero_without_artifact_is_failure(tmp_path):
    artifact = tmp_path / "missing.ply"
    result = run_command(tmp_path / "run", [sys.executable, "-c", "pass"], [str(artifact)])
    assert result["returncode"] == 0
    assert result["status"] == "failed"


def test_unreadable_mask_fails_without_shifting_camera_alignment(tmp_path):
    Image.fromarray(np.ones((8, 8), np.uint8) * 255).save(tmp_path / "view_0000.png")
    (tmp_path / "view_0001.png").write_bytes(b"corrupt image")
    with pytest.raises(ValueError, match="alignment"):
        load_view_masks(str(tmp_path))


def test_binary_colmap_projection_uses_world_to_camera_translation(tmp_path):
    sparse = tmp_path / "sparse/0"
    sparse.mkdir(parents=True)
    (sparse / "cameras.bin").write_bytes(struct.pack("<QiiQQdddd", 1, 7, 1, 80, 60, 50, 55, 40, 30))
    (sparse / "images.bin").write_bytes(
        struct.pack("<Qidddddddi", 1, 9, 1, 0, 0, 0, 0, 0, 2, 7)
        + b"view_0000.png\0"
        + struct.pack("<Q", 0)
    )
    name, projection, width, height = read_colmap(tmp_path)[0]
    hp = projection @ [1, 1, 0, 1]
    np.testing.assert_allclose(hp[:2] / hp[2], [65, 57.5])
    assert (name, width, height) == ("view_0000.png", 80, 60)


def test_texture_renderer_resolves_occlusion_independent_of_face_order():
    front = np.array([[1, 1, 1], [5, 1, 1], [1, 5, 1]], float)
    vertices = np.vstack([front, front * 2])
    mesh = trimesh.Trimesh(vertices=vertices, faces=[[0, 1, 2], [3, 4, 5]], process=False)
    atlas = Image.fromarray(np.array([[[255, 0, 0], [0, 0, 255]]], np.uint8))
    mesh.visual = trimesh.visual.texture.TextureVisuals(
        uv=np.array([[0, 0]] * 3 + [[1, 0]] * 3), image=atlas
    )
    projection = np.column_stack([np.eye(3), np.zeros(3)])
    first = render_textured(mesh, projection, 8, 8)
    mesh.faces = mesh.faces[::-1]
    second = render_textured(mesh, projection, 8, 8)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first[2, 2], [255, 0, 0])
    np.testing.assert_array_equal(first[7, 7], [0, 0, 0])


def test_fit_mask_constraint_removes_floaters_without_reading_heldout_for_geometry(tmp_path):
    root = tmp_path / "fixture"
    synthetic(root, size=64, views=10)
    mesh = trimesh.load(root / "truth.ply", force="mesh")
    floater = mesh.copy()
    floater.apply_translation([4, 0, 0])
    combined = trimesh.util.concatenate([mesh, floater])
    path = root / "with-floater.ply"
    combined.export(path)
    before = constrain_mesh(root, path, root / "before")
    assert 0 < before["faces"] <= len(mesh.faces)
    data = json.loads((root / "dataset.json").read_text())
    for i in data["heldout"]:
        Image.fromarray(np.zeros((64, 64), np.uint8)).save(
            root / "reference/masks" / f"view_{i:04d}.png"
        )
    after = constrain_mesh(root, path, root / "after")
    assert before["mesh_sha256"] == after["mesh_sha256"]
    assert after["heldout"]["iou_mean"] == 0


def test_real_export_keeps_neural_alpha_and_excludes_heldout_from_sfm(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from boardgen.evaluation import benchmark3d
    from boardgen.video import neuralseg, turntable

    memory_events = []

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            mps=SimpleNamespace(
                set_per_process_memory_fraction=lambda value: memory_events.append(
                    ("limit", value)
                ),
                empty_cache=lambda: memory_events.append(("release", None)),
            )
        ),
    )

    monkeypatch.chdir(tmp_path)
    video = tmp_path / "toy.mp4"
    video.write_bytes(b"video input for hash; decoder is replaced below")
    probes = []

    def fake_probe(path, cache_dir):
        probes.append(cache_dir)
        return SimpleNamespace(
            to_json=lambda: {
                "bbox": [0, 0, 48, 64],
                "width": 48,
                "height": 64,
                "n_frames": 20,
                "period_frames": 10,
                "period_confidence": 0.8,
            }
        )

    monkeypatch.setattr(turntable, "probe", fake_probe)

    class Capture:
        def __init__(self, _):
            pass

        def set(self, *args):
            pass

        def read(self):
            return True, np.full((64, 48, 3), 120, np.uint8)

        def release(self):
            pass

    alpha = np.zeros((64, 48), np.uint8)
    alpha[8:56, 8:40] = 170
    alpha[28:36, 20:28] = 0
    monkeypatch.setattr(cv2, "VideoCapture", Capture)
    monkeypatch.setattr(neuralseg, "load_model", lambda: (None, "mps"))
    monkeypatch.setattr(neuralseg, "matte_batch", lambda *args, **kwargs: alpha[None])
    monkeypatch.setattr(neuralseg, "model_metadata", lambda: {"model": "test", "revision": "fixed"})
    from boardgen.video import views

    monkeypatch.setattr(views, "source_provenance", lambda: {"seed": 42})
    root = tmp_path / "real"
    benchmark3d.real(root, video, size=64, views=10)
    assert memory_events == [("limit", 0.8)] + [("release", None)] * 10
    data = json.loads((root / "dataset.json").read_text())
    fit_names = {f"view_{i:04d}.png" for i in data["fit"]}
    for folder in ("images", "masks", "feature_masks"):
        assert {p.name for p in (root / "train" / folder).glob("*.png")} == fit_names
    np.testing.assert_array_equal(cv2.imread(str(root / "train/masks/view_0000.png"), 0), alpha)
    np.testing.assert_array_equal(
        cv2.imread(str(root / "train/feature_masks/view_0000.png"), 0), (alpha >= 128) * 255
    )
    assert cv2.imread(str(root / "contact.png")).shape[:2] == (256, 1024)

    assert data["source_frame_indices"][-1] == 9
    assert probes[0].endswith(benchmark3d.file_hash(video))
    full = tmp_path / "full"
    benchmark3d.real(full, video, size=64, views=10, full_clip=True)
    assert json.loads((full / "dataset.json").read_text())["source_frame_indices"][-1] == 19
    video.write_bytes(b"different video with the same filename")
    benchmark3d.real(tmp_path / "changed", video, size=64, views=10)
    assert probes[-1] != probes[0]

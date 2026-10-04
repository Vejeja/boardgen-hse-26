import json
import shutil
import struct

import numpy as np
import pytest

from boardgen.camera.camera_audit import audit, orbit_diagnostics
from boardgen.evaluation.benchmark_preview import read_colmap, thumbnail
from boardgen.methods.orbit.refine_camera import refine


def sparse_fixture(root, kind=0, count=4):
    sparse = root / "sparse/0"
    sparse.mkdir(parents=True)
    params = (50, 40, 30) if kind == 0 else (60, 50, 40, 30)
    (sparse / "cameras.bin").write_bytes(
        struct.pack("<QiiQQ", 1, 7, kind, 80, 60) + struct.pack("<" + "d" * len(params), *params)
    )
    fx, fy = (50, 50) if kind == 0 else (60, 50)
    payload = bytearray(struct.pack("<Q", count))
    for i in range(count):
        theta = 2 * np.pi * i / count
        tx, ty = -np.cos(theta), -np.sin(theta)
        payload.extend(struct.pack("<idddddddi", i + 1, 1, 0, 0, 0, tx, ty, 0, 7))
        payload.extend(f"view_{i:04d}.png\0".encode())
        payload.extend(
            struct.pack("<Qddqddq", 2, fx * tx / 10 + 40, fy * ty / 10 + 30, 3, 0, 0, -1)
        )
    (sparse / "images.bin").write_bytes(payload)
    (sparse / "points3D.bin").write_bytes(
        struct.pack("<QQdddBBBdQ", 1, 3, 0, 0, 10, 255, 255, 255, 0, 0)
    )
    return sparse


@pytest.mark.parametrize("kind", [0, 1])
def test_audit_recomputes_tracks_and_parses_both_pinhole_models(tmp_path, kind):
    sparse = sparse_fixture(tmp_path, kind)
    names = {f"view_{i:04d}.png" for i in range(4)}
    row = audit(sparse, tmp_path / "audit.json", names)
    assert row["reprojection_mean_px"] < 1e-12
    assert all(v["observations"] == 1 for v in row["per_view"])
    assert not row["missing_fit_views"]
    assert row["cameras"][7]["fx_over_fy"] == (1 if kind == 0 else 1.2)
    assert any("Anisotropic" in w for w in row["warnings"]) == (kind == 1)
    _, p, _, _ = read_colmap(tmp_path)[0]
    uv = p @ [0, 0, 10, 1]
    np.testing.assert_allclose(uv[:2] / uv[2], [34 if kind == 1 else 35, 30])


def test_uneven_angular_sampling_does_not_make_a_circle_elliptical():
    theta = np.r_[np.linspace(0, 0.2, 25), np.linspace(0.4, 5.9, 12)]
    centers = np.column_stack([3 * np.cos(theta), 3 * np.sin(theta), np.full(len(theta), 2)])
    row = orbit_diagnostics(centers)
    assert row["radius_cv"] < 1e-12
    assert row["plane_rms_over_radius"] < 1e-12
    assert abs(row["estimated_sweep_deg"] - np.degrees(5.9)) < 1e-9


def test_preview_preserves_aspect_ratio():
    rgb = np.full((20, 10, 3), 255, np.uint8)
    result = thumbnail(rgb, 40, 40)
    assert result.shape == (40, 40, 3)
    assert np.all(result[:, 10:30] == 255)
    assert not result[:, :10].any()
    assert not result[:, 30:].any()


def test_refinement_rejects_heldout_and_incomplete_models_before_running(tmp_path):
    root = tmp_path / "dataset"
    sparse = sparse_fixture(root)
    (root / "dataset.json").write_text(json.dumps({"fit": [0, 1, 2], "heldout": [3]}))
    with pytest.raises(ValueError, match="outside the fit"):
        refine(root, tmp_path / "target", tmp_path / "unused-executable", sparse)
    (root / "dataset.json").write_text(json.dumps({"fit": list(range(10)), "heldout": [10]}))
    with pytest.raises(ValueError, match="incomplete SfM"):
        refine(root, tmp_path / "target", tmp_path / "unused-executable", sparse, True)
    assert not (tmp_path / "target").exists()


def test_partial_refinement_copies_only_registered_fit_and_preserves_source(tmp_path, monkeypatch):
    from boardgen.methods.orbit import refine_camera as camera_refine

    root = tmp_path / "dataset"
    sparse = sparse_fixture(root, kind=1, count=9)
    original = (sparse / "cameras.bin").read_bytes()
    (root / "reference").mkdir()
    (root / "dataset.json").write_text(json.dumps({"fit": list(range(10)), "heldout": [10]}))
    for folder in ("images", "masks", "feature_masks"):
        dest = root / "train" / folder
        dest.mkdir(parents=True)
        for i in range(11):
            (dest / f"view_{i:04d}.png").write_bytes(b"fixture")

    def fake_solve(out, command, expected, timeout):
        model = tmp_path / "target/train/sparse/0"
        model.mkdir(parents=True)
        for name in ("cameras.bin", "images.bin", "points3D.bin"):
            shutil.copy2(tmp_path / "target/camera-fit/seed" / name, model / name)
        return {"status": "success", "runtime_seconds": 0}

    monkeypatch.setattr(camera_refine, "run_command", fake_solve)
    target = tmp_path / "target"
    row = refine(root, target, tmp_path / "unused", sparse, True)
    assert row["unused_fit"] == [9]
    for folder in ("images", "masks", "feature_masks"):
        assert {p.name for p in (target / "train" / folder).glob("*.png")} == {
            f"view_{i:04d}.png" for i in range(9)
        }
    data = json.loads((target / "dataset.json").read_text())
    assert data["heldout"] == [10] and data["planned_fit"] == list(range(10))
    assert (sparse / "cameras.bin").read_bytes() == original

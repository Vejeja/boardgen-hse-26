import numpy as np

from boardgen.camera.camera_audit import read_model
from boardgen.methods.orbit.calibrate_orbit import (
    repeat_period,
    rotations,
    triangulate,
    write_model,
)


def test_turntable_rotations_are_proper_and_look_at_origin():
    angles = np.radians([0, 90, 180, 270, 360])
    matrices = rotations(angles, np.radians(24), np.radians(-1))
    for r in matrices:
        np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-12)
        assert abs(np.linalg.det(r) - 1) < 1e-12
    np.testing.assert_allclose(matrices[0], matrices[-1], atol=1e-12)
    for angle, r in zip(angles, matrices, strict=True):
        pos = np.array(
            [
                np.sin(angle) * np.cos(np.radians(24)),
                np.sin(np.radians(24)),
                np.cos(angle) * np.cos(np.radians(24)),
            ]
        )
        np.testing.assert_allclose(r @ pos, [0, 0, -1], atol=1e-12)


def test_retriangulation_and_binary_model_round_trip(tmp_path):
    rotations_ = rotations(np.radians([-20, 0, 20]), np.radians(25), 0)
    ts = np.tile([0.0, 0.0, 1.0], (3, 1))
    point = np.array([0.03, 0.01, 0.02])
    k = np.array([[1200, 0, 200], [0, 1200, 260], [0, 0, 1]])
    cp = np.einsum("nab,b->na", rotations_, point) + ts
    xy = cp[:, :2] / cp[:, 2:] * 1200 + [200, 260]
    recovered, err, depth = triangulate(rotations_, ts, k, np.arange(3), xy)
    np.testing.assert_allclose(recovered, point, atol=1e-12)
    assert err.max() < 1e-10 and (depth > 0).all()
    ims = [
        {"id": i + 1, "name": f"view_{i:04d}.png", "r": r, "t": t}
        for i, (r, t) in enumerate(zip(rotations_, ts, strict=True))
    ]
    folder = tmp_path / "model"
    write_model(
        folder,
        {"width": 400, "height": 520, "fx": 1200, "cx": 200, "cy": 260},
        ims,
        [point],
        [(i, 0, uv) for i, uv in enumerate(xy)],
        np.array([[10, 20, 30]]),
    )
    cams, actual, pts = read_model(folder)
    np.testing.assert_allclose(pts[1], point)
    for im, wanted in zip(actual, ims, strict=True):
        np.testing.assert_allclose(im["r"], wanted["r"], atol=1e-12)
        np.testing.assert_allclose(im["t"], wanted["t"], atol=1e-12)
        assert im["observations"]["pid"].tolist() == [1]
    assert cams[1]["fx"] == 1200


def test_period_estimation_never_reads_heldout_frames(tmp_path):
    import cv2

    rng = np.random.default_rng(2)
    (tmp_path / "reference/images").mkdir(parents=True)
    (tmp_path / "reference/masks").mkdir()
    im = rng.integers(0, 255, (20, 20, 3), dtype="uint8")
    mask = np.full((20, 20), 255, dtype="uint8")
    for i in [0, 10]:
        cv2.imwrite(str(tmp_path / f"reference/images/view_{i:04d}.png"), im)
        cv2.imwrite(str(tmp_path / f"reference/masks/view_{i:04d}.png"), mask)
    period, rows = repeat_period(tmp_path, [0, 10], [10])
    assert period == 10 and rows[0]["pairs"] == 1 and rows[0]["loss"] == 0


def test_geometry_proxy_preserves_uv_split_surface_without_chart_fragments():
    import trimesh

    from boardgen.methods.orbit.geometry_proxy import proxy

    mesh = trimesh.Trimesh(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 0, 0], [1, 1, 0], [0, 1, 0]],
        [[0, 1, 2], [3, 4, 5]],
        process=False,
        visual=trimesh.visual.TextureVisuals(
            uv=[[0, 0], [1, 0], [1, 1], [0.2, 0.2], [0.8, 0.8], [0, 1]]
        ),
    )
    result = proxy(mesh, 2)
    assert len(result.vertices) == 4 and len(result.faces) == 2
    assert abs(result.area - 1) < 1e-12
    np.testing.assert_allclose(result.bounds, mesh.bounds)
    assert len(mesh.vertices) == 6

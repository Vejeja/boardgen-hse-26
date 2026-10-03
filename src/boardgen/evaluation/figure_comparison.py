from __future__ import annotations

import numpy as np


def camera_matrices(camera: dict, rotation, translation, near=0.001, far=100.0):
    r, t = np.asarray(rotation, float), np.asarray(translation, float)
    if r.shape != (3, 3) or t.shape != (3,) or not np.isfinite(r).all() or not np.isfinite(t).all():
        raise ValueError("Invalid camera extrinsics")
    if not np.allclose(r @ r.T, np.eye(3), atol=1e-5) or not np.isclose(
        np.linalg.det(r), 1, atol=1e-5
    ):
        raise ValueError("Camera rotation must be a proper orthogonal matrix")
    w, h, fx, fy, cx, cy = (float(camera[k]) for k in ["width", "height", "fx", "fy", "cx", "cy"])
    if min(w, h, fx, fy, near) <= 0 or far <= near or not np.isfinite([w, h, fx, fy, cx, cy]).all():
        raise ValueError("Invalid camera intrinsics or clip range")
    projection = np.array(
        [
            [2 * fx / w, 0, 1 - 2 * cx / w, 0],
            [0, 2 * fy / h, 2 * cy / h - 1, 0],
            [0, 0, -(far + near) / (far - near), -2 * far * near / (far - near)],
            [0, 0, -1, 0],
        ]
    )
    extrinsics = np.eye(4)
    extrinsics[:3, :3], extrinsics[:3, 3] = r, t
    return projection, np.diag([1, -1, -1, 1]) @ extrinsics


def model_to_world(transform):
    matrix = np.asarray(transform, float)
    if (
        matrix.shape != (4, 4)
        or not np.isfinite(matrix).all()
        or not np.allclose(matrix[3], [0, 0, 0, 1])
    ):
        raise ValueError("Invalid model normalization")
    scales = np.linalg.svd(matrix[:3, :3], compute_uv=False)
    if (
        scales.min() <= 0
        or not np.allclose(scales, scales[0], rtol=1e-5)
        or np.linalg.det(matrix[:3, :3]) <= 0
    ):
        raise ValueError("Model comparison requires a uniform, orientation-preserving similarity")
    return np.linalg.inv(matrix)

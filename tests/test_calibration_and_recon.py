import math

import cv2
import numpy as np
import pytest

from boardgen.camera.camera_model import CameraIntrinsics, TurntableCameraModel, TurntableView
from boardgen.reconstruction.recon3d import carve_visual_hull_calibrated


def _project_points(p, points):
    pts_h = np.column_stack([points, np.ones(points.shape[0])])
    hp = pts_h @ p.T
    uv = hp[:, :2] / hp[:, 2:3]
    return uv


@pytest.mark.unit
def test_calibrated_visual_hull_on_synthetic_box_silhouettes():
    intr = CameraIntrinsics(width=96, height=96, fx=80.0, fy=80.0, cx=47.5, cy=47.5)
    views = [
        TurntableView(frame_index=i, theta_rad=i * 2.0 * math.pi / 8.0, view_id=f"view_{i:04d}")
        for i in range(8)
    ]
    model = TurntableCameraModel(
        image_width=96,
        image_height=96,
        intrinsics=intr,
        views=views,
        elevation_deg=18.0,
        camera_distance=2.2,
    )
    corners = np.array(
        [[x, y, z] for x in (-0.18, 0.18) for y in (-0.14, 0.14) for z in (0.05, 0.42)],
        dtype=np.float64,
    )
    masks = np.zeros((len(views), 96, 96), np.uint8)
    for i, p in enumerate(model.projection_matrices()):
        uv = _project_points(p, corners)
        hull = cv2.convexHull(np.round(uv).astype(np.int32))
        cv2.fillConvexPoly(masks[i], hull, 1)

    verts, faces = carve_visual_hull_calibrated(
        masks,
        model.projection_matrices(),
        grid=24,
        consensus=0.85,
        bounds=((-0.35, 0.35), (-0.35, 0.35), (0.0, 0.6)),
        samples_per_axis=1,
    )

    assert verts.shape[0] > 0
    assert faces.shape[0] > 0
    assert float(verts[:, 2].min()) >= -1e-6
    assert float(verts[:, 2].max()) <= 0.65

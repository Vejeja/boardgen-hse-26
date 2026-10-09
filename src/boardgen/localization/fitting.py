from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from .assets import FigureGeometry
from .geometry import Camera


def box_from_points(points):
    low, high = np.min(points, axis=0), np.max(points, axis=0)
    return np.r_[(low + high) / 2, high - low]


def fit_figure(camera: Camera, geometry: FigureGeometry, box_px, board_size, *, yaw_step=15):
    box = np.asarray(box_px, float)
    if box.shape != (4,) or not np.isfinite(box).all() or min(box[2:]) < 2:
        raise ValueError("Некорректная рамка фигуры")
    if not 1 <= yaw_step <= 90:
        raise ValueError("Шаг поворота должен быть от 1 до 90°")
    bounds = np.asarray(board_size, float) / 2
    if bounds.shape != (2,) or not np.isfinite(bounds).all() or np.any(bounds <= 0):
        raise ValueError("Размеры поля должны быть положительными")
    initial = camera.plane_point([box[0], box[1] + box[3] / 2])
    initial = np.clip(initial, -bounds + 1e-6, bounds - 1e-6)
    normalizer = max(box[3], 8)
    weights = np.array([1.0, 1.0, 0.5, 0.5]) / normalizer

    def residual(xy, yaw):
        try:
            projected = box_from_points(camera.project(geometry.placed(xy, yaw)))
            return (projected - box) * weights
        except ValueError:
            return np.full(4, 1e3)

    candidates = []
    for yaw in np.arange(0, 360, yaw_step):
        fit = least_squares(
            residual,
            initial,
            args=(yaw,),
            bounds=(-bounds, bounds),
            loss="soft_l1",
            f_scale=0.05,
            max_nfev=35,
            diff_step=1e-4,
        )
        candidates.append((float(np.linalg.norm(residual(fit.x, yaw))), fit.x, float(yaw)))
    candidates.sort(key=lambda row: row[0])
    refined = []
    for _, xy, yaw in candidates[:4]:

        def refine(parameters):
            return residual(parameters[:2], parameters[2])

        fit = least_squares(
            refine,
            np.r_[xy, yaw],
            bounds=(np.r_[-bounds, yaw - yaw_step], np.r_[bounds, yaw + yaw_step]),
            loss="soft_l1",
            f_scale=0.05,
            max_nfev=60,
            diff_step=1e-4,
        )
        refined.append((float(np.linalg.norm(refine(fit.x))), fit.x[:2], float(fit.x[2] % 360)))
    candidates = sorted(candidates + refined, key=lambda row: row[0])
    score, xy, yaw = candidates[0]
    plausible = [row for row in candidates if row[0] <= score + 0.02]
    centers = np.array([geometry.base_at(p, angle) for _, p, angle in plausible])
    center = geometry.base_at(xy, yaw)
    spread = float(np.max(np.linalg.norm(centers - center, axis=1)))
    projection = box_from_points(camera.project(geometry.placed(xy, yaw)))
    clipped = bool(
        np.any(box[:2] - box[2:] / 2 < 2)
        or np.any(box[:2] + box[2:] / 2 > [camera.width - 2, camera.height - 2])
    )
    reasons = []
    if score > 0.12:
        reasons.append("bbox_mismatch")
    if spread > max(0.01, geometry.radius_m / 2):
        reasons.append("ambiguous_position")
    if clipped:
        reasons.append("image_boundary")
    if np.any(np.abs(center) > bounds):
        reasons.append("base_outside_board")
    return {
        "xy_m": center.tolist(),
        "placement_xy_m": xy.tolist(),
        "yaw_deg": yaw,
        "yaw_observed": False,
        "assumptions": ["upright", "known_size", "full_visible_bbox"],
        "radius_m": geometry.radius_m,
        "bbox_fit_error": score,
        "position_spread_m": spread,
        "projected_box_px": projection.tolist(),
        "hypotheses": len(plausible),
        "status": "uncertain" if reasons else "estimated",
        "reasons": reasons,
    }

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .geometry import Camera, board_corners, estimate_camera


@dataclass(frozen=True)
class BoardFeatures:
    width: int
    height: int
    points: tuple
    descriptors: np.ndarray | None


def prepare_template(template):
    scale = min(1.0, 1400 / max(template.shape[:2]))
    reference = cv2.resize(template, None, fx=scale, fy=scale) if scale < 1 else template
    points, descriptors = cv2.SIFT_create(nfeatures=3500).detectAndCompute(
        cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY), None
    )
    return BoardFeatures(reference.shape[1], reference.shape[0], tuple(points), descriptors)


def register_board(image, template, k, distortion, board_size, *, features=None):
    h, w = image.shape[:2]
    corrected = (
        cv2.undistort(image, np.asarray(k, float), np.asarray(distortion, float))
        if any(distortion)
        else image
    )
    features = features or prepare_template(template)
    sift = cv2.SIFT_create(nfeatures=3500)
    ka, da = features.points, features.descriptors
    kb, db = sift.detectAndCompute(cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY), None)
    if da is None or db is None or min(len(ka), len(kb)) < 12:
        raise ValueError("На поле недостаточно различимых деталей; укажите углы вручную")
    pairs = cv2.BFMatcher().knnMatch(da, db, k=2)
    matches = [
        a for pair in pairs if len(pair) == 2 for a, b in [pair] if a.distance < 0.7 * b.distance
    ]
    if len(matches) < 12:
        raise ValueError("Недостаточно совпадений с картой; укажите углы вручную")
    source = np.array([ka[m.queryIdx].pt for m in matches], float)
    target = np.array([kb[m.trainIdx].pt for m in matches], float)
    homography, mask = cv2.findHomography(source, target, cv2.RANSAC, 3.0)
    if homography is None or mask.sum() < 10:
        raise ValueError("Карта не зарегистрирована: недостаточно согласованных совпадений")
    keep = mask.ravel().astype(bool)
    tw, th = features.width, features.height
    coverage = cv2.contourArea(cv2.convexHull(source[keep].astype(np.float32))) / (tw * th)
    if coverage < 0.04:
        raise ValueError("Совпадения сосредоточены на малой части карты; укажите углы вручную")
    world = np.zeros((keep.sum(), 3))
    world[:, 0] = (source[keep, 0] / tw - 0.5) * board_size[0]
    world[:, 1] = (0.5 - source[keep, 1] / th) * board_size[1]
    camera, quality = estimate_camera(k, (), (w, h), target[keep], *board_size, world_points=world)
    camera = Camera(w, h, k, camera.R, camera.t, tuple(distortion))
    undistorted = Camera(w, h, k, camera.R, camera.t)
    rmse = float(np.sqrt(np.mean(np.sum((undistorted.project(world) - target[keep]) ** 2, axis=1))))
    if rmse > 5 or (-camera.R.T @ camera.t)[2] <= 0:
        raise ValueError("Камера не согласуется с картой; проверьте K и размеры поля")
    quality.update(
        method="template_sift_pnp",
        inliers=int(keep.sum()),
        matches=len(matches),
        template_coverage=float(coverage),
        inlier_rmse_px=rmse,
        corners_px=camera.project(board_corners(*board_size)).tolist(),
    )
    return camera, quality

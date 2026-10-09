from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


def board_corners(width: float, height: float) -> np.ndarray:
    if not np.isfinite([width, height]).all() or min(width, height) <= 0:
        raise ValueError("Размеры поля должны быть положительными")
    return np.array(
        [
            [-width / 2, height / 2, 0],
            [width / 2, height / 2, 0],
            [width / 2, -height / 2, 0],
            [-width / 2, -height / 2, 0],
        ],
        dtype=float,
    )


@dataclass(frozen=True)
class Camera:
    width: int
    height: int
    K: np.ndarray
    R: np.ndarray
    t: np.ndarray
    distortion: tuple[float, ...] = ()

    def __post_init__(self):
        k, r, t = np.asarray(self.K, float), np.asarray(self.R, float), np.asarray(self.t, float)
        if (
            self.width <= 0
            or self.height <= 0
            or k.shape != (3, 3)
            or r.shape != (3, 3)
            or t.shape != (3,)
        ):
            raise ValueError("Некорректные размеры камеры")
        if not all(np.isfinite(v).all() for v in [k, r, t, self.distortion]):
            raise ValueError("Параметры камеры должны быть конечными")
        if k[0, 0] <= 0 or k[1, 1] <= 0 or not np.allclose(k[2], [0, 0, 1]):
            raise ValueError("Некорректная матрица K")
        if not np.allclose(r @ r.T, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(r), 1):
            raise ValueError("Некорректный поворот камеры")
        if len(self.distortion) not in {0, 4, 5, 8}:
            raise ValueError("Нужно 0, 4, 5 или 8 коэффициентов дисторсии")
        for key, value in [("K", k), ("R", r), ("t", t)]:
            value = value.copy()
            value.setflags(write=False)
            object.__setattr__(self, key, value)

    def project(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, float).reshape(-1, 3)
        if np.any((points @ self.R.T + self.t)[:, 2] <= 1e-6):
            raise ValueError("Геометрия находится за камерой")
        if any(self.distortion):
            return cv2.projectPoints(
                points, cv2.Rodrigues(self.R)[0], self.t, self.K, np.asarray(self.distortion)
            )[0].reshape(-1, 2)
        hp = (points @ self.R.T + self.t) @ self.K.T
        return hp[:, :2] / hp[:, 2:3]

    def plane_point(self, pixel, z: float = 0) -> np.ndarray:
        ray = cv2.undistortPoints(
            np.asarray(pixel, float).reshape(1, 1, 2),
            self.K,
            np.asarray(self.distortion) if self.distortion else None,
        )[0, 0]
        direction = self.R.T @ np.array([*ray, 1])
        center = -self.R.T @ self.t
        if abs(direction[2]) < 1e-8:
            raise ValueError("Луч параллелен полю")
        distance = (z - center[2]) / direction[2]
        if distance <= 0:
            raise ValueError("Пересечение с полем находится за камерой")
        return (center + distance * direction)[:2]

    def to_dict(self):
        return {
            "width": self.width,
            "height": self.height,
            "K": self.K.tolist(),
            "R": self.R.tolist(),
            "t": self.t.tolist(),
            "distortion": list(self.distortion),
            "convention": "world:x-right-y-map-top-z-up; camera:opencv",
        }


def blender_camera(spec: dict, image_size) -> Camera:
    width, height = image_size
    center = np.asarray(spec["position_m"], float)
    forward = np.asarray(spec.get("target_m", [0, 0, 0]), float) - center
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, 0, 1])
    if np.linalg.norm(right) < 1e-10:
        right = np.array([1.0, 0, 0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    r = np.stack([right, down, forward])
    angle = np.deg2rad(spec["roll_deg"])
    c, s = np.cos(angle), np.sin(angle)
    r = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ r
    focal = max(width, height) / (2 * np.tan(np.deg2rad(spec["fov_deg"]) / 2))
    k = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1.0]])
    return Camera(width, height, k, r, -r @ center)


def estimate_camera(k, distortion, image_size, pixels, width_m, height_m, *, world_points=None):
    pixels = np.asarray(pixels, float)
    if pixels.ndim != 2 or pixels.shape[1] != 2 or len(pixels) < 4 or not np.isfinite(pixels).all():
        raise ValueError("Укажите четыре угла поля в порядке углов исходной карты")
    contour = pixels.astype(np.float32)
    if world_points is None and (
        len(pixels) != 4 or not cv2.isContourConvex(contour) or abs(cv2.contourArea(contour)) < 64
    ):
        raise ValueError("Углы поля должны образовывать выпуклый четырёхугольник")
    world = (
        board_corners(width_m, height_m)
        if world_points is None
        else np.asarray(world_points, float)
    )
    if (
        world.shape != (len(pixels), 3)
        or not np.isfinite(world).all()
        or not np.allclose(world[:, 2], 0)
    ):
        raise ValueError("Точки карты должны лежать в плоскости Z=0")
    _, rotations, translations, _ = cv2.solvePnPGeneric(
        world,
        pixels,
        np.asarray(k, float),
        np.asarray(distortion, float) if distortion else None,
        flags=cv2.SOLVEPNP_IPPE,
    )
    candidates = []
    for rv, tv in zip(rotations, translations, strict=True):
        rv, tv = cv2.solvePnPRefineLM(
            world,
            pixels,
            np.asarray(k, float),
            np.asarray(distortion, float) if distortion else None,
            rv.copy(),
            tv.reshape(3, 1).copy(),
        )
        r = cv2.Rodrigues(rv)[0]
        t = tv.ravel()
        if (-r.T @ t)[2] <= 0 or np.any((world @ r.T + t)[:, 2] <= 0):
            continue
        camera = Camera(*image_size, k, r, t, tuple(distortion))
        error = float(np.sqrt(np.mean(np.sum((camera.project(world) - pixels) ** 2, axis=1))))
        candidates.append((error, camera))
    if not candidates:
        raise ValueError("Не удалось восстановить камеру над полем; проверьте порядок углов и K")
    error, camera = min(candidates, key=lambda item: item[0])
    if error > 5:
        raise ValueError(f"Ошибка проекции углов {error:.1f} px; проверьте параметры камеры и поля")
    return camera, {"reprojection_rmse_px": error, "pose_candidates": len(candidates)}

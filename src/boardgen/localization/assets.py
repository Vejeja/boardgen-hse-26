from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import trimesh
from scipy.spatial import ConvexHull


@dataclass(frozen=True)
class FigureGeometry:
    vertices: np.ndarray
    base_xy: np.ndarray
    radius_m: float
    height_m: float
    source: str
    base_method: str

    def placed(self, xy, yaw_deg):
        angle = np.deg2rad(yaw_deg)
        c, s = np.cos(angle), np.sin(angle)
        rotation = np.array([[c, -s], [s, c]])
        points = self.vertices.copy()
        points[:, :2] = points[:, :2] @ rotation.T + xy
        return points

    def base_at(self, xy, yaw_deg):
        angle = np.deg2rad(yaw_deg)
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, -s], [s, c]]) @ self.base_xy + xy

    def metadata(self):
        return {
            "height_m": self.height_m,
            "base_xy_m": self.base_xy.tolist(),
            "radius_m": self.radius_m,
            "base_method": self.base_method,
            "source": self.source,
        }


def load_figure(path: Path, height_m: float, *, base_xy_m=None, radius_m=None) -> FigureGeometry:
    if not np.isfinite(height_m) or height_m <= 0:
        raise ValueError("Высота фигуры должна быть положительной")
    mesh = trimesh.load(path, force="scene", process=False).to_geometry()
    vertices = np.asarray(mesh.vertices, float)
    if not len(mesh.faces) or not np.isfinite(vertices).all():
        raise ValueError(f"Некорректная GLB: {path}")
    vertices = vertices[:, [0, 2, 1]].copy()
    vertices[:, 1] *= -1
    low, high = vertices.min(axis=0), vertices.max(axis=0)
    if high[2] <= low[2]:
        raise ValueError("GLB должна иметь вертикальную ось Y")
    vertices -= [(low[0] + high[0]) / 2, (low[1] + high[1]) / 2, low[2]]
    vertices *= height_m / (high[2] - low[2])
    triangles = vertices[mesh.faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    area = np.linalg.norm(cross, axis=1)
    z = triangles[:, :, 2].mean(axis=1)
    horizontal = (area > 1e-15) & (np.abs(cross[:, 2]) > 0.9 * area) & (z < 0.1 * height_m)
    method = "lower_2pct_fallback"
    ceiling = height_m * 0.02
    if horizontal.any():
        step = height_m * 0.005
        bins = np.floor(z[horizontal] / step).astype(int)
        support = np.bincount(bins, weights=area[horizontal])
        coherent = np.flatnonzero(support >= support.max() * 0.2)
        ceiling = (int(coherent[0]) + 2) * step
        method = "horizontal_support_area"
    lower = vertices[vertices[:, 2] <= ceiling, :2]
    if len(lower) < 3:
        raise ValueError("Недостаточно геометрии основания")
    center, radius = cv2.minEnclosingCircle(lower.astype(np.float32))
    base = np.asarray(center if base_xy_m is None else base_xy_m, float)
    radius = float(radius if radius_m is None else radius_m)
    if base.shape != (2,) or not np.isfinite(base).all() or not np.isfinite(radius) or radius <= 0:
        raise ValueError("Некорректный центр или радиус основания")
    hull = ConvexHull(vertices)
    return FigureGeometry(
        vertices[hull.vertices],
        base,
        radius,
        height_m,
        str(path),
        "override" if base_xy_m is not None or radius_m is not None else method,
    )

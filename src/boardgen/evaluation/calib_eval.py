from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np
import trimesh

from boardgen.camera.camera_model import TurntableCameraModel


class SilhouetteRenderer(Protocol):
    def render(
        self, mesh: trimesh.Trimesh, camera_model: TurntableCameraModel, view_indices: list[int]
    ) -> np.ndarray: ...


@dataclass(frozen=True)
class ViewSplit:
    fit: list[int]
    heldout: list[int]


class OpenCVSilhouetteRenderer:
    backend_name = "opencv-cpu-triangle-rasterizer"

    def render(
        self, mesh: trimesh.Trimesh, camera_model: TurntableCameraModel, view_indices: list[int]
    ) -> np.ndarray:
        height, width = camera_model.image_height, camera_model.image_width
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int32)
        projections = camera_model.projection_matrices()
        output = []
        points_h = np.column_stack([vertices, np.ones(len(vertices))])
        for index in view_indices:
            hp = points_h @ projections[index].T
            depth = hp[:, 2]
            uv = hp[:, :2] / np.maximum(depth[:, None], 1e-12)
            mask = np.zeros((height, width), np.uint8)
            for face in faces:
                if np.all(depth[face] > 1e-8):
                    polygon = np.rint(uv[face]).astype(np.int32)
                    cv2.fillConvexPoly(mask, polygon, 1)
            output.append(mask)
        return np.stack(output) if output else np.zeros((0, height, width), np.uint8)


def silhouette_metrics(target: np.ndarray, rendered: np.ndarray) -> dict[str, float]:
    target = target > 0
    rendered = rendered > 0
    intersection = float(np.logical_and(target, rendered).sum())
    union = float(np.logical_or(target, rendered).sum())
    total = float(target.sum() + rendered.sum())
    distances = _symmetric_boundary_distances(target, rendered)
    return {
        "iou": intersection / union if union else 1.0,
        "dice": 2.0 * intersection / total if total else 1.0,
        "boundary_mean_px": float(np.mean(distances)) if distances.size else 0.0,
        "boundary_p95_px": float(np.percentile(distances, 95)) if distances.size else 0.0,
        "centroid_error_px": _centroid_error(target, rendered),
        "bbox_error_px": _bbox_error(target, rendered),
    }


def evaluate_model(
    masks: np.ndarray,
    mesh: trimesh.Trimesh,
    camera_model: TurntableCameraModel,
    split: ViewSplit,
    renderer: SilhouetteRenderer | None = None,
) -> dict:
    renderer = renderer or OpenCVSilhouetteRenderer()
    result = {}
    for name, indices in (("train", split.fit), ("heldout", split.heldout)):
        rendered = renderer.render(mesh, camera_model, indices)
        per_view = [
            silhouette_metrics(masks[idx], pred)
            for idx, pred in zip(indices, rendered, strict=True)
        ]
        result[name] = _aggregate(per_view)
        result[name]["view_indices"] = indices
        result[name]["per_view"] = per_view
    return result


def write_qc_sheet(
    path: str,
    masks: np.ndarray,
    rendered: np.ndarray,
    indices: list[int],
    label: str,
) -> None:
    cells = []
    for index, pred in zip(indices, rendered, strict=True):
        target = (masks[index] > 0).astype(np.uint8)
        pred = (pred > 0).astype(np.uint8)
        image = np.zeros((*target.shape, 3), np.uint8)
        image[:, :, 1] = target * 180
        image[:, :, 2] = pred * 180
        image[np.logical_and(target, pred)] = (230, 230, 230)
        cv2.putText(
            image, f"{label} v{index}", (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1
        )
        cells.append(image)
    if not cells:
        return
    height = cells[0].shape[0]
    cells = [cv2.resize(cell, (max(1, int(cell.shape[1] * 160 / height)), 160)) for cell in cells]
    cv2.imwrite(path, np.concatenate(cells, axis=1))


def _aggregate(per_view: list[dict[str, float]]) -> dict[str, float]:
    if not per_view:
        return {}
    output = {}
    for key in per_view[0]:
        name = key if key.endswith("_px") else f"{key}_mean"
        output[name] = float(np.mean([row[key] for row in per_view]))
    return output


def _boundary(mask: np.ndarray) -> np.ndarray:
    binary = mask.astype(np.uint8)
    return binary ^ cv2.erode(binary, np.ones((3, 3), np.uint8))


def _symmetric_boundary_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ba, bb = _boundary(a), _boundary(b)
    if not ba.any() or not bb.any():
        return np.array([float(max(a.shape))], dtype=np.float64)
    da = cv2.distanceTransform((1 - ba).astype(np.uint8), cv2.DIST_L2, 3)
    db = cv2.distanceTransform((1 - bb).astype(np.uint8), cv2.DIST_L2, 3)
    return np.concatenate([db[ba > 0], da[bb > 0]]).astype(np.float64)


def _centroid_error(a: np.ndarray, b: np.ndarray) -> float:
    if not a.any() or not b.any():
        return float(max(a.shape))
    ay, ax = np.nonzero(a)
    by, bx = np.nonzero(b)
    return float(np.hypot(ax.mean() - bx.mean(), ay.mean() - by.mean()))


def _bbox_error(a: np.ndarray, b: np.ndarray) -> float:
    def bbox(mask: np.ndarray) -> np.ndarray:
        y, x = np.nonzero(mask)
        return np.array([x.min(), y.min(), x.max(), y.max()], dtype=np.float64)

    if not a.any() or not b.any():
        return float(max(a.shape))
    return float(np.abs(bbox(a) - bbox(b)).mean())

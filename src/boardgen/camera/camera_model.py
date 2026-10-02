from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import yaml

SCHEMA_VERSION = "boardgen.turntable_camera_model.v2"
SUPPORTED_SCHEMA_VERSIONS = {"boardgen.turntable_camera_model.v1", SCHEMA_VERSION}
DISTORTION_MODELS = {"none_assumed", "none_measured", "unknown_unmodeled", "opencv_fixed"}


@dataclass
class CropResizeTransform:
    source_width: int
    source_height: int
    crop_xyxy: tuple[float, float, float, float]
    output_width: int
    output_height: int

    @classmethod
    def identity(cls, width: int, height: int) -> CropResizeTransform:
        return cls(width, height, (0.0, 0.0, float(width), float(height)), width, height)

    @property
    def scale_x(self) -> float:
        x0, _, x1, _ = self.crop_xyxy
        return self.output_width / max(1e-12, x1 - x0)

    @property
    def scale_y(self) -> float:
        _, y0, _, y1 = self.crop_xyxy
        return self.output_height / max(1e-12, y1 - y0)

    def transform_intrinsics(self, intrinsics: CameraIntrinsics) -> CameraIntrinsics:
        self.validate()
        x0, y0, _, _ = self.crop_xyxy
        sx = self.scale_x
        sy = self.scale_y
        return CameraIntrinsics(
            width=int(self.output_width),
            height=int(self.output_height),
            fx=float(intrinsics.fx) * sx,
            fy=float(intrinsics.fy) * sy,
            cx=(float(intrinsics.cx) - float(x0)) * sx,
            cy=(float(intrinsics.cy) - float(y0)) * sy,
            distortion_model=intrinsics.distortion_model,
            distortion=list(intrinsics.distortion),
        )

    def validate(self) -> None:
        if self.source_width <= 0 or self.source_height <= 0:
            raise ValueError("source image size must be positive")
        if self.output_width <= 0 or self.output_height <= 0:
            raise ValueError("output image size must be positive")
        if len(self.crop_xyxy) != 4:
            raise ValueError("crop_xyxy must contain four values")
        x0, y0, x1, y1 = [float(v) for v in self.crop_xyxy]
        vals = np.array([x0, y0, x1, y1], dtype=np.float64)
        if not np.all(np.isfinite(vals)):
            raise ValueError("crop coordinates must be finite")
        if x1 <= x0 or y1 <= y0:
            raise ValueError("crop must have positive size")
        if x0 < 0 or y0 < 0 or x1 > self.source_width or y1 > self.source_height:
            raise ValueError("crop must lie inside source image")
        if self.scale_x <= 0 or self.scale_y <= 0:
            raise ValueError("crop/resize scale must be positive")


@dataclass
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    distortion_model: str = "none_assumed"
    distortion: list[float] = field(default_factory=list)

    @classmethod
    def from_fov(cls, width: int, height: int, fov_deg: float) -> CameraIntrinsics:
        fx = (float(width) / 2.0) / math.tan(math.radians(float(fov_deg)) / 2.0)
        return cls(
            width=int(width),
            height=int(height),
            fx=float(fx),
            fy=float(fx),
            cx=(float(width) - 1.0) / 2.0,
            cy=(float(height) - 1.0) / 2.0,
        )

    def K(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def validate(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("intrinsics image size must be positive")
        vals = np.array([self.fx, self.fy, self.cx, self.cy], dtype=np.float64)
        if not np.all(np.isfinite(vals)):
            raise ValueError("intrinsics must be finite")
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError("fx/fy must be positive")
        if self.distortion_model not in DISTORTION_MODELS:
            raise ValueError(f"unsupported distortion_model {self.distortion_model!r}")
        dist = np.asarray(self.distortion, dtype=np.float64)
        if dist.size and not np.all(np.isfinite(dist)):
            raise ValueError("distortion coefficients must be finite")


@dataclass
class TurntableView:
    frame_index: int
    theta_rad: float
    view_id: str = ""


@dataclass
class TurntableCameraModel:
    image_width: int
    image_height: int
    intrinsics: CameraIntrinsics
    views: list[TurntableView]
    source_clip: str | None = None
    source_image_size: tuple[int, int] | None = None
    axis_direction_world: tuple[float, float, float] = (0.0, 0.0, 1.0)
    axis_origin_world: tuple[float, float, float] = (0.0, 0.0, 0.0)
    camera_distance: float = 1.2
    elevation_deg: float = 0.0
    roll_deg: float = 0.0
    metric_scale: float | None = None
    scale_mode: str = "normalized"
    zero_angle_definition: str = "theta=0 at first sampled frame"
    theta_direction: str = "positive theta follows source frame order around axis_direction_world"
    crop_resize_transform: CropResizeTransform | None = None
    calibration_residual: float | None = None
    confidence: float | None = None
    initialization_method: str = "legacy-harmonic-initializer"
    optimization_method: str = "none"
    version: str = "0.1"
    convention: str = "world:x-right-y-forward-z-up; opencv:x-right-y-down-z-forward"
    status: str = "initial"
    metrics: dict[str, Any] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        axis = np.asarray(self.axis_direction_world, dtype=np.float64)
        if axis.shape == (3,) and np.all(np.isfinite(axis)) and np.linalg.norm(axis) > 1e-12:
            axis /= np.linalg.norm(axis)
        self.axis_direction_world = tuple(float(v) for v in axis)
        self.axis_origin_world = tuple(float(v) for v in self.axis_origin_world)
        if self.source_image_size is not None:
            self.source_image_size = tuple(int(v) for v in self.source_image_size)

    def effective_intrinsics(self) -> CameraIntrinsics:
        if self.crop_resize_transform is None:
            return self.intrinsics
        return self.crop_resize_transform.transform_intrinsics(self.intrinsics)

    def projection_matrices(self) -> list[np.ndarray]:
        self.validate(require_projection=False)
        k = self.effective_intrinsics().K()
        r_cam, t_cam = _look_at_world_to_opencv(
            np.asarray(self.camera_center_world(), dtype=np.float64),
            np.asarray(self.axis_origin_world, dtype=np.float64),
            roll_deg=self.roll_deg,
        )
        origin = np.asarray(self.axis_origin_world, dtype=np.float64)
        axis = _unit(np.asarray(self.axis_direction_world, dtype=np.float64))
        out = []
        for view in self.views:
            r_obj = _rodrigues(axis, -float(view.theta_rad))
            r = r_cam @ r_obj
            t = r_cam @ (origin - r_obj @ origin) + t_cam
            out.append(k @ np.column_stack([r, t]))
        for p in out:
            if not np.all(np.isfinite(p)):
                raise ValueError("projection matrix contains NaN/Inf")
        return out

    def opencv_extrinsics(self) -> list[tuple[np.ndarray, np.ndarray]]:
        p_mats = self.projection_matrices()
        k_inv = np.linalg.inv(self.effective_intrinsics().K())
        return [(k_inv @ p[:, :3], k_inv @ p[:, 3]) for p in p_mats]

    def camera_center_world(self) -> tuple[float, float, float]:
        elev = math.radians(float(self.elevation_deg))
        origin = np.asarray(self.axis_origin_world, dtype=np.float64)
        center = origin + np.array(
            [0.0, -self.camera_distance * math.cos(elev), self.camera_distance * math.sin(elev)],
            dtype=np.float64,
        )
        return (float(center[0]), float(center[1]), float(center[2]))

    def project_points(
        self, points_world: np.ndarray, view_index: int = 0
    ) -> tuple[np.ndarray, np.ndarray]:
        pts = np.asarray(points_world, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError("points_world must have shape (N, 3)")
        if view_index < 0 or view_index >= len(self.views):
            raise IndexError("view_index out of range")
        pts_h = np.column_stack([pts, np.ones(pts.shape[0], dtype=np.float64)])
        hp = pts_h @ self.projection_matrices()[view_index].T
        depth = hp[:, 2]
        uv = hp[:, :2] / np.maximum(depth[:, None], 1e-12)
        return uv, depth

    def validate(self, require_projection: bool = True) -> None:
        if self.schema_version not in SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError(f"unsupported camera model schema {self.schema_version!r}")
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("working image size must be positive")
        self.intrinsics.validate()
        if self.source_image_size is not None:
            if len(self.source_image_size) != 2 or min(self.source_image_size) <= 0:
                raise ValueError("source_image_size must contain positive width/height")
        if self.crop_resize_transform is not None:
            self.crop_resize_transform.validate()
            if (
                self.intrinsics.width != self.crop_resize_transform.source_width
                or self.intrinsics.height != self.crop_resize_transform.source_height
            ):
                raise ValueError("intrinsics size must match crop source size")
            if (
                self.crop_resize_transform.output_width != self.image_width
                or self.crop_resize_transform.output_height != self.image_height
            ):
                raise ValueError("crop output size must match working image size")
            if self.source_image_size is not None and tuple(self.source_image_size) != (
                self.crop_resize_transform.source_width,
                self.crop_resize_transform.source_height,
            ):
                raise ValueError("source_image_size must match crop source size")
        axis = np.asarray(self.axis_direction_world, dtype=np.float64)
        origin = np.asarray(self.axis_origin_world, dtype=np.float64)
        if axis.shape != (3,) or origin.shape != (3,):
            raise ValueError("axis direction/origin must be 3D vectors")
        if not np.all(np.isfinite(axis)) or not np.all(np.isfinite(origin)):
            raise ValueError("axis direction/origin must be finite")
        if np.linalg.norm(axis) <= 1e-12:
            raise ValueError("axis direction must be nonzero")
        vals = np.array(
            [self.camera_distance, self.elevation_deg, self.roll_deg],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(vals)):
            raise ValueError("camera distance/elevation/roll must be finite")
        if self.camera_distance <= 1e-6:
            raise ValueError("camera distance must be positive")
        if not self.views:
            raise ValueError("camera model must contain at least one view")
        frame_indices = [int(v.frame_index) for v in self.views]
        if len(frame_indices) != len(self.views):
            raise ValueError("len(theta) must match len(frame_indices)")
        if len(set(frame_indices)) != len(frame_indices):
            raise ValueError("duplicate frame indices are not allowed")
        theta = np.array([float(v.theta_rad) for v in self.views], dtype=np.float64)
        if not np.all(np.isfinite(theta)):
            raise ValueError("theta values must be finite")
        center = np.asarray(self.camera_center_world(), dtype=np.float64)
        if not np.all(np.isfinite(center)):
            raise ValueError("camera center contains NaN/Inf")
        if require_projection:
            for p in self.projection_matrices():
                if p.shape != (3, 4) or not np.all(np.isfinite(p)):
                    raise ValueError("projection matrices must be finite 3x4 arrays")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        data = asdict(self)
        data["intrinsics"] = asdict(self.intrinsics)
        data["views"] = [asdict(v) for v in self.views]
        data["working_image_size"] = [int(self.image_width), int(self.image_height)]
        if self.crop_resize_transform is not None:
            data["crop_resize_transform"] = asdict(self.crop_resize_transform)
        return _plain(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TurntableCameraModel:
        schema = data.get("schema_version")
        if schema not in SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError(
                f"unsupported camera model schema {schema!r}; expected one of {sorted(SUPPORTED_SCHEMA_VERSIONS)}"
            )
        data = _migrate_v1_to_v2(data) if schema != SCHEMA_VERSION else dict(data)
        data.pop("working_image_size", None)
        intr = CameraIntrinsics(**data["intrinsics"])
        views = [TurntableView(**v) for v in data.get("views", [])]
        crop_data = data.get("crop_resize_transform")
        crop = CropResizeTransform(**crop_data) if crop_data else None
        kwargs = dict(data)
        kwargs["intrinsics"] = intr
        kwargs["views"] = views
        kwargs["crop_resize_transform"] = crop
        out = cls(**kwargs)
        out.validate()
        return out

    def save_yaml(self, path: str) -> None:
        self.validate()
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False, allow_unicode=True)

    @classmethod
    def load_yaml(cls, path: str) -> TurntableCameraModel:
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f) or {})


def _look_at_world_to_opencv(
    center: np.ndarray, target: np.ndarray, roll_deg: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    forward = _unit(target - center)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(forward, world_up))) > 0.98:
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    right = _unit(np.cross(forward, world_up))
    down = _unit(np.cross(forward, right))
    r = np.stack([right, down, forward], axis=0)
    if abs(roll_deg) > 1e-12:
        rr = _roll_camera_matrix(math.radians(float(roll_deg)))
        r = rr @ r
    t = -r @ center
    return r, t


def _roll_camera_matrix(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _rodrigues(axis: np.ndarray, angle: float) -> np.ndarray:
    a = _unit(axis)
    x, y, z = a
    k = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)
    eye = np.eye(3, dtype=np.float64)
    return eye + math.sin(angle) * k + (1.0 - math.cos(angle)) * (k @ k)


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= 1e-12:
        raise ValueError("zero-length vector")
    return v / n


def _migrate_v1_to_v2(data: dict[str, Any]) -> dict[str, Any]:
    out = dict(data)
    out["schema_version"] = SCHEMA_VERSION
    if "method" in out and "initialization_method" not in out:
        out["initialization_method"] = out.pop("method")
    out.setdefault("optimization_method", "none")
    out.setdefault("status", "initial")
    out.setdefault("metrics", {})
    out.setdefault("scale_mode", "normalized")
    out.setdefault(
        "theta_direction", "positive theta follows source frame order around axis_direction_world"
    )
    intr = dict(out.get("intrinsics", {}))
    if intr.get("distortion_model") in (None, "none"):
        intr["distortion_model"] = "none_assumed"
    out["intrinsics"] = intr
    crop = out.get("crop_resize_transform")
    if crop and out.get("source_image_size") is None:
        out["source_image_size"] = [int(crop["source_width"]), int(crop["source_height"])]
    return out


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_plain(v) for v in value]
    if isinstance(value, list):
        return [_plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return copy.deepcopy(value)

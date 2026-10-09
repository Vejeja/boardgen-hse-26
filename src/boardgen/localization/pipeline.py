from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .fitting import fit_figure
from .geometry import estimate_camera
from .registration import prepare_template, register_board

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi"}


@dataclass
class Frame:
    image: np.ndarray
    source: Path
    index: int
    timestamp_s: float | None


def read_frames(source: Path, stride=1, resolve=lambda p: p):
    paths = sorted(source.iterdir()) if source.is_dir() else [source]
    for path in paths:
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES | VIDEO_SUFFIXES:
            continue
        path = resolve(path)
        if path.suffix.lower() in IMAGE_SUFFIXES:
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f"Не читается изображение: {path}")
            yield Frame(image, path, 0, None)
        else:
            capture = cv2.VideoCapture(str(path))
            try:
                if not capture.isOpened():
                    raise ValueError(f"Не открывается видео: {path}")
                fps = capture.get(cv2.CAP_PROP_FPS)
                index = 0
                while True:
                    ok = capture.grab()
                    if not ok:
                        break
                    if index % stride == 0:
                        ok, image = capture.retrieve()
                        if not ok:
                            raise ValueError(f"Не читается кадр {index}: {path}")
                        yield Frame(
                            image,
                            path,
                            index,
                            index / fps if np.isfinite(fps) and fps > 0 else None,
                        )
                    index += 1
            finally:
                capture.release()


def scaled_intrinsics(calibration, image_size):
    width, height = image_size
    if abs(width / height - calibration.width / calibration.height) > 0.005:
        raise ValueError("Пропорции кадра отличаются от калибровки: для crop нужна отдельная K")
    sx, sy = width / calibration.width, height / calibration.height
    return np.array(
        [
            [calibration.fx * sx, 0, calibration.cx * sx],
            [0, calibration.fy * sy, calibration.cy * sy],
            [0, 0, 1.0],
        ]
    )


class Localizer:
    def __init__(self, settings, template, geometries):
        self.settings = settings
        self.template = template
        self.features = prepare_template(template) if settings.registration == "template" else None
        self.geometries = geometries
        self.cached = None
        self.source = None

    def process(self, frame: Frame, detections):
        h, w = frame.image.shape[:2]
        result = {
            "source": str(frame.source),
            "frame_index": frame.index,
            "timestamp_s": frame.timestamp_s,
            "image_size": [w, h],
            "figures": [],
        }
        if self.source != frame.source:
            self.cached = None
            self.source = frame.source
        settings = self.settings
        size = [settings.board.width_m, settings.board.height_m]
        try:
            k = scaled_intrinsics(settings.camera, (w, h))
            if self.cached is not None and settings.fixed_camera:
                camera, quality = self.cached
                if (camera.width, camera.height) != (w, h):
                    raise ValueError("Разрешение изменилось внутри съёмки")
                quality = {**quality, "reused_fixed_camera": True}
            elif settings.registration == "corners":
                camera, quality = estimate_camera(
                    k,
                    settings.camera.distortion,
                    (w, h),
                    np.asarray(settings.corners) * [w, h],
                    *size,
                )
                quality["method"] = "manual_corners"
            else:
                camera, quality = register_board(
                    frame.image,
                    self.template,
                    k,
                    settings.camera.distortion,
                    size,
                    features=self.features,
                )
            if settings.fixed_camera:
                self.cached = (camera, quality)
        except (ValueError, cv2.error) as error:
            result.update(status="registration_failed", error=str(error))
            result["figures"] = [
                {**item, "status": "unavailable", "reasons": ["registration_failed"]}
                for item in detections
            ]
            return result, None
        result.update(status="processed", camera=camera.to_dict(), registration=quality)
        for detection in detections:
            item = {**detection}
            geometry = self.geometries.get(item["class_id"])
            if geometry is None:
                item.update(status="unavailable", reasons=["missing_model"])
            else:
                try:
                    box = np.asarray(item["box"]) * [w, h, w, h]
                    item.update(fit_figure(camera, geometry, box, size))
                except ValueError as error:
                    item.update(status="unavailable", reasons=[str(error)])
            result["figures"].append(item)
        return result, camera

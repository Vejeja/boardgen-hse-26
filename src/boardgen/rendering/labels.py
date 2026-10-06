from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import OpenEXR
from numpy.typing import NDArray

ID_SCALE = 256
MIN_COVERAGE = 0.5
ID_ROUNDING_TOLERANCE = 0.05
MAX_MIXTURE_VARIANCE = 0.2


def yolo_box_from_mask(mask: np.ndarray) -> tuple[float, float, float, float] | None:
    """Вернуть нормированную рамку видимых пикселей или None для пустой маски."""
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return None
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    h, w = mask.shape[:2]
    return ((x0 + x1 + 1) / 2 / w, (y0 + y1 + 1) / 2 / h, (x1 - x0 + 1) / w, (y1 - y0 + 1) / h)


def decode_color_ids(channels: Mapping[str, OpenEXR.Channel]) -> NDArray[np.uint16]:
    """Декодировать ID из прохода с первым и вторым моментами и покрытием.

    Нормировка на покрытие устраняет смешение с фоном. Ненулевая дисперсия
    обнаруживает смесь разных экземпляров; такие пиксели получают фоновый ID 0.
    """
    if not {"R", "G", "B"} <= channels.keys():
        raise ValueError("Expected RGB colour ID pass")
    red, green, coverage = (channels[k].pixels for k in ("R", "G", "B"))
    if not all(np.isfinite(a).all() for a in (red, green, coverage)):
        raise ValueError("Non-finite colour ID pass")
    foreground = coverage >= MIN_COVERAGE
    mean = np.divide(red * ID_SCALE, coverage, out=np.zeros_like(red), where=foreground)
    second = np.divide(green * ID_SCALE**2, coverage, out=np.zeros_like(green), where=foreground)
    rounded = np.rint(mean)
    valid = (
        foreground
        & (np.abs(mean - rounded) < ID_ROUNDING_TOLERANCE)
        & (np.abs(second - mean**2) < MAX_MIXTURE_VARIANCE)
    )
    return np.where(valid, np.clip(rounded, 0, 65535), 0).astype(np.uint16)


def write_yolo_labels_from_id_pass(
    path: Path | str,
    id2cls: Mapping[int, int],
    out: Path | str,
    *,
    image_size: tuple[int, int] | list[int],
) -> None:
    with OpenEXR.File(str(path), separate_channels=True) as image:
        ids = decode_color_ids(image.channels())
    if ids.shape != (image_size[1], image_size[0]):
        raise ValueError("ID pass resolution differs from RGB image")
    unexpected = set(np.unique(ids)) - {0, *id2cls}
    if unexpected:
        raise ValueError(f"Unknown instance IDs in Blender pass: {unexpected}")
    lines = []
    for pid, cls in id2cls.items():
        box = yolo_box_from_mask(ids == pid)
        if box is not None:
            lines.append(f"{cls} " + " ".join(f"{value:.6f}" for value in box) + "\n")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(lines))

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass

import cv2
import numpy as np

from boardgen.common.image_utils import largest_component as _dominant_component
from boardgen.common.logger import get_logger
from boardgen.common.manifest import sha256_file
from boardgen.common.utils import ensure_dir

logger = get_logger()


PROBE_WIDTH = 960

DESC_SIZE = 40


@dataclass
class ClipInfo:
    path: str
    n_frames: int
    fps: float
    width: int
    height: int
    period_frames: float
    period_confidence: float
    revolutions: float

    bbox: tuple[int, int, int, int]
    median_bbox: tuple[int, int, int, int]

    def to_json(self) -> dict:
        d = asdict(self)
        d["bbox"] = list(self.bbox)
        d["median_bbox"] = list(self.median_bbox)
        return d


def _open(path: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"не удалось открыть видео: {path}")
    return cap


def read_probe_stack(path: str, width: int = PROBE_WIDTH) -> tuple[np.ndarray, float]:
    cap = _open(path)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frames = []
    scale = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if scale is None:
            h, w = frame.shape[:2]
            scale = width / float(w)
        small = cv2.resize(
            frame, (width, int(round(frame.shape[0] * scale))), interpolation=cv2.INTER_AREA
        )
        frames.append(small)
    cap.release()
    if not frames:
        raise RuntimeError(f"видео не содержит кадров: {path}")
    return np.stack(frames, 0), fps


def estimate_period(desc: np.ndarray, min_frac: float = 0.05) -> tuple[float, float]:
    n = desc.shape[0]
    p_min = max(2, int(n * min_frac * 0.2))
    p_max = n - 8
    if p_max <= p_min:
        return float(n), 0.0

    d = desc - desc.mean(axis=1, keepdims=True)
    nrm = np.linalg.norm(d, axis=1, keepdims=True)
    d = d / np.maximum(nrm, 1e-6)

    lags = np.arange(p_min, p_max + 1)
    err = np.empty(lags.shape[0], np.float64)
    for i, p in enumerate(lags):
        diff = d[: n - p] - d[p:]
        err[i] = float((diff * diff).sum(axis=1).mean())

    k = max(3, len(err) // 200)
    sm = np.convolve(err, np.ones(2 * k + 1) / (2 * k + 1), mode="same")

    sm[:k] = err[:k]
    sm[-k:] = err[-k:]

    peak = int(np.argmax(sm))
    if peak >= len(sm) - 2:
        return float(n), 0.0
    tail = sm[peak:]
    idx = peak + int(np.argmin(tail))
    if idx >= len(sm) - 2:
        idx = len(sm) - 2

    e_peak = float(sm[peak])
    e_min = float(sm[idx])
    if e_peak <= 1e-12:
        return float(n), 0.0

    period = float(lags[idx])
    if 0 < idx < len(lags) - 1:
        y0, y1, y2 = err[idx - 1], err[idx], err[idx + 1]
        denom = y0 - 2 * y1 + y2
        if abs(denom) > 1e-12:
            period += 0.5 * (y0 - y2) / denom

    confidence = float(1.0 - e_min / e_peak)
    return period, max(0.0, min(1.0, confidence))


def _temporal_median(stack: np.ndarray, chunk_rows: int = 128) -> np.ndarray:
    n, h, w, c = stack.shape
    out = np.empty((h, w, c), np.uint8)
    for y0 in range(0, h, chunk_rows):
        y1 = min(h, y0 + chunk_rows)
        block = stack[:, y0:y1].astype(np.float32)
        out[y0:y1] = np.median(block, axis=0).astype(np.uint8)
    return out


def localize(stack: np.ndarray, lab_thr: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
    bg = _temporal_median(stack)
    lab_bg = cv2.cvtColor(bg, cv2.COLOR_BGR2LAB).astype(np.float32)
    n = stack.shape[0]
    cover = np.zeros(stack.shape[1:3], np.float32)
    dmax = np.zeros(stack.shape[1:3], np.float32)
    for i in range(n):
        lab = cv2.cvtColor(stack[i], cv2.COLOR_BGR2LAB).astype(np.float32)
        d = np.linalg.norm(lab - lab_bg, axis=2)
        cover += (d > lab_thr).astype(np.float32)
        np.maximum(dmax, d, out=dmax)
    cover /= float(n)
    return cover, dmax


def probe(path: str, cache_dir: str = "cache/turntable") -> ClipInfo:
    name = os.path.splitext(os.path.basename(path))[0]
    ensure_dir(cache_dir)
    cache_path = os.path.join(cache_dir, f"{name}_{sha256_file(path)}_probe.json")
    if os.path.exists(cache_path):
        try:
            src_mtime = os.path.getmtime(path)
            if os.path.getmtime(cache_path) >= src_mtime:
                with open(cache_path) as f:
                    d = json.load(f)
                d["bbox"] = tuple(d["bbox"])
                d["median_bbox"] = tuple(d["median_bbox"])
                d["path"] = path
                logger.info(f"[probe] {name}: из кэша")
                return ClipInfo(**d)
        except Exception:
            pass

    stack, fps = read_probe_stack(path)
    n, ph, pw = stack.shape[:3]
    cap = _open(path)
    full_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    full_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    up = full_w / float(pw)

    cover, dmax = localize(stack)

    hi = float(np.percentile(dmax, 99.5))
    strong_thr = max(18.0, 0.35 * hi)
    moving = ((dmax > strong_thr) & (cover > 0.05) & (cover < 0.995)).astype(np.uint8)
    moving = cv2.morphologyEx(moving, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    moving = cv2.morphologyEx(moving, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    comp = _dominant_component(moving)
    if comp is None:
        raise RuntimeError(f"{name}: не удалось локализовать вращающийся объект")

    core = ((cover > 0.85) & (comp > 0)).astype(np.uint8)
    if core.sum() < 20:
        core = comp

    ys, xs = np.where(comp > 0)
    px0, px1 = int(xs.min()), int(xs.max())
    py0, py1 = int(ys.min()), int(ys.max())

    cx0 = max(0, px0 - 8)
    cx1 = min(pw, px1 + 8)
    cy0 = max(0, py0 - 8)
    cy1 = min(ph, py1 + 8)
    desc = np.empty((n, DESC_SIZE * DESC_SIZE), np.float32)
    for i in range(n):
        g = cv2.cvtColor(stack[i, cy0:cy1, cx0:cx1], cv2.COLOR_BGR2GRAY)
        g = cv2.GaussianBlur(g, (5, 5), 0)
        desc[i] = cv2.resize(g, (DESC_SIZE, DESC_SIZE), interpolation=cv2.INTER_AREA).ravel()

    period, conf = estimate_period(desc)

    lab_bg = cv2.cvtColor(_temporal_median(stack), cv2.COLOR_BGR2LAB).astype(np.float32)
    widths, heights = [], []
    for i in range(0, n, max(1, n // 48)):
        lab = cv2.cvtColor(stack[i], cv2.COLOR_BGR2LAB).astype(np.float32)
        d = (np.linalg.norm(lab - lab_bg, axis=2) > 10.0).astype(np.uint8)
        d = cv2.morphologyEx(d, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        d[comp == 0] = 0
        c = _dominant_component(d)
        if c is None:
            continue
        yy, xx = np.where(c > 0)
        widths.append(xx.max() - xx.min() + 1)
        heights.append(yy.max() - yy.min() + 1)
    mw = int(np.median(widths)) if widths else (px1 - px0 + 1)
    mh = int(np.median(heights)) if heights else (py1 - py0 + 1)

    info = ClipInfo(
        path=path,
        n_frames=n,
        fps=fps,
        width=full_w,
        height=full_h,
        period_frames=period,
        period_confidence=conf,
        revolutions=n / period if period > 0 else 0.0,
        bbox=(
            int(round(px0 * up)),
            int(round(py0 * up)),
            int(round((px1 + 1) * up)),
            int(round((py1 + 1) * up)),
        ),
        median_bbox=(0, 0, int(round(mw * up)), int(round(mh * up))),
    )
    with open(cache_path, "w") as f:
        json.dump(info.to_json(), f, indent=1)
    logger.info(
        f"[probe] {name}: {n} кадров, период {period:.1f} ({info.revolutions:.3f} об., "
        f"уверенность {conf:.2f}), объект {info.median_bbox[2]}x{info.median_bbox[3]}px"
    )
    return info


def sample_view_indices(info: ClipInfo, n_views: int) -> tuple[np.ndarray, np.ndarray]:
    period = float(info.period_frames)
    idx = np.arange(n_views, dtype=np.float64) * (period / float(n_views))
    want = np.clip(np.round(idx).astype(int), 0, info.n_frames - 1)
    angles = (want.astype(np.float64) / period) * 2.0 * math.pi
    return want, angles

from __future__ import annotations

import json
import math
import os

import cv2
import numpy as np

from boardgen.common.image_utils import largest_component as _dominant
from boardgen.common.logger import get_logger
from boardgen.common.utils import ensure_dir

logger = get_logger()


THR_K = 3.2
THR_FLOOR = 4.5


SHADOW_MAX_DL = 16.0
SHADOW_MAX_DC = 5.0

SHADOW_MIN_RATIO = 0.55
SHADOW_MAX_RATIO = 0.96

SHADOW_PENALTY = 3.0

INPAINT_PENALTY = 2.6


GAP_THIN_FRAC = 0.0

ENVELOPE_GAIN = 0.70
OUTSIDE_GAIN = 1.60


def _smooth_inpaint(plate: np.ndarray, hole: np.ndarray, scale: int = 8) -> np.ndarray:
    h, w = plate.shape[:2]
    sw, sh = max(8, w // scale), max(8, h // scale)
    small = cv2.resize(plate, (sw, sh), interpolation=cv2.INTER_AREA)

    hole_s = cv2.resize(hole.astype(np.float32), (sw, sh), interpolation=cv2.INTER_AREA)
    hole_s = (hole_s > 0.02).astype(np.uint8)
    if hole_s.sum() >= hole_s.size * 0.98:
        return plate
    filled = cv2.inpaint(small, hole_s, 5, cv2.INPAINT_TELEA)
    filled = cv2.GaussianBlur(filled, (0, 0), 1.2)
    up = cv2.resize(filled, (w, h), interpolation=cv2.INTER_LINEAR)
    out = plate.copy()
    m = hole > 0
    out[m] = up[m]
    return out


def _temporal_std(frames: np.ndarray, chunk_rows: int = 64) -> np.ndarray:
    v, h, w = frames.shape[:3]
    out = np.empty((h, w), np.float32)
    for r0 in range(0, h, chunk_rows):
        r1 = min(h, r0 + chunk_rows)
        lab = np.empty((v, r1 - r0, w, 3), np.float32)
        for i in range(v):
            lab[i] = cv2.cvtColor(frames[i, r0:r1], cv2.COLOR_BGR2LAB).astype(np.float32)
        out[r0:r1] = lab.std(axis=0).mean(axis=2)
    return out


def _plate_and_noise(frames: np.ndarray, chunk_rows: int = 64) -> tuple[np.ndarray, np.ndarray]:
    v, h, w = frames.shape[:3]
    plate = np.empty((h, w, 3), np.uint8)
    sigma = np.empty((h, w), np.float32)
    for r0 in range(0, h, chunk_rows):
        r1 = min(h, r0 + chunk_rows)
        lab = np.empty((v, r1 - r0, w, 3), np.float32)
        for i in range(v):
            lab[i] = cv2.cvtColor(frames[i, r0:r1], cv2.COLOR_BGR2LAB).astype(np.float32)
        med = np.median(lab, axis=0)
        q1 = np.percentile(lab, 25, axis=0)
        q3 = np.percentile(lab, 75, axis=0)
        iqr = np.linalg.norm(q3 - q1, axis=2) / 1.349
        plate[r0:r1] = cv2.cvtColor(med.astype(np.uint8), cv2.COLOR_LAB2BGR)
        sigma[r0:r1] = iqr
    np.clip(sigma, 1.2, 8.0, out=sigma)
    return plate, sigma


def _grad_mag(l_channel: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(l_channel, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(l_channel, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.GaussianBlur(np.sqrt(gx * gx + gy * gy), (0, 0), 1.5)


def _residual(
    lab: np.ndarray, lab_plate: np.ndarray, grad_plate: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    diff = lab - lab_plate
    dl = diff[:, :, 0]
    dc = np.linalg.norm(diff[:, :, 1:], axis=2)
    dist = np.linalg.norm(diff, axis=2)
    ratio = lab[:, :, 0] / np.maximum(lab_plate[:, :, 0], 1e-3)
    grad_f = _grad_mag(lab[:, :, 0])

    no_new_structure = grad_f <= grad_plate * 1.25 + 4.0
    shadow = (
        (dl < 0)
        & (dc < SHADOW_MAX_DC)
        & (ratio > SHADOW_MIN_RATIO)
        & (ratio < SHADOW_MAX_RATIO)
        & no_new_structure
    )

    shadow |= (dl < 0) & (np.abs(dl) < SHADOW_MAX_DL) & (dc < SHADOW_MAX_DC)
    return dist, shadow


def _clean(mask: np.ndarray, min_area: int) -> np.ndarray:
    m = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return m
    out = np.zeros_like(m)
    for k in range(1, n):
        if stats[k, cv2.CC_STAT_AREA] >= min_area:
            out[lab == k] = 1
    return out


def _keep_object(mask: np.ndarray, anchor: np.ndarray, min_area: int) -> np.ndarray:
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return mask
    keep = np.zeros(n, bool)
    for k in range(1, n):
        if stats[k, cv2.CC_STAT_AREA] < min_area:
            continue
        if np.any(anchor[lab == k]):
            keep[k] = True
    if not keep.any():
        k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        keep[k] = True

    accepted = np.isin(lab, np.where(keep)[0]).astype(np.uint8)
    near = cv2.dilate(accepted, np.ones((15, 15), np.uint8))
    for k in range(1, n):
        if keep[k] or stats[k, cv2.CC_STAT_AREA] < min_area:
            continue
        if np.any(near[lab == k]):
            keep[k] = True
    return np.isin(lab, np.where(keep)[0]).astype(np.uint8)


def _refine_grabcut(
    frame: np.ndarray,
    dist: np.ndarray,
    thr: np.ndarray,
    envelope: np.ndarray,
    trust: np.ndarray,
    max_width: int = 1100,
) -> np.ndarray:
    h, w = dist.shape
    scale = min(1.0, max_width / float(max(h, w)))
    if scale < 1.0:
        sw, sh = int(round(w * scale)), int(round(h * scale))
        img_s = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA)
        d_s = cv2.resize(dist, (sw, sh), interpolation=cv2.INTER_AREA)
        t_s = cv2.resize(thr, (sw, sh), interpolation=cv2.INTER_AREA)
        env_s = cv2.resize(envelope, (sw, sh), interpolation=cv2.INTER_NEAREST)
        tr_s = cv2.resize(trust, (sw, sh), interpolation=cv2.INTER_NEAREST)
    else:
        img_s, d_s, t_s, env_s, tr_s = frame, dist, thr, envelope, trust

    ratio = d_s / np.maximum(t_s, 1e-3)

    ratio = np.where(tr_s > 0, ratio, 0.0)
    trimap = np.full(d_s.shape, cv2.GC_PR_BGD, np.uint8)
    trimap[(ratio > 0.8) & (env_s > 0)] = cv2.GC_PR_FGD
    sure_fg = (ratio > 2.2) & (env_s > 0)
    sure_fg = cv2.erode(sure_fg.astype(np.uint8), np.ones((3, 3), np.uint8))
    trimap[sure_fg > 0] = cv2.GC_FGD
    far_bg = cv2.dilate(env_s, np.ones((25, 25), np.uint8), iterations=2) == 0
    trimap[far_bg] = cv2.GC_BGD

    trimap[(ratio < 0.25) & (env_s == 0)] = cv2.GC_BGD

    if not np.any((trimap == cv2.GC_FGD) | (trimap == cv2.GC_PR_FGD)):
        return np.zeros((h, w), np.uint8)
    if not np.any((trimap == cv2.GC_BGD) | (trimap == cv2.GC_PR_BGD)):
        return np.ones((h, w), np.uint8)

    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(img_s, trimap, None, bgd, fgd, 3, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return ((ratio > 1.0) & (env_s > 0)).astype(np.uint8)
    out = ((trimap == cv2.GC_FGD) | (trimap == cv2.GC_PR_FGD)).astype(np.uint8)
    if scale < 1.0:
        out = cv2.resize(out, (w, h), interpolation=cv2.INTER_NEAREST)
    return out


def segment_frames(frames, info, crop, angles, frame_indices, path, name, out_dir=None):
    v, h, w = frames.shape[:3]
    logger.info(f"[segment] {name}: {v} видов, кроп {w}x{h} (нативное разрешение)")

    plate, sigma = _plate_and_noise(frames)

    tstd = _temporal_std(frames)
    bg_level = float(np.percentile(tstd, 40))
    env_thr = max(bg_level * 3.0 + 0.5, float(np.percentile(tstd, 99.5)) * 0.06)
    envelope = (tstd > env_thr).astype(np.uint8)

    lab_plate0 = cv2.cvtColor(plate, cv2.COLOR_BGR2LAB).astype(np.float32)
    grad0 = _grad_mag(lab_plate0[:, :, 0])
    thr_pre = np.maximum(THR_FLOOR, THR_K * sigma)
    always = np.zeros((h, w), np.float32)
    for i in range(v):
        lab = cv2.cvtColor(frames[i], cv2.COLOR_BGR2LAB).astype(np.float32)
        d, _ = _residual(lab, lab_plate0, grad0)
        always += (d > thr_pre).astype(np.float32)
    always /= float(v)
    envelope |= (always > 0.85).astype(np.uint8)

    envelope = cv2.morphologyEx(envelope, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    envelope = cv2.morphologyEx(envelope, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    comp = _dominant(envelope)
    if comp is not None:
        envelope = comp

    ff = envelope.copy()
    cv2.floodFill(ff, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 1)
    envelope |= 1 - ff
    envelope = cv2.dilate(envelope, np.ones((5, 5), np.uint8))
    logger.info(
        f"[segment] {name}: оболочка {100.0 * envelope.mean():.1f}% кропа, "
        f"контраст дисперсии {float(np.median(tstd[envelope > 0])) / max(bg_level, 1e-6):.1f}x"
    )

    lab_plate = cv2.cvtColor(plate, cv2.COLOR_BGR2LAB).astype(np.float32)
    grad_plate = _grad_mag(lab_plate[:, :, 0])
    thr0 = np.maximum(THR_FLOOR, THR_K * sigma)
    cover = np.zeros((h, w), np.float32)
    for i in range(v):
        lab = cv2.cvtColor(frames[i], cv2.COLOR_BGR2LAB).astype(np.float32)
        dist, _ = _residual(lab, lab_plate, grad_plate)
        cover += (dist > thr0).astype(np.float32)
    cover /= float(v)

    contaminated = ((cover > 0.55) | ((envelope > 0) & (cover < 0.05))).astype(np.uint8)
    contaminated = cv2.morphologyEx(contaminated, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    contaminated = cv2.dilate(contaminated, np.ones((7, 7), np.uint8), iterations=2)
    contaminated[envelope == 0] = 0
    if contaminated.any():
        plate = _smooth_inpaint(plate, contaminated)
        lab_plate = cv2.cvtColor(plate, cv2.COLOR_BGR2LAB).astype(np.float32)
        grad_plate = _grad_mag(lab_plate[:, :, 0])
        clean_sigma = sigma[(contaminated == 0) & (envelope == 0)]
        med_sigma = float(np.median(clean_sigma)) if clean_sigma.size else 3.0

        sigma = np.where(contaminated > 0, med_sigma * INPAINT_PENALTY, sigma).astype(np.float32)
    logger.info(f"[segment] {name}: достроено {100.0 * contaminated.mean():.1f}% пластины фона")

    thr = np.maximum(THR_FLOOR, THR_K * sigma)
    gain = np.where(envelope > 0, ENVELOPE_GAIN, OUTSIDE_GAIN).astype(np.float32)
    thr = thr * gain

    trust = (contaminated == 0).astype(np.uint8)

    anchor = ((cover > 0.80) | (contaminated > 0)).astype(np.uint8)
    if anchor.sum() < 30:
        anchor = (cover > 0.5).astype(np.uint8)
    anchor = cv2.dilate(anchor, np.ones((9, 9), np.uint8))

    min_area = max(24, int(0.00002 * h * w))
    masks = np.zeros((v, h, w), np.uint8)
    alphas = np.zeros((v, h, w), np.uint8)
    for i in range(v):
        lab = cv2.cvtColor(frames[i], cv2.COLOR_BGR2LAB).astype(np.float32)
        dist, shadow = _residual(lab, lab_plate, grad_plate)
        thr_i = thr * np.where(shadow, SHADOW_PENALTY, 1.0).astype(np.float32)
        hard = _refine_grabcut(frames[i], dist, thr_i, envelope, trust)
        hard[envelope == 0] = 0

        hard |= ((dist > thr_i * 2.2) & (envelope > 0) & (trust > 0)).astype(np.uint8)

        if GAP_THIN_FRAC > 0:
            dt = cv2.distanceTransform(hard, cv2.DIST_L2, 3)
            thin = dt < max(2.0, GAP_THIN_FRAC * math.sqrt(max(1.0, float(hard.sum()))))
            leak = (trust > 0) & (dist < thr_i * 0.30) & thin & (hard > 0)
            hard[leak] = 0
        hard = _clean(hard, min_area)
        hard = _keep_object(hard, anchor, min_area)

        core = cv2.erode(hard, np.ones((3, 3), np.uint8))
        feather = cv2.GaussianBlur(hard.astype(np.float32), (0, 0), 1.1)
        soft = np.where(core > 0, 1.0, feather).astype(np.float32)
        soft[hard == 0] = np.minimum(soft[hard == 0], 0.5 * feather[hard == 0])
        masks[i] = hard
        alphas[i] = np.clip(soft * 255.0, 0, 255).astype(np.uint8)

    area = masks.reshape(v, -1).mean(1)

    rel = np.abs(np.diff(area)) / np.maximum(area[:-1], 1e-9)
    jitter = float(np.median(rel))
    worst = float(np.max(rel)) if rel.size else 0.0

    wrap = float(abs(area[0] - area[-1]) / max(area[-1], 1e-9))
    logger.info(
        f"[segment] {name}: доля кропа {area.min():.4f}..{area.max():.4f} "
        f"(разброс {area.max() / max(area.min(), 1e-9):.2f}x, "
        f"дёрганье med {100 * jitter:.1f}% max {100 * worst:.1f}%, "
        f"стык оборота {100 * wrap:.1f}%)"
    )

    meta = {
        "name": name,
        "video": path,
        "crop": list(crop),
        "n_views": int(v),
        "angles_rad": [float(a) for a in angles],
        "frame_indices": [int(i) for i in frame_indices[:v]],
        "period_frames": info.period_frames,
        "period_confidence": info.period_confidence,
        "revolutions": info.revolutions,
        "area_min": float(area.min()),
        "area_max": float(area.max()),
        "area_spread": float(area.max() / max(area.min(), 1e-9)),
        "jitter": jitter,
        "jitter_max": worst,
        "wrap_gap": wrap,
        "full_size": [info.width, info.height],
    }

    if out_dir:
        ensure_dir(out_dir)
        for i in range(v):
            rgba = cv2.cvtColor(frames[i], cv2.COLOR_BGR2BGRA)
            rgba[:, :, 3] = alphas[i]
            cv2.imwrite(os.path.join(out_dir, f"view_{i:04d}.png"), rgba)
        cv2.imwrite(os.path.join(out_dir, "_plate.png"), plate)
        with open(os.path.join(out_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=1)

    meta["_frames"] = frames
    meta["_masks"] = masks
    meta["_alphas"] = alphas
    meta["_plate"] = plate
    meta["_envelope"] = envelope
    return meta

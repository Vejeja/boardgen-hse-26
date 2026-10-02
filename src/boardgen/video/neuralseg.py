from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from boardgen.common.logger import get_logger

logger = get_logger()


DEFAULT_MODEL = "ZhengPeng7/BiRefNet_HR-matting"
FALLBACK_MODEL = "ZhengPeng7/BiRefNet"


SIZE_LADDER = (1024, 1536, 2048)


def size_for_crop(h: int, w: int) -> int:
    long_side = max(h, w)
    for s in SIZE_LADDER:
        if long_side <= s * 1.05:
            return s
    return SIZE_LADDER[-1]


_MODEL: Any = None
_DEVICE: str | None = None
_MODEL_NAME: str | None = None


def _pick_device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model(name: str = DEFAULT_MODEL):
    global _MODEL, _DEVICE, _MODEL_NAME
    if _MODEL is not None:
        return _MODEL, _DEVICE
    from transformers import AutoModelForImageSegmentation

    _DEVICE = _pick_device()
    logger.info(f"[neural] загружаю {name} на {_DEVICE}")
    try:
        model = AutoModelForImageSegmentation.from_pretrained(name, trust_remote_code=True)
        _MODEL_NAME = name
    except Exception as exc:
        logger.warning(f"[neural] {name} недоступна ({exc}); беру {FALLBACK_MODEL}")
        model = AutoModelForImageSegmentation.from_pretrained(
            FALLBACK_MODEL, trust_remote_code=True
        )
        _MODEL_NAME = FALLBACK_MODEL
    model.eval()

    if _DEVICE in ("mps", "cuda"):
        try:
            model = model.half()
        except Exception:
            pass
    model = model.to(_DEVICE)
    _MODEL = model
    return _MODEL, _DEVICE


def model_metadata() -> dict:
    from pathlib import Path

    from huggingface_hub import try_to_load_from_cache

    revision = getattr(getattr(_MODEL, "config", None), "_commit_hash", None)
    if not revision and _MODEL_NAME:
        config_path = try_to_load_from_cache(_MODEL_NAME, "config.json")
        if isinstance(config_path, str):
            parts = Path(config_path).parts
            if "snapshots" in parts:
                revision = parts[parts.index("snapshots") + 1]
    return {"model": _MODEL_NAME, "revision": revision, "device": _DEVICE}


def _preprocess(bgr: np.ndarray, device: str, half: bool, size: int):
    import torch

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    interp = cv2.INTER_AREA if max(bgr.shape[:2]) > size else cv2.INTER_CUBIC
    img = cv2.resize(rgb, (size, size), interpolation=interp)
    x = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    x = (x - mean) / std
    x = x.unsqueeze(0).to(device)
    return x.half() if half else x


def matte_batch(frames: np.ndarray, batch: int = 4, size: int | None = None) -> np.ndarray:
    import torch

    model, device = load_model()
    half = device in ("mps", "cuda")
    v, h, w = frames.shape[:3]
    size = size or size_for_crop(h, w)
    out = np.zeros((v, h, w), np.uint8)
    with torch.no_grad():
        for i0 in range(0, v, batch):
            i1 = min(v, i0 + batch)
            xs = torch.cat([_preprocess(frames[i], device, half, size) for i in range(i0, i1)], 0)
            preds = model(xs)

            if isinstance(preds, (list, tuple)):
                pred = preds[-1]
            else:
                pred = preds
            if isinstance(pred, (list, tuple)):
                pred = pred[-1]
            pred = pred.sigmoid().float().cpu().numpy()
            for k in range(i1 - i0):
                a = pred[k, 0]
                a = cv2.resize(a, (w, h), interpolation=cv2.INTER_LINEAR)
                out[i0 + k] = np.clip(a * 255.0, 0, 255).astype(np.uint8)
    return out


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum())
    return inter / union if union > 0 else 0.0


def recover_warm_foreground(bgr: np.ndarray, alpha: np.ndarray) -> tuple[np.ndarray, dict]:
    h, w = alpha.shape
    band = max(1, round(min(h, w) * 0.01))
    yellow = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[:, :, 2].astype(np.float32)
    border = np.concatenate(
        [
            yellow[:band].ravel(),
            yellow[-band:].ravel(),
            yellow[:, :band].ravel(),
            yellow[:, -band:].ravel(),
        ]
    )
    background = float(np.percentile(border, 99))
    seeds = yellow[alpha >= 200]
    contrast = float(np.median(seeds) - background) if seeds.size else 0.0
    metadata = {"background_b_p99": background, "foreground_b_contrast": contrast, "applied": False}
    if contrast < 6:
        return alpha.copy(), metadata
    recovered = np.clip((yellow - background - 1) / 3, 0, 1)

    count, labels, stats, _ = cv2.connectedComponentsWithStats((recovered > 0).astype(np.uint8))
    minimum = max(4, round(h * w * 0.0001))
    keep = np.zeros(count, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= minimum
    recovered[~keep[labels]] = 0
    result = np.maximum(alpha, (recovered * 255).astype(np.uint8))
    metadata.update(applied=True, added_area=float(((result >= 128) & (alpha < 128)).mean()))
    return result, metadata


def cross_check(neural_alpha: np.ndarray, classic_masks: np.ndarray, thr: int = 128) -> dict:
    v = min(neural_alpha.shape[0], classic_masks.shape[0])
    ious = np.array(
        [iou(neural_alpha[i] > thr, classic_masks[i] > 0) for i in range(v)], dtype=float
    )
    n_area = np.array([(neural_alpha[i] > thr).mean() for i in range(v)])
    c_area = np.array([(classic_masks[i] > 0).mean() for i in range(v)])
    with np.errstate(divide="ignore", invalid="ignore"):
        bias = float(np.median(n_area / np.maximum(c_area, 1e-9)))
    return {
        "iou_median": float(np.median(ious)),
        "iou_min": float(ious.min()) if ious.size else 0.0,
        "iou_p10": float(np.percentile(ious, 10)) if ious.size else 0.0,
        "area_ratio_neural_over_classic": bias,
        "n_views": int(v),
        "worst_views": [int(i) for i in np.argsort(ious)[:3]],
    }


def unload_model():
    global _MODEL, _DEVICE, _MODEL_NAME
    previous = _DEVICE
    _MODEL = _DEVICE = _MODEL_NAME = None
    import gc

    gc.collect()
    import torch

    if previous == "mps":
        torch.mps.empty_cache()
    elif previous == "cuda":
        torch.cuda.empty_cache()

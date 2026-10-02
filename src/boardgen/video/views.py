from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from boardgen.common.manifest import source_provenance, write_json
from boardgen.common.utils import file_sha256 as file_hash


def prepare_views(
    root: Path,
    video: Path,
    size: int = 512,
    views: int = 40,
    full_clip: bool = False,
    *,
    matte_method: str = "birefnet",
    mask_size: int = 1024,
    cross_check: bool = False,
    recover_warm: bool = False,
    crop_padding: float = 0.08,
) -> None:
    """Подготовить кадры, мягкие маски и метаданные ракурсов видеозаписи.

    Каждый пятый вид откладывается для проверки реконструкции; остальные входят
    в train. Порядок исходных кадров и SHA видео записываются в dataset.json.
    """
    from boardgen.video import neuralseg

    if root.exists():
        raise ValueError(f"dataset already exists: {root}")
    if views < 10 or views % 5:
        raise ValueError("views must be a multiple of 5, at least 10")
    from boardgen.video.turntable import probe

    source_hash = file_hash(video)
    info = probe(str(video.resolve()), cache_dir=f"cache/benchmark3d/{source_hash}").to_json()
    x0, y0, x1, y1 = info["bbox"]
    if not 0 <= crop_padding <= 1:
        raise ValueError("crop padding must be between 0 and 1")
    pad = round(max(x1 - x0, y1 - y0) * crop_padding)
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(info["width"], x1 + pad), min(info["height"], y1 + pad)
    last_frame = info["n_frames"] - 1
    if not full_clip:
        last_frame = min(last_frame, info["period_frames"] - 1)
    indices = np.rint(np.linspace(0, last_frame, views)).astype(int)
    if len(set(indices)) != views:
        raise ValueError("duplicate source frames")
    root.mkdir(parents=True)
    for sub in [
        "reference/images",
        "reference/masks",
        "train/images",
        "train/masks",
        "train/feature_masks",
    ]:
        (root / sub).mkdir(parents=True)
    model, device = neuralseg.load_model() if matte_method == "birefnet" else (None, "cpu")
    if device == "mps":
        import torch

        torch.mps.set_per_process_memory_fraction(0.8)
    classic = None
    if matte_method == "classic" or cross_check:
        from boardgen.video.segment import segment_frames
        from boardgen.video.turntable import ClipInfo

        stack = []
        capture = cv2.VideoCapture(str(video))
        try:
            for index in indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                ok, f = capture.read()
                if not ok:
                    raise ValueError("classical frame decode failed")
                stack.append(f[y0:y1, x0:x1])
        finally:
            capture.release()
        classic = segment_frames(
            np.array(stack),
            ClipInfo(**info),
            (x0, y0, x1, y1),
            np.linspace(0, 2 * np.pi, views),
            indices,
            str(video),
            video.stem,
        )
    output_size = (
        round((x1 - x0) * min(1, size / max(x1 - x0, y1 - y0))),
        round((y1 - y0) * min(1, size / max(x1 - x0, y1 - y0))),
    )
    fit = [i for i in range(views) if i % 5 != 4]
    heldout = [i for i in range(views) if i % 5 == 4]
    cap = cv2.VideoCapture(str(video))
    diagnostics, tiles = [], []
    started = time.perf_counter()
    try:
        for i, index in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f"could not decode source frame {index}")
            crop = frame[y0:y1, x0:x1]
            alpha = (
                neuralseg.matte_batch(crop[None], batch=1, size=mask_size)[0]
                if matte_method == "birefnet"
                else classic["_alphas"][i]
            )
            if device == "mps":
                torch.mps.empty_cache()
            recovery = None
            if recover_warm:
                alpha, recovery = neuralseg.recover_warm_foreground(crop, alpha)
            frame = cv2.resize(crop, output_size, interpolation=cv2.INTER_AREA)
            alpha = cv2.resize(alpha, output_size, interpolation=cv2.INTER_AREA)
            name = f"view_{i:04d}.png"
            cv2.imwrite(str(root / "reference/images" / name), frame)
            cv2.imwrite(str(root / "reference/masks" / name), alpha)
            if i in fit:
                cv2.imwrite(str(root / "train/images" / name), np.dstack([frame, alpha]))
                cv2.imwrite(str(root / "train/masks" / name), alpha)
                cv2.imwrite(
                    str(root / "train/feature_masks" / name), (alpha >= 128).astype(np.uint8) * 255
                )
            hard = alpha >= 128
            check_iou = None
            if cross_check and matte_method == "birefnet":
                comparison = (
                    cv2.resize(classic["_masks"][i], output_size, interpolation=cv2.INTER_NEAREST)
                    > 0
                )
                check_iou = neuralseg.iou(hard, comparison)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sharpness = float(cv2.Laplacian(gray, cv2.CV_32F)[hard].var()) if hard.any() else 0
            diagnostics.append(
                {
                    "view": i,
                    "source_frame": int(index),
                    "cross_check_iou": check_iou,
                    "warm_recovery": recovery,
                    "area": float(hard.mean()),
                    "sharpness": sharpness,
                    "empty": not bool(hard.any()),
                    "touches_border": bool(
                        hard[0].any() or hard[-1].any() or hard[:, 0].any() or hard[:, -1].any()
                    ),
                }
            )
            checker = np.indices(alpha.shape).sum(axis=0) // 16 % 2
            bg = np.where(checker[:, :, None], 180, 230)
            tile = (frame * (alpha[:, :, None] / 255) + bg * (1 - alpha[:, :, None] / 255)).astype(
                np.uint8
            )
            tile = cv2.resize(tile, (128, 128))
            cv2.putText(
                tile,
                f"{i} {'TEST' if i in heldout else 'fit'}",
                (4, 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 0),
                1,
            )
            tiles.append(tile)
            print(f"neural {i + 1}/{views}, {time.perf_counter() - started:.1f}s", flush=True)
    finally:
        cap.release()
    while len(tiles) % 8:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(
        str(root / "contact.png"),
        np.vstack([np.hstack(tiles[i : i + 8]) for i in range(0, len(tiles), 8)]),
    )
    write_json(
        root / "dataset.json",
        {
            "schema": "boardgen.benchmark3d.v1",
            "kind": "real-turntable",
            "fit": fit,
            "heldout": heldout,
            "source_video": str(video),
            "source_video_sha256": source_hash,
            "source_frame_indices": indices.tolist(),
            "crop_xyxy": [x0, y0, x1, y1],
            "crop_padding": crop_padding,
            "output_size": list(output_size),
            "mask_model": neuralseg.model_metadata()["model"]
            if model is not None
            else "classical-static-background",
            "mask_revision": neuralseg.model_metadata()["revision"] if model is not None else None,
            "device": device,
            "mask_input_size": mask_size if model is not None else None,
            "mps_memory_fraction": 0.8 if device == "mps" else None,
            "mps_cache_policy": "release after each frame" if device == "mps" else None,
            "mask_postprocessing": "warm-chroma recovery; resampling"
            if recover_warm
            else "none; resampling only",
            "diagnostics": diagnostics,
            "period_frames": info["period_frames"],
            "period_confidence": info["period_confidence"],
            "sampling": "full clip including repeated views"
            if full_clip
            else "estimated first orbit",
            "cameras": "not calibrated; requires fit-only SfM",
            "runtime_seconds": time.perf_counter() - started,
            "provenance": source_provenance(),
            "input_hashes": {
                str(p.relative_to(root)): file_hash(p)
                for p in sorted((root / "reference").rglob("*.png"))
            },
        },
    )

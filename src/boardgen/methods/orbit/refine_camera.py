from __future__ import annotations

import argparse
import json
import math
import shutil
import struct
from pathlib import Path

from boardgen.camera.camera_audit import audit, read_model
from boardgen.evaluation.benchmark3d import run_command, write_json


def refine(
    root: Path,
    target: Path,
    executable: Path,
    sparse: Path | None = None,
    allow_missing_fit: bool = False,
) -> dict:
    root, target, executable = root.resolve(), target.resolve(), executable.resolve()
    sparse = (sparse or root / "train/sparse/0").resolve()
    if target.exists():
        raise ValueError("camera refinement requires a fresh output dataset")
    data = json.loads((root / "dataset.json").read_text())
    cameras, images, _ = read_model(sparse)
    expected = {f"view_{i:04d}.png" for i in data["fit"]}
    registered = {i["name"] for i in images}
    if not registered <= expected:
        raise ValueError("sparse model includes images outside the fit set")
    missing = expected - registered
    if missing and (not allow_missing_fit or len(registered) / len(expected) < 0.9):
        raise ValueError(
            "incomplete SfM: require all fit views or explicit >=90% coverage experiment"
        )
    target.mkdir(parents=True)
    seed = target / "camera-fit/seed"
    seed.mkdir(parents=True)
    for name in ("images.bin", "points3D.bin"):
        shutil.copy2(sparse / name, seed / name)
    payload = bytearray(struct.pack("<Q", len(cameras)))
    for cid, c in cameras.items():
        payload.extend(
            struct.pack(
                "<iiQQddd",
                cid,
                0,
                c["width"],
                c["height"],
                math.sqrt(c["fx"] * c["fy"]),
                c["cx"],
                c["cy"],
            )
        )
    (seed / "cameras.bin").write_bytes(payload)
    model = target / "train/sparse/0"
    row = run_command(
        target / "camera-fit/solve",
        [
            str(executable),
            "--lang",
            "en",
            "sfm",
            "ba",
            str(seed),
            str(model),
            "--shared-intrinsics",
            "--max-iters",
            "200",
        ],
        [str(model / n) for n in ("cameras.bin", "images.bin", "points3D.bin")],
        timeout=180,
    )
    if row["status"] != "success":
        raise RuntimeError(
            "square-pixel bundle adjustment failed; see camera-fit/solve/process.log"
        )
    review = audit(model, target / "camera-fit/audit.json", registered)
    if review["missing_fit_views"] or review["unexpected_views"]:
        raise ValueError("bundle adjustment changed registered view membership")
    (target / "reference").symlink_to(root / "reference", target_is_directory=True)
    for folder in ("images", "masks", "feature_masks"):
        dest = target / "train" / folder
        dest.mkdir(parents=True)
        for name in registered:
            shutil.copy2(root / "train" / folder / name, dest / name)
    data["source_dataset"] = str(root)
    data["planned_fit"] = data["fit"]
    data["fit"] = sorted(int(Path(name).stem[5:]) for name in registered)
    data["unused_fit"] = sorted(int(Path(name).stem[5:]) for name in missing)
    data["cameras"] = "SIMPLE_PINHOLE, shared focal, bundle adjustment of existing correspondences"
    data["camera_provenance"] = {
        "source": str(sparse),
        "hashes": review["hashes"],
        "partial_fit_experiment": bool(missing),
    }
    write_json(target / "dataset.json", data)
    return {
        "dataset": str(target),
        "registered_views": len(registered),
        "unused_fit": data["unused_fit"],
        "reprojection_mean_px": review["reprojection_mean_px"],
        "camera_fit_seconds": row["runtime_seconds"],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Refit a sparse model with shared square-pixel focal length in a fresh dataset."
    )
    parser.add_argument("root", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("executable", type=Path)
    parser.add_argument("--sparse", type=Path)
    parser.add_argument("--allow-missing-fit", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            refine(args.root, args.target, args.executable, args.sparse, args.allow_missing_fit)
        )
    )


if __name__ == "__main__":
    main()

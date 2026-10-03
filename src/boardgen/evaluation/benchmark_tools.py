from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from boardgen.evaluation.benchmark3d import file_hash, run_command, write_json
from boardgen.evaluation.benchmark_preview import read_colmap


def sfm(
    root: Path,
    executable: Path,
    name: str = "sfm-aliked",
    features: str = "aliked-n16rot",
    max_image_size: int = 768,
    *,
    max_features: int = 8192,
    peak_threshold: float = 0.003,
    edge_threshold: float = 10,
    match_ratio: float = 0.8,
) -> dict:
    root, executable = root.resolve(), executable.resolve()
    out = root / "runs" / name
    if out.exists():
        raise ValueError(f"run already exists: {out}")
    target = root / "train/sparse/0"
    if any((target / name).exists() for name in ("cameras.bin", "images.bin", "points3D.bin")):
        raise ValueError(
            "dataset already has SfM cameras; use a fresh dataset to preserve model inputs"
        )
    workspace = out / "workspace"
    sparse = workspace / "sparse/0"
    row = run_command(
        out,
        [
            str(executable),
            "--lang",
            "en",
            "sfm",
            "auto",
            str(root / "train/images"),
            "-o",
            str(workspace),
            "--masks",
            str(root / "train/feature_masks"),
            "--quality",
            "high",
            "--data-type",
            "video",
            "--camera-mode",
            "single",
            "--camera-model",
            "simple-pinhole",
            "--features",
            features,
            "--max-image-size",
            str(max_image_size),
            *(
                [
                    "--max-features",
                    str(max_features),
                    "--peak-threshold",
                    str(peak_threshold),
                    "--edge-threshold",
                    str(edge_threshold),
                    "--ratio",
                    str(match_ratio),
                ]
                if features == "sift"
                else []
            ),
            "--pairs",
            "exhaustive",
            "--sequence",
            ".",
        ],
        [str(sparse / n) for n in ("cameras.bin", "images.bin", "points3D.bin")],
        timeout=180,
    )
    if row["status"] != "success":
        raise RuntimeError(f"SfM failed; see {out / 'process.log'}")
    data = json.loads((root / "dataset.json").read_text())
    names = {v[0] for v in read_colmap(workspace)}
    expected = {f"view_{i:04d}.png" for i in data["fit"]}
    row["registered_views"] = len(names)
    if names != expected:
        row["status"] = "failed"
        row["error"] = "SfM did not register exactly the fit set"
    write_json(out / "run.json", row)
    if row["status"] != "success":
        raise RuntimeError(row["error"])
    target.mkdir(parents=True, exist_ok=True)
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        shutil.copy2(sparse / name, target / name)
    data["cameras"] = f"SIMPLE_PINHOLE, fit-only {features} SfM; arbitrary nonmetric scale"
    data["camera_provenance"] = {
        "run": str(out.relative_to(root)),
        "hashes": {
            name: file_hash(target / name) for name in ("cameras.bin", "images.bin", "points3D.bin")
        },
    }
    write_json(root / "dataset.json", data)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit-only SfM adapter shared by Studio and reconstruction diagnostics."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("sfm")
    s.add_argument("root", type=Path)
    s.add_argument("executable", type=Path)
    s.add_argument("--name", default="sfm-aliked")
    s.add_argument("--features", choices=["aliked-n16rot", "sift"], default="aliked-n16rot")
    s.add_argument("--max-image-size", type=int, default=768)
    s.add_argument("--max-features", type=int, default=8192)
    s.add_argument("--peak-threshold", type=float, default=0.003)
    s.add_argument("--edge-threshold", type=float, default=10)
    s.add_argument("--ratio", type=float, default=0.8)
    args = parser.parse_args()
    result = sfm(
        args.root,
        args.executable,
        args.name,
        args.features,
        args.max_image_size,
        max_features=args.max_features,
        peak_threshold=args.peak_threshold,
        edge_threshold=args.edge_threshold,
        match_ratio=args.ratio,
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main()

import argparse
import json
import time
from pathlib import Path

import numpy as np
import trimesh

from boardgen.camera.camera_audit import read_model
from boardgen.evaluation.benchmark3d import run_command, write_json
from boardgen.evaluation.benchmark_surface import assess
from boardgen.methods.orbit.calibrate_orbit import rotations
from boardgen.methods.orbit.constrain_surface import constrain
from boardgen.methods.orbit.geometry_proxy import proxy

CONFIGS = {
    "coverage": {
        "primitive": "3dgs",
        "floater": "mild",
        "normal": 0.02,
        "depth": 0.01,
        "normal_distortion": 0.007,
        "underfill": 0.25,
        "sh": 3,
        "mesh_iso": 0.4,
        "mesh_carve_k": 8,
    },
    "control": {
        "primitive": "3dgs",
        "floater": "strong",
        "normal": 0.04,
        "depth": 0.01,
        "normal_distortion": 0.01,
        "underfill": 0.0,
        "sh": 3,
    },
    "detail": {
        "primitive": "mip",
        "floater": "mild",
        "normal": 0.015,
        "depth": 0.012,
        "normal_distortion": 0.007,
        "underfill": 0.3,
        "sh": 2,
    },
}


def validation_poses(root):
    data = json.loads((root / "dataset.json").read_text())
    cal = json.loads((root / "calibration.json").read_text())
    cams, ims, _ = read_model(root / "train/sparse/0")
    camera = next(iter(cams.values()))
    cp = cal["camera_parameters"]
    indices = np.array([a["index"] for a in cp])
    pars = np.array([[a["yaw_deg"], a["dx_px"], a["dy_px"]] for a in cp])
    views = []
    for index in data["heldout"]:
        q = np.array([np.interp(index, indices, pars[:, j]) for j in range(3)])
        r = rotations(
            np.radians([q[0]]), np.radians(cal["elevation_deg"]), np.radians(cal["roll_deg"])
        )[0]
        t = [q[1] / camera["fx"], q[2] / camera["fy"], 1.0]
        views.append(
            {"name": f"view_{index:04d}.png", "status": "accepted", "r": r.tolist(), "t": t}
        )
    write_json(
        root / "validation-orbit.json",
        {
            "camera": camera,
            "views": views,
            "assessment": "Temporal interpolation of fit-only calibrated cameras; selection views, not an untouched final test.",
        },
    )


def export(root, mesh_path, out):
    mesh = trimesh.load(mesh_path, force="mesh")
    center = mesh.bounds.mean(0)
    shift = np.array([-center[0], -mesh.bounds[0, 1], -center[2]])
    scale = 1 / np.ptp(mesh.vertices[:, 1])
    matrix = np.eye(4)
    matrix[:3, :3] *= scale
    matrix[:3, 3] = shift * scale
    mesh.apply_transform(matrix)
    out.mkdir(exist_ok=True)
    mesh.export(out / "photo.glb")
    write_json(
        out / "asset.json",
        {
            "transform": matrix.tolist(),
            "faces": len(mesh.faces),
            "bounds": mesh.bounds.tolist(),
            "source": str(mesh_path),
            "normalization": "Y-up height1 base0; proper uniform scale and translation only; front from physical video anchor",
        },
    )
    return mesh


def run(obj, label, iterations, cap, dataset=None, texture_size=2048, timeout=780, executable=None):
    from boardgen.studio.environment import spirula

    if dataset is None:
        raise ValueError("An explicit prepared dataset is required")
    root = dataset.resolve()
    exe = Path(executable or spirula())
    has_orbit = (root / "calibration.json").exists()
    if has_orbit:
        validation_poses(root)
    cfg = CONFIGS[label]
    name = f"spirula-{label}-{iterations}"
    out = root / "runs" / name
    out.parent.mkdir(exist_ok=True)
    started = time.perf_counter()
    flags = [
        str(exe),
        "--lang",
        "en",
        "train",
        "meshing",
        "--data",
        str(root / "train"),
        "--data-format",
        "colmap",
        "--output-dir-prefix",
        str(root / "runs"),
        "--output-dir-name",
        name,
        "--primitive",
        cfg["primitive"],
        "--num-iterations",
        str(iterations),
        "--steps-per-save",
        str(iterations),
        "--cap-max",
        str(cap),
        "--sh-degree",
        str(cfg["sh"]),
        "--train-resolution-divisor",
        "0",
        "--use-bilateral-grid",
        "0",
        "--use-ppisp",
        "0",
        "--apply-loss-for-mask",
        "1",
        "--floater-suppression",
        cfg["floater"],
        "--normal-reg-weight",
        str(cfg["normal"]),
        "--depth-distortion-reg",
        str(cfg["depth"]),
        "--normal-distortion-reg",
        str(cfg["normal_distortion"]),
        "--alpha-loss-weight-under",
        str(cfg["underfill"]),
        "--normal-reg-warmup",
        "1000",
        "--distortion-reg-warmup",
        "1000",
        "--alpha-reg-warmup",
        "1000",
        "--refine-stop-num-iter",
        "500",
        "--refine-stop-iter",
        str(iterations),
        "--disable-viewer",
        "1",
        "--keep-viewer-alive",
        "0",
    ]
    train = run_command(
        root / "runs" / f"{name}-training",
        flags,
        [str(out / f"step-{iterations:09d}.ckpt/splat.ply")],
        timeout=timeout,
    )
    print(obj, label, "TRAIN", train["status"], train["runtime_seconds"], flush=True)
    if train["status"] != "success":
        raise RuntimeError("Training failed")
    mesh = run_command(
        out / "meshing",
        [
            str(exe),
            "--lang",
            "en",
            "mesh",
            str(out),
            "--output",
            str(out / "mesh"),
            "--format",
            "glb",
            "--color",
            "texture",
            "--texture-size",
            str(texture_size),
            "--max-grid-res",
            "256",
            "--carve-k",
            str(cfg.get("mesh_carve_k", 2)),
            *(["--iso", str(cfg["mesh_iso"])] if "mesh_iso" in cfg else []),
            "--floater-min-faces",
            "100",
            "--num-threads",
            "6",
        ],
        [str(out / "mesh.glb")],
        timeout=100,
    )
    if mesh["status"] != "success":
        raise RuntimeError("Meshing failed")
    m = constrain(root, out / "mesh.glb", out / "constrained.glb")
    proxy(m).export(out / "score-mesh.ply")
    score = {"summary": None}
    if has_orbit:
        score = assess(
            root,
            out / "score-mesh.ply",
            root / "validation-orbit.json",
            out / "surface-score",
            max_side=180,
            views_per_split=4,
        )
    export(root, out / "constrained.glb", out / "review")
    row = {
        "object": obj,
        "config": cfg,
        "label": label,
        "iterations": iterations,
        "cap": cap,
        "training_seconds": train["runtime_seconds"],
        "meshing_seconds": mesh["runtime_seconds"],
        "total_seconds": time.perf_counter() - started,
        "faces": len(m.faces),
        "surface_score": score["summary"],
        "run": str(out),
        "notes": "Two bundled configurations, not an exhaustive optimum. Corrected fit-only cameras; held-out silhouettes/shading used for selection; baked shadows not relief.",
    }
    write_json(out / "result.json", row)
    print("RESULT", json.dumps(row), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("object", help="object identifier for provenance")
    p.add_argument("label", choices=CONFIGS)
    p.add_argument("--iterations", type=int, default=4500)
    p.add_argument("--cap", type=int, default=120000)
    p.add_argument("--dataset", type=Path, required=True, help="fresh calibrated dataset")
    a = p.parse_args()
    run(a.object, a.label, a.iterations, a.cap, a.dataset)


if __name__ == "__main__":
    main()

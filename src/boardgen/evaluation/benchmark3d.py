from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from boardgen.camera.camera_model import CameraIntrinsics, TurntableCameraModel, TurntableView
from boardgen.common.manifest import source_provenance as provenance
from boardgen.common.manifest import write_json
from boardgen.common.utils import file_sha256 as file_hash
from boardgen.evaluation.calib_eval import (
    OpenCVSilhouetteRenderer,
    ViewSplit,
    evaluate_model,
    write_qc_sheet,
)
from boardgen.reconstruction.recon3d import reconstruct
from boardgen.video.views import prepare_views as real


def texture(points: np.ndarray, normal: np.ndarray) -> np.ndarray:
    base = 0.5 + 0.22 * np.sin(points * np.array([53, 67, 79]) + np.array([0, 2, 4]))
    flecks = np.sin(points[:, 0] * 119 + points[:, 1] * 137 + points[:, 2] * 151)
    base += 0.15 * flecks[:, None]
    light = 0.65 + 0.35 * max(0, float(normal @ np.array([0.3, -0.4, 0.866])))
    return np.clip(base * light * 255, 0, 255).astype(np.uint8)


def rasterize_pattern(mesh: trimesh.Trimesh, projection: np.ndarray, size: int):
    vertices = np.asarray(mesh.vertices)
    hp = np.column_stack([vertices, np.ones(len(vertices))]) @ projection.T
    uv = hp[:, :2] / hp[:, 2:3]
    depth = np.full((size, size), np.inf)
    image = np.zeros((size, size, 3), np.uint8)
    for face, normal in zip(mesh.faces, mesh.face_normals, strict=True):
        if np.any(hp[face, 2] <= 0):
            continue
        xy = uv[face]
        lo = np.maximum(0, np.floor(xy.min(axis=0)).astype(int))
        hi = np.minimum(size - 1, np.ceil(xy.max(axis=0)).astype(int))
        if np.any(hi < lo):
            continue
        xx, yy = np.meshgrid(np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1))
        x, y = xx.ravel(), yy.ravel()
        matrix = np.vstack([xy.T, np.ones(3)])
        if abs(np.linalg.det(matrix)) < 1e-10:
            continue
        bary = np.column_stack([x, y, np.ones(len(x))]) @ np.linalg.inv(matrix).T
        keep = np.all(bary >= -1e-8, axis=1)
        weights = bary / hp[face, 2]
        z = 1 / np.maximum(weights.sum(axis=1), 1e-12)
        keep &= z < depth[y, x]
        weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
        points = weights[keep] @ vertices[face]
        depth[y[keep], x[keep]] = z[keep]
        image[y[keep], x[keep]] = texture(points, normal)
    return image, np.isfinite(depth).astype(np.uint8) * 255


def synthetic(root: Path, size: int = 512, views: int = 40) -> None:
    if root.exists():
        raise ValueError(f"dataset already exists: {root}")
    if views < 10 or views % 5:
        raise ValueError("views must be a multiple of 5, at least 10")
    root.mkdir(parents=True)

    mesh = trimesh.creation.box(extents=(0.5, 0.36, 0.6))
    mesh.vertices[mesh.vertices[:, 2] > 0, 2] += mesh.vertices[mesh.vertices[:, 2] > 0, 0] * 0.35
    mesh.export(root / "truth.ply")
    model = TurntableCameraModel(
        image_width=size,
        image_height=size,
        intrinsics=CameraIntrinsics(
            size, size, size * 1.65, size * 1.65, (size - 1) / 2, (size - 1) / 2
        ),
        views=[TurntableView(i, i * 2 * np.pi / views, f"view_{i:04d}") for i in range(views)],
        camera_distance=2.0,
        elevation_deg=30,
        initialization_method="synthetic-ground-truth",
    )
    model.save_yaml(str(root / "cameras.yaml"))
    reference = root / "reference"
    (reference / "images").mkdir(parents=True)
    (reference / "masks").mkdir()
    for i, projection in enumerate(model.projection_matrices()):
        rgb, alpha = rasterize_pattern(mesh, projection, size)
        name = f"view_{i:04d}.png"
        cv2.imwrite(str(reference / "images" / name), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(reference / "masks" / name), alpha)
    fit = [i for i in range(views) if i % 5 != 4]
    heldout = [i for i in range(views) if i % 5 == 4]
    write_json(
        root / "dataset.json",
        {
            "schema": "boardgen.benchmark3d.v1",
            "kind": "synthetic-patterned-wedge",
            "fit": fit,
            "heldout": heldout,
            "image_size": size,
            "bounds": [[-0.45, 0.45], [-0.4, 0.4], [-0.4, 0.5]],
            "mask_source": "analytic-z-buffer",
            "cameras": "ground-truth",
            "note": "Single 30-degree ring; underside is unobserved. No truth points seed training.",
            "provenance": provenance(),
            "input_hashes": {
                str(p.relative_to(root)): file_hash(p) for p in sorted(reference.rglob("*.png"))
            },
        },
    )
    export_training(root)
    contact_sheet(root)


def inputs(root: Path):
    data = json.loads((root / "dataset.json").read_text())
    model = TurntableCameraModel.load_yaml(str(root / "cameras.yaml"))
    masks = np.stack(
        [
            cv2.imread(str(root / "reference/masks" / f"view_{i:04d}.png"), 0) >= 128
            for i in range(len(model.views))
        ]
    ).astype(np.uint8)
    return data, model, masks


def export_training(root: Path) -> None:
    data, model, _ = inputs(root)
    train = root / "train"
    for sub in ("images", "masks", "sparse/0"):
        (train / sub).mkdir(parents=True, exist_ok=True)
    intr = model.effective_intrinsics()
    camera_line = f"1 PINHOLE {intr.width} {intr.height} {intr.fx} {intr.fy} {intr.cx} {intr.cy}\n"
    (train / "sparse/0/cameras.txt").write_text(camera_line)
    frames, colmap = [], []
    for index in data["fit"]:
        name = f"view_{index:04d}.png"
        bgr = cv2.imread(str(root / "reference/images" / name))
        alpha = cv2.imread(str(root / "reference/masks" / name), 0)
        rgba = np.dstack([bgr, alpha])
        cv2.imwrite(str(train / "images" / name), rgba)
        cv2.imwrite(str(train / "masks" / name), alpha)

        cv2.imwrite(str(train / "images" / (Path(name).stem + ".mask.png")), alpha)
        r, t = model.opencv_extrinsics()[index]
        c2w = np.linalg.inv(np.vstack([np.column_stack([r, t]), [0, 0, 0, 1]]))
        c2w[:, 1:3] *= -1
        frames.append({"file_path": f"images/{name}", "transform_matrix": c2w.tolist()})
        q = Rotation.from_matrix(r).as_quat()
        values = [q[3], *q[:3], *t]
        colmap.append(f"{index + 1} " + " ".join(f"{v:.12g}" for v in values) + f" 1 {name}\n\n")
    (train / "sparse/0/images.txt").write_text("".join(colmap))
    (train / "sparse/0/points3D.txt").write_text("# No ground-truth geometry provided.\n")
    write_json(
        train / "transforms.json",
        {
            "camera_model": "OPENCV",
            "w": intr.width,
            "h": intr.height,
            "fl_x": intr.fx,
            "fl_y": intr.fy,
            "cx": intr.cx,
            "cy": intr.cy,
            "orientation_override": "none",
            "frames": frames,
        },
    )


def contact_sheet(root: Path) -> None:
    data, model, _ = inputs(root)
    tiles = []
    for i in range(len(model.views)):
        rgb = cv2.imread(str(root / "reference/images" / f"view_{i:04d}.png"))
        tile = cv2.resize(rgb, (128, 128))
        cv2.putText(
            tile,
            f"{i} {'TEST' if i in data['heldout'] else 'fit'}",
            (4, 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
        )
        tiles.append(tile)
    while len(tiles) % 8:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(
        str(root / "contact.png"),
        np.vstack([np.hstack(tiles[i : i + 8]) for i in range(0, len(tiles), 8)]),
    )


def evaluate(root: Path, mesh_path: Path, out: Path) -> dict:
    data, model, masks = inputs(root)
    mesh = trimesh.load(mesh_path, force="mesh")
    if not len(mesh.faces) or not np.isfinite(mesh.vertices).all():
        raise ValueError("output is not a finite triangle mesh")
    split = ViewSplit(data["fit"], data["heldout"])
    result = evaluate_model(masks, mesh, model, split)
    result.update(
        {
            "mesh": str(mesh_path),
            "vertices": len(mesh.vertices),
            "faces": len(mesh.faces),
            "watertight": bool(mesh.is_watertight),
            "bounds": mesh.bounds.tolist(),
            "mesh_sha256": file_hash(mesh_path),
        }
    )
    if (root / "truth.ply").exists():
        truth = trimesh.load(root / "truth.ply", force="mesh")
        a, _ = trimesh.sample.sample_surface(mesh, 10000, seed=42)
        b, _ = trimesh.sample.sample_surface(truth, 10000, seed=42)
        distance = np.concatenate([cKDTree(a).query(b)[0], cKDTree(b).query(a)[0]])
        result["surface_distance"] = {
            "mean": float(distance.mean()),
            "p95": float(np.percentile(distance, 95)),
            "normalized_mean": float(distance.mean() / np.linalg.norm(truth.extents)),
            "note": "Full surface, including the unobserved underside; approximate sampled distance.",
        }
    out.mkdir(parents=True, exist_ok=True)
    predicted = OpenCVSilhouetteRenderer().render(mesh, model, split.heldout)
    write_qc_sheet(str(out / "heldout.png"), masks, predicted, split.heldout, mesh_path.stem)
    write_json(out / "metrics.json", result)
    return result






def hull(root: Path, grid: int = 64, consensus: float = 0.95) -> dict:
    data, model, masks = inputs(root)
    started = time.perf_counter()
    projections = model.projection_matrices()
    vertices, faces = reconstruct(
        masks[data["fit"]],
        [projections[i] for i in data["fit"]],
        {
            "grid_res": grid,
            "consensus": consensus,
            "samples_per_axis": 1,
            "bounds": data["bounds"],
        },
    )
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    out = root / "runs" / f"hull-{grid}"
    out.mkdir(parents=True, exist_ok=True)
    mesh.export(out / "mesh.ply")
    result = evaluate(root, out / "mesh.ply", out)
    result["runtime_seconds"] = time.perf_counter() - started
    result["geometry_input_view_indices"] = data["fit"]
    result["settings"] = {"grid": grid, "consensus": consensus, "bounds": data["bounds"]}
    write_json(out / "metrics.json", result)

    return result








def main() -> None:
    parser = argparse.ArgumentParser(
        description="Small reconstruction benchmark with physically separate fit/evaluation inputs.\n\nRun with ``python -m boardgen.evaluation.benchmark3d --help``. External tools are optional;\nthe synthetic fixture and evaluation run on CPU without Blender or CUDA."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    syn = sub.add_parser("synthetic")
    syn.add_argument("root", type=Path)
    syn.add_argument("--size", type=int, default=512)
    syn.add_argument("--views", type=int, default=40)
    r = sub.add_parser("real")
    r.add_argument("root", type=Path)
    r.add_argument("video", type=Path)
    r.add_argument("--size", type=int, default=512)
    r.add_argument("--views", type=int, default=40)
    r.add_argument("--mask-size", type=int, choices=[1024, 1536, 2048], default=1024)
    r.add_argument("--crop-padding", type=float, default=0.08)
    r.add_argument(
        "--recover-warm",
        action="store_true",
        help="opt-in warm plastic recovery against a neutral crop border; requires visual review",
    )
    r.add_argument("--full-clip", action="store_true")
    h = sub.add_parser("hull")
    h.add_argument("root", type=Path)
    h.add_argument("--grid", type=int, default=64)
    h.add_argument("--consensus", type=float, default=0.95)
    ev = sub.add_parser("evaluate")
    ev.add_argument("root", type=Path)
    ev.add_argument("mesh", type=Path)
    ev.add_argument("out", type=Path)
    args = parser.parse_args()
    if args.command == "synthetic":
        synthetic(args.root, args.size, args.views)
    elif args.command == "real":
        real(
            args.root,
            args.video,
            args.size,
            args.views,
            args.full_clip,
            mask_size=args.mask_size,
            recover_warm=args.recover_warm,
            crop_padding=args.crop_padding,
        )
    elif args.command == "hull":
        print(json.dumps(hull(args.root, args.grid, args.consensus)))
    elif args.command == "evaluate":
        print(json.dumps(evaluate(args.root, args.mesh, args.out)))



if __name__ == "__main__":
    main()

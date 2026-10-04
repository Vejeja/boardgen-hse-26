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
    if mesh.visual.kind in ("texture", "vertex"):
        result["appearance"] = appearance(root, mesh, model, split.heldout, out)
    write_json(out / "metrics.json", result)
    return result


def render_textured(mesh: trimesh.Trimesh, projection: np.ndarray, width: int, height: int):
    vertices = np.asarray(mesh.vertices)
    hp = np.column_stack([vertices, np.ones(len(vertices))]) @ projection.T
    uv = hp[:, :2] / np.maximum(hp[:, 2:3], 1e-12)
    depth = np.full((height, width), np.inf)
    image = np.zeros((height, width, 3), np.uint8)
    texture_image = None
    if mesh.visual.kind == "texture":
        material = mesh.visual.material
        texture_image = getattr(material, "image", None)
        if texture_image is None:
            texture_image = getattr(material, "baseColorTexture", None)
        if texture_image is None:
            raise ValueError("mesh has no supported base-color texture")
        texture_image = np.asarray(texture_image.convert("RGB"))
    for face in mesh.faces:
        if np.any(hp[face, 2] <= 0):
            continue
        xy = uv[face]
        lo = np.maximum(0, np.floor(xy.min(axis=0)).astype(int))
        hi = np.minimum([width - 1, height - 1], np.ceil(xy.max(axis=0)).astype(int))
        if np.any(hi < lo):
            continue
        matrix = np.vstack([xy.T, np.ones(3)])
        if abs(np.linalg.det(matrix)) < 1e-10:
            continue
        xx, yy = np.meshgrid(np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1))
        x, y = xx.ravel(), yy.ravel()
        bary = np.column_stack([x, y, np.ones(len(x))]) @ np.linalg.inv(matrix).T
        keep = np.all(bary >= -1e-8, axis=1)
        weights = bary / hp[face, 2]
        z = 1 / np.maximum(weights.sum(axis=1), 1e-12)
        keep &= z < depth[y, x]
        if not keep.any():
            continue
        weights = weights[keep]
        weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
        if texture_image is not None:
            tex_uv = weights @ mesh.visual.uv[face]
            th, tw = texture_image.shape[:2]
            tx = np.clip(np.rint(tex_uv[:, 0] * (tw - 1)).astype(int), 0, tw - 1)
            ty = np.clip(np.rint((1 - tex_uv[:, 1]) * (th - 1)).astype(int), 0, th - 1)
            color = texture_image[ty, tx]
        else:
            color = weights @ mesh.visual.vertex_colors[face, :3]
        image[y[keep], x[keep]] = np.clip(color, 0, 255).astype(np.uint8)
        depth[y[keep], x[keep]] = z[keep]
    return image


def appearance(
    root: Path, mesh: trimesh.Trimesh, model: TurntableCameraModel, indices: list[int], out: Path
) -> dict:
    rows, tiles = [], []
    projections = model.projection_matrices()
    for i in indices:
        pred = render_textured(mesh, projections[i], model.image_width, model.image_height)
        target = cv2.cvtColor(
            cv2.imread(str(root / "reference/images" / f"view_{i:04d}.png")), cv2.COLOR_BGR2RGB
        )
        alpha = cv2.imread(str(root / "reference/masks" / f"view_{i:04d}.png"), 0)
        target = (target.astype(float) * (alpha[:, :, None] / 255)).astype(np.uint8)
        delta = (target.astype(float) - pred.astype(float)) / 255
        foreground = alpha >= 128
        mse = float(np.square(delta[foreground]).mean())
        rows.append(
            {
                "view": i,
                "foreground_psnr": float(-10 * np.log10(max(mse, 1e-12))),
                "foreground_mae": float(np.abs(delta[foreground]).mean()),
            }
        )
        cv2.imwrite(str(out / f"render_{i:04d}.png"), cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))
        tile = np.hstack([cv2.resize(target, (192, 192)), cv2.resize(pred, (192, 192))])
        tiles.append(tile)
    cv2.imwrite(str(out / "appearance.png"), cv2.cvtColor(np.vstack(tiles), cv2.COLOR_RGB2BGR))
    return {
        "foreground_psnr_mean": float(np.mean([r["foreground_psnr"] for r in rows])),
        "foreground_mae_mean": float(np.mean([r["foreground_mae"] for r in rows])),
        "per_view": rows,
        "renderer": "CPU z-buffer, baked color, nearest texture sampling",
        "note": "Mesh appearance only; not splat render metrics or relighting quality.",
    }


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

    seed_points(root, mesh, out / "mesh.ply")
    return result


def seed_points(root: Path, mesh: trimesh.Trimesh, mesh_path: Path) -> None:
    data, model, masks = inputs(root)
    points, faces = trimesh.sample.sample_surface(mesh, 2500, seed=42)
    normals = mesh.face_normals[faces]
    observations = {i: [] for i in data["fit"]}
    candidates = {}
    extrinsics = model.opencv_extrinsics()
    projections = model.projection_matrices()
    for i in data["fit"]:
        r, t = extrinsics[i]
        center = -r.T @ t
        hp = np.column_stack([points, np.ones(len(points))]) @ projections[i].T
        xy = np.rint(hp[:, :2] / np.maximum(hp[:, 2:3], 1e-12)).astype(int)
        valid = (hp[:, 2] > 0) & ((normals * (center - points)).sum(axis=1) > 0)
        valid &= (
            (xy[:, 0] >= 0)
            & (xy[:, 0] < model.image_width)
            & (xy[:, 1] >= 0)
            & (xy[:, 1] < model.image_height)
        )
        valid_indices = np.flatnonzero(valid)
        valid[valid_indices] &= masks[i, xy[valid_indices, 1], xy[valid_indices, 0]] > 0
        candidates[i] = (xy, valid)
    point_rows, kept, colors = [], [], []
    for pi, point in enumerate(points):
        seen, rgb = [], []
        for i in data["fit"]:
            xy, valid = candidates[i]
            if valid[pi]:
                x, y = xy[pi]
                seen.append((i, x, y))
        if len(seen) < 2:
            continue
        pid = len(kept) + 1
        tracks = []
        for i, x, y in seen:
            tracks.extend([i + 1, len(observations[i])])
            observations[i].append((x, y, pid))

        i, x, y = seen[0]
        rgb.append((i, x, y))
        kept.append(point)
        colors.append(rgb[0])
        point_rows.append((pid, point, tracks))
    image_cache = {
        i: cv2.cvtColor(
            cv2.imread(str(root / "reference/images" / f"view_{i:04d}.png")), cv2.COLOR_BGR2RGB
        )
        for i in data["fit"]
    }
    color_values = np.array([image_cache[i][y, x] for i, x, y in colors], np.uint8)
    sparse = root / "train/sparse/0"
    old = (sparse / "images.txt").read_text().splitlines()
    headers = [line for line in old if line and not line.startswith("#")][::1]

    headers = [
        line for line in headers if len(line.split()) >= 10 and line.split()[-1].endswith(".png")
    ]
    lines = []
    for header in headers:
        i = int(header.split()[0]) - 1
        lines.append(
            header + "\n" + " ".join(f"{x} {y} {pid}" for x, y, pid in observations[i]) + "\n"
        )
    (sparse / "images.txt").write_text("".join(lines))
    text = []
    for (pid, point, tracks), color in zip(point_rows, color_values, strict=True):
        text.append(
            f"{pid} "
            + " ".join(str(float(x)) for x in point)
            + " "
            + " ".join(str(int(x)) for x in color)
            + " 0 "
            + " ".join(str(x) for x in tracks)
            + "\n"
        )
    (sparse / "points3D.txt").write_text("".join(text))
    trimesh.points.PointCloud(np.array(kept), colors=color_values).export(root / "train/points.ply")
    transforms_path = root / "train/transforms.json"
    transforms = json.loads(transforms_path.read_text())
    transforms["ply_file_path"] = "points.ply"
    write_json(transforms_path, transforms)
    write_json(
        root / "train/seed.json",
        {
            "source": "fit-only-visual-hull",
            "points": len(kept),
            "fit": data["fit"],
            "mesh_sha256": file_hash(mesh_path),
            "note": "Shared initialization, not ground truth or held-out geometry.",
        },
    )

    for name in ("cameras.txt", "images.txt", "points3D.txt"):
        (sparse.parent / name).write_bytes((sparse / name).read_bytes())


def run_command(out: Path, command: list[str], expected: list[str], timeout: int = 600) -> dict:
    if not command or not expected:
        raise ValueError("a command and at least one expected artifact are required")
    if out.exists():
        raise ValueError("run directory already exists; choose a fresh run directory")
    out.mkdir(parents=True, exist_ok=True)
    paths = [Path(p) for p in expected]
    if any(p.exists() for p in paths):
        raise ValueError("expected output already exists; choose a fresh run directory")
    started = time.perf_counter()
    row = {"command": command, "provenance": provenance(), "expected": expected}
    if command and Path(command[0]).is_file():
        row["executable_sha256"] = file_hash(Path(command[0]))
    with (out / "process.log").open("w") as log:
        try:
            process = subprocess.run(
                command, stdout=log, stderr=subprocess.STDOUT, timeout=timeout, check=False
            )
            row["returncode"] = process.returncode
            row["status"] = (
                "success"
                if process.returncode == 0 and all(p.is_file() and p.stat().st_size for p in paths)
                else "failed"
            )
        except subprocess.TimeoutExpired:
            row["status"] = "timeout"
        except OSError as exc:
            row.update(status="failed", error=str(exc))
    row["runtime_seconds"] = time.perf_counter() - started
    row["artifacts"] = {str(p): file_hash(p) for p in paths if p.is_file()}
    write_json(out / "run.json", row)
    return row


def constrain_mesh(root: Path, mesh_path: Path, out: Path, consensus: float = 0.95) -> dict:
    data, model, masks = inputs(root)
    mesh = trimesh.load(mesh_path, force="mesh")
    points = np.asarray(mesh.vertices)
    support = np.zeros(len(points), np.int32)
    for i in data["fit"]:
        uv, depth = model.project_points(points, i)
        xy = np.rint(uv).astype(int)
        valid = (
            (depth > 0)
            & (xy[:, 0] >= 0)
            & (xy[:, 0] < model.image_width)
            & (xy[:, 1] >= 0)
            & (xy[:, 1] < model.image_height)
        )
        idx = np.flatnonzero(valid)
        mask = cv2.dilate(masks[i], np.ones((3, 3), np.uint8))
        support[idx] += mask[xy[idx, 1], xy[idx, 0]] > 0
    keep_vertices = support / len(data["fit"]) >= consensus
    keep = np.all(keep_vertices[mesh.faces], axis=1)
    original = len(mesh.faces)
    if not keep.any():
        raise ValueError("no mesh faces are supported by the fit silhouettes")
    mesh.update_faces(keep)
    mesh.remove_unreferenced_vertices()
    out.mkdir(parents=True, exist_ok=True)
    mesh.export(out / "mesh.glb")
    row = evaluate(root, out / "mesh.glb", out)
    row["constraint"] = {
        "method": "fit-mask-triangle-culling",
        "consensus": consensus,
        "pixel_tolerance": 1,
        "fit": data["fit"],
        "source_mesh_sha256": file_hash(mesh_path),
        "faces_before": original,
        "faces_after": len(mesh.faces),
    }
    write_json(out / "metrics.json", row)
    return row


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
    clip = sub.add_parser("constrain")
    clip.add_argument("root", type=Path)
    clip.add_argument("mesh", type=Path)
    clip.add_argument("out", type=Path)
    run = sub.add_parser("run")
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--expected", action="append", required=True)
    run.add_argument("--timeout", type=int, default=600)
    run.add_argument("argv", nargs=argparse.REMAINDER)
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
    elif args.command == "constrain":
        print(json.dumps(constrain_mesh(args.root, args.mesh, args.out)))
    elif args.command == "run":
        command = args.argv[1:] if args.argv[:1] == ["--"] else args.argv
        row = run_command(args.out, command, args.expected, args.timeout)
        print(json.dumps(row))
        if row["status"] != "success":
            raise SystemExit(1)


if __name__ == "__main__":
    main()

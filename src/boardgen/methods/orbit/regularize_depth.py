import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pymeshlab
import trimesh

from boardgen.camera.camera_audit import read_model
from boardgen.evaluation.benchmark3d import write_json
from boardgen.evaluation.benchmark_preview import read_colmap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("run", type=Path)
    ap.add_argument("--output-name", default="depth-fusion-consistent-v2")
    ap.add_argument("--source", type=Path)
    ap.add_argument("--output-directory", type=Path)
    ap.add_argument("--free-space-max", type=float, default=0.2)
    ap.add_argument("--min-depth-views", type=int, default=8)
    ap.add_argument("--depth-tolerance", type=float, default=0.015)
    ap.add_argument("--poisson-depth", type=int, default=8)
    ap.add_argument("--point-weight", type=float, default=2)
    ap.add_argument("--samples-per-node", type=float, default=3)
    a = ap.parse_args()
    if not (
        0 <= a.free_space_max <= 0.2
        and 1 <= a.min_depth_views <= 32
        and 0 < a.depth_tolerance <= 0.03
        and 7 <= a.poisson_depth <= 10
        and 1 <= a.point_weight <= 10
        and 1 <= a.samples_per_node <= 5
    ):
        ap.error("Invalid bounded depth-fusion parameters")
    root = a.root.resolve()
    run = a.run.resolve()
    old = (a.source or run / "depth-fusion-oct7").resolve()
    out = (a.output_directory or run / a.output_name).resolve()
    out.mkdir(exist_ok=False)
    start = time.perf_counter()
    unitmesh = trimesh.load(old / "unit-mesh.ply", force="mesh")
    worldmesh = trimesh.load(old / "mesh.ply", force="mesh")
    v = unitmesh.vertices
    w = worldmesh.vertices
    scale = np.sum((v - v.mean(0)) * (w - w.mean(0))) / np.sum((v - v.mean(0)) ** 2)
    offset = w.mean(0) - scale * v.mean(0)
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(old / "oriented.ply"))
    p = ms.current_mesh().vertex_matrix() * scale + offset
    normals = ms.current_mesh().vertex_normal_matrix()
    colors = ms.current_mesh().vertex_color_matrix()
    views = read_colmap(root / "train")
    _, images, _ = read_model(root / "train/sparse/0")
    centres = {im["name"]: -im["r"].T @ im["t"] for im in images}
    names = [
        Path(n).name for n in json.loads((old / "cameras.json").read_text())["image_filenames"]
    ]
    mapping = {name: i for i, name in enumerate(names)}
    vote = np.zeros(len(p))
    free = np.zeros(len(p))
    valid_depth = np.zeros(len(p))
    ph = np.column_stack([p, np.ones(len(p))])
    height = np.ptp(worldmesh.vertices[:, 1])
    tol = height * a.depth_tolerance
    for name, proj, width, h in views:
        hp = ph @ proj.T
        xy = np.rint(hp[:, :2] / np.maximum(hp[:, 2:], 1e-9)).astype(int)
        inside = (
            (hp[:, 2] > 0) & (xy[:, 0] >= 0) & (xy[:, 0] < width) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
        )
        ids = np.flatnonzero(inside)
        mask = cv2.imread(str(root / "reference/masks" / name), 0) > 128
        mask = cv2.dilate(mask.astype("uint8"), np.ones((3, 3), "uint8"))
        vote[ids] += mask[xy[ids, 1], xy[ids, 0]]
        f = old / f"moments-{mapping[name]:03d}.f32"
        if not f.exists():
            continue
        with f.open("rb") as stream:
            hh, ww, c = np.fromfile(stream, "int32", 3)
            mom = np.fromfile(stream, "float32").reshape(hh, ww, c)

        uv = xy[ids] * [ww / width, hh / h]
        uv = np.rint(uv).astype(int)
        good = (uv[:, 0] >= 0) & (uv[:, 0] < ww) & (uv[:, 1] >= 0) & (uv[:, 1] < hh)
        ids = ids[good]
        uv = uv[good]
        sample = mom[uv[:, 1], uv[:, 0]]
        good = np.isfinite(sample).all(1) & (sample[:, 0] > 0.7) & (sample[:, 2] > 1e-6)
        ids = ids[good]
        sample = sample[good]
        depth = (np.sqrt(-np.log(0.5)) - sample[:, 1]) / sample[:, 2]

        distance = np.linalg.norm(p[ids] - centres[name], axis=1)
        valid_depth[ids] += 1
        free[ids] += distance < depth - tol
    keep = (
        (vote / len(views) >= 0.94)
        & (valid_depth >= a.min_depth_views)
        & (free / np.maximum(valid_depth, 1) <= a.free_space_max)
    )
    p, normals, colors = p[keep], normals[keep], colors[keep]
    if len(p) < 500:
        raise ValueError("Too few consistent surface points")
    origin = (p.min(0) + p.max(0)) / 2
    unit = np.ptp(p, axis=0).max()
    ms = pymeshlab.MeshSet()
    ms.add_mesh(
        pymeshlab.Mesh(
            vertex_matrix=(p - origin) / unit, v_normals_matrix=normals, v_color_matrix=colors
        )
    )
    ms.save_current_mesh(str(out / "consistent-points.ply"), save_vertex_normal=True)
    ms.generate_surface_reconstruction_screened_poisson(
        depth=a.poisson_depth,
        pointweight=a.point_weight,
        samplespernode=a.samples_per_node,
        threads=1,
    )
    ms.meshing_decimation_quadric_edge_collapse(
        targetfacenum=100000, preservenormal=True, preservetopology=True
    )
    ms.save_current_mesh(str(out / "unit-mesh.ply"))
    mesh = trimesh.load(out / "unit-mesh.ply", force="mesh")
    mesh.vertices = mesh.vertices * unit + origin
    mesh.export(out / "mesh.ply")
    mesh.export(out / "mesh.glb")
    write_json(
        out / "result.json",
        {
            "runtime_seconds": time.perf_counter() - start,
            "retained_points": len(p),
            "original_points": len(keep),
            "faces": len(mesh.faces),
            "mask_support": 0.94,
            "depth_free_space_violation_max": a.free_space_max,
            "min_depth_views": a.min_depth_views,
            "poisson_depth": a.poisson_depth,
            "point_weight": a.point_weight,
            "samples_per_node": a.samples_per_node,
            "depth_tolerance": tol,
            "source_depth": str(old),
            "assessment": "Fit-only multi-view free-space and neural-matte filtering before Poisson; no generated shape or hidden evaluation inputs.",
        },
    )
    print(out, len(p), len(mesh.faces), time.perf_counter() - start, flush=True)


if __name__ == "__main__":
    main()

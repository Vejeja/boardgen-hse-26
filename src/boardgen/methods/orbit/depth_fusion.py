import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import pymeshlab as ml
import trimesh

from boardgen.evaluation.benchmark3d import file_hash, write_json


def main():
    parser = argparse.ArgumentParser(
        description="Fuse native Spirula depth into a regularized surface on corrected cameras.\n\nSerial diagnostic exports use a fixed temporary filename in this pinned tool.\nDo not run concurrently with another native moment export or Apple GPU training."
    )
    parser.add_argument("root", type=Path)
    parser.add_argument("run", type=Path)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--output-name", default="depth-fusion-raw")
    parser.add_argument("--output-directory", type=Path)
    args = parser.parse_args()
    exe = args.executable.resolve()
    shared = Path("/tmp/ss_moments.f32")
    if shared.exists():
        raise RuntimeError(
            "The native diagnostic filename is already occupied; preserve that file first."
        )
    root = args.root.resolve()
    dataset = root.name
    run = args.run.resolve()
    out = (args.output_directory or run / args.output_name).resolve()
    out.mkdir()
    cfg = json.loads((run / "config.json").read_text())
    start = time.perf_counter()
    dump = out / "cameras.json"
    env = os.environ.copy()
    env["SS_DUMP_CAMERAS"] = str(dump)
    with (out / "camera-dump.log").open("w") as log:
        subprocess.run(
            [
                str(exe),
                "--lang",
                "en",
                "train",
                "meshing",
                "--data",
                str(root / "train"),
                "--data-format",
                "colmap",
                "--num-iterations",
                "0",
                "--train-resolution-divisor",
                str(cfg["train_resolution_divisor"]),
                "--disable-viewer",
                "1",
                "--keep-viewer-alive",
                "0",
            ],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=30,
        )
    cameras = json.loads(dump.read_text())
    poses = np.array(cameras["c2w"]).reshape(-1, 3, 4)
    intrins = np.array(cameras["intrins"]).reshape(-1, 4)
    points, normals, colors, rows = [], [], [], []
    selected = set(np.linspace(0, len(cameras["image_filenames"]) - 1, 32).astype(int).tolist())
    for i, image in enumerate(cameras["image_filenames"]):
        if i not in selected:
            continue
        env = os.environ.copy()
        env["SS_MESH_DEBUG_RENDER"] = str(i)

        with (out / f"view-{i:03d}.log").open("w") as log:
            result = subprocess.run(
                [
                    str(exe),
                    "--lang",
                    "en",
                    "mesh",
                    str(run),
                    "--output",
                    str(out / f"unused-{i}"),
                    "--format",
                    "ply",
                    "--color",
                    "none",
                ],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=30,
            )
        if not shared.exists():
            raise RuntimeError("missing native moment dump")
        with shared.open("rb") as f:
            h, w, c = np.fromfile(f, "int32", 3)
            m = np.fromfile(f, "float32").reshape(h, w, c)
        shared.rename(out / f"moments-{i:03d}.f32")
        valid = np.isfinite(m).all(2) & (m[:, :, 0] > 0.7) & (m[:, :, 2] > 1e-6)
        depth = np.zeros((h, w))
        depth[valid] = (np.sqrt(-np.log(0.5)) - m[:, :, 1][valid]) / m[:, :, 2][valid]
        valid &= depth > 0
        nm = Path(image).name
        mask = (
            cv2.resize(
                cv2.imread(str(root / "train/masks" / nm), 0),
                (w, h),
                interpolation=cv2.INTER_NEAREST,
            )
            >= 128
        )
        valid &= mask
        fx, fy, cx, cy = intrins[i]
        xx, yy = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
        rays = np.stack([(xx - cx) / fx, -(yy - cy) / fy, -np.ones_like(xx)], 2)
        rays /= np.linalg.norm(rays, axis=2)[:, :, None]
        cam = rays * depth[:, :, None]
        xyz = cam @ poses[i, :, :3].T + poses[i, :, 3]
        dx = xyz[1:-1, 2:] - xyz[1:-1, :-2]
        dy = xyz[2:, 1:-1] - xyz[:-2, 1:-1]
        normal = np.cross(dx, dy)
        length = np.linalg.norm(normal, axis=2)
        good = (
            valid[1:-1, 1:-1]
            & valid[1:-1, 2:]
            & valid[1:-1, :-2]
            & valid[2:, 1:-1]
            & valid[:-2, 1:-1]
        )
        good &= (
            (length > 1e-12)
            & (np.linalg.norm(dx, axis=2) < 0.01 * depth[1:-1, 1:-1])
            & (np.linalg.norm(dy, axis=2) < 0.01 * depth[1:-1, 1:-1])
        )
        grid = np.zeros(good.shape, bool)
        grid[::2, ::2] = True
        good &= grid
        normal /= np.maximum(length[:, :, None], 1e-12)
        n = normal[good]
        p = xyz[1:-1, 1:-1][good]
        sign = np.where(np.sum(n * (poses[i, :, 3] - p), 1) >= 0, 1, -1)
        n *= sign[:, None]
        photo = cv2.cvtColor(cv2.imread(str(root / "reference/images" / nm)), cv2.COLOR_BGR2RGB)
        photo = cv2.resize(photo, (w, h), interpolation=cv2.INTER_AREA)
        rgb = photo[1:-1, 1:-1][good] / 255
        points.append(p)
        normals.append(n)
        colors.append(rgb)
        rows.append(
            {"camera": i, "name": nm, "points": len(p), "native_returncode": result.returncode}
        )
        if i % 8 == 0:
            print(dataset, i, len(p), flush=True)
    xyz = np.concatenate(points)
    n = np.concatenate(normals)
    rgb = np.concatenate(colors)
    origin = (xyz.min(0) + xyz.max(0)) / 2
    unit = np.ptp(xyz, axis=0).max()
    xyz = (xyz - origin) / unit

    voxel = np.floor(xyz / 0.002).astype(int)
    _, index, inv = np.unique(voxel, axis=0, return_index=True, return_inverse=True)
    count = np.bincount(inv)
    fused = []
    for array in [xyz, n, rgb]:
        fused.append(
            np.column_stack([np.bincount(inv, weights=array[:, j]) / count for j in range(3)])
        )
    xyz, n, rgb = fused
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    ms = ml.MeshSet()
    ms.add_mesh(
        ml.Mesh(
            vertex_matrix=xyz,
            v_normals_matrix=n,
            v_color_matrix=np.column_stack([rgb, np.ones(len(rgb))]),
        )
    )
    ms.save_current_mesh(str(out / "oriented.ply"), save_vertex_normal=True)
    ms.generate_surface_reconstruction_screened_poisson(
        depth=8, pointweight=2, samplespernode=3, threads=1
    )
    ms.meshing_decimation_quadric_edge_collapse(
        targetfacenum=100000, preservenormal=True, preservetopology=True
    )
    ms.save_current_mesh(str(out / "unit-mesh.ply"))
    mesh = trimesh.load(out / "unit-mesh.ply", force="mesh")
    mesh.vertices = mesh.vertices * unit + origin
    mesh.export(out / "mesh.ply")
    mesh.export(out / "mesh.glb")
    row = {
        "points": len(xyz),
        "faces": len(mesh.faces),
        "views": rows,
        "runtime_seconds": time.perf_counter() - start,
        "settings": {
            "depth_iso": 0.5,
            "alpha_min": 0.7,
            "stride": 2,
            "voxel_fraction": 0.002,
            "poisson_depth": 8,
            "point_weight": 2,
            "samples_per_node": 3,
            "threads": 1,
        },
        "native_sha256": file_hash(exe),
        "source_camera_sha256": file_hash(root / "train/sparse/0/images.bin"),
        "assessment": "Fit-only depth fusion, no independent geometric ground truth.",
    }
    write_json(out / "result.json", row)
    print(dataset, "done", row["runtime_seconds"], flush=True)


if __name__ == "__main__":
    main()

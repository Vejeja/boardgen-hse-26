from __future__ import annotations

import json
import struct
from pathlib import Path

import cv2
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation


def read_colmap(dataset: Path) -> list[tuple[str, np.ndarray, int, int]]:
    sparse = dataset / "sparse/0"
    cameras = {}
    with (sparse / "cameras.bin").open("rb") as stream:
        count = struct.unpack("<Q", stream.read(8))[0]
        for _ in range(count):
            cid, kind, width, height = struct.unpack("<iiQQ", stream.read(24))
            if kind == 0:
                focal, cx, cy = struct.unpack("<ddd", stream.read(24))
                fx = fy = focal
            elif kind == 1:
                fx, fy, cx, cy = struct.unpack("<dddd", stream.read(32))
            else:
                raise ValueError("preview expects SIMPLE_PINHOLE or PINHOLE without distortion")
            cameras[cid] = (np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]), width, height)
    views = []
    with (sparse / "images.bin").open("rb") as stream:
        count = struct.unpack("<Q", stream.read(8))[0]
        for _ in range(count):
            _, qw, qx, qy, qz, tx, ty, tz, cid = struct.unpack("<idddddddi", stream.read(64))
            name = bytearray()
            while True:
                byte = stream.read(1)
                if not byte:
                    raise ValueError("truncated COLMAP image name")
                if byte == b"\0":
                    break
                name.extend(byte)
            points = struct.unpack("<Q", stream.read(8))[0]
            stream.seek(points * 24, 1)
            k, width, height = cameras[cid]
            r = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            projection = k @ np.column_stack([r, [tx, ty, tz]])
            views.append((name.decode("utf8"), projection, width, height))
    return sorted(views, key=lambda view: view[0])


def thumbnail(rgb: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = min(width / rgb.shape[1], height / rgb.shape[0])
    w, h = max(1, round(rgb.shape[1] * scale)), max(1, round(rgb.shape[0] * scale))
    result = np.zeros((height, width, 3), np.uint8)
    x, y = (width - w) // 2, (height - h) // 2
    result[y : y + h, x : x + w] = cv2.resize(rgb, (w, h))
    return result


def preview(
    root: Path, mesh_path: Path, out: Path, constrain: bool = False, support_threshold: float = 0.95
) -> dict:
    from boardgen.evaluation.benchmark3d import file_hash, render_textured, write_json

    data = json.loads((root / "dataset.json").read_text())
    views = read_colmap(root / "train")
    fit_names = {f"view_{i:04d}.png" for i in data["fit"]}
    if {v[0] for v in views} != fit_names:
        raise ValueError("SfM must register all fit views; partial captures are not accepted")
    mesh = trimesh.load(mesh_path, force="mesh")
    out.mkdir(parents=True, exist_ok=True)
    original_faces = len(mesh.faces)
    if not 0 < support_threshold <= 1:
        raise ValueError("mask support must be in (0, 1]")
    if constrain:
        points_h = np.column_stack([mesh.vertices, np.ones(len(mesh.vertices))])
        support = np.zeros(len(mesh.vertices), np.int32)
        for name, projection, width, height in views:
            hp = points_h @ projection.T
            xy = np.rint(hp[:, :2] / np.maximum(hp[:, 2:3], 1e-12)).astype(int)
            valid = (hp[:, 2] > 0) & (xy[:, 0] >= 0) & (xy[:, 0] < width)
            valid &= (xy[:, 1] >= 0) & (xy[:, 1] < height)
            idx = np.flatnonzero(valid)
            alpha = cv2.imread(str(root / "reference/masks" / name), 0)
            mask = cv2.dilate((alpha >= 128).astype(np.uint8), np.ones((3, 3), np.uint8))
            support[idx] += mask[xy[idx, 1], xy[idx, 0]] > 0
        keep = np.all((support / len(views) >= support_threshold)[mesh.faces], axis=1)
        if not keep.any():
            raise ValueError("no triangles agree with fit silhouettes")
        mesh.update_faces(keep)
        mesh.remove_unreferenced_vertices()
        mesh.export(out / "mesh.glb")
    tiles = []
    for pos in np.linspace(0, len(views) - 1, 8).astype(int):
        name, projection, width, height = views[pos]
        scale = min(1, 400 / max(width, height))
        w, h = max(1, round(width * scale)), max(1, round(height * scale))
        preview_projection = np.diag([w / width, h / height, 1]) @ projection
        pred = render_textured(mesh, preview_projection, w, h)
        target = cv2.cvtColor(cv2.imread(str(root / "reference/images" / name)), cv2.COLOR_BGR2RGB)
        alpha = cv2.imread(str(root / "reference/masks" / name), 0)
        target = (target * (alpha[:, :, None] / 255)).astype(np.uint8)
        target = cv2.resize(target, (w, h), interpolation=cv2.INTER_AREA)
        pair = np.hstack([thumbnail(target, 160, 200), thumbnail(pred, 160, 200)])
        cv2.putText(
            pair, f"{name}: input / mesh", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1
        )
        tiles.append(pair)
    sheet = np.vstack([np.hstack(tiles[:4]), np.hstack(tiles[4:])])
    cv2.imwrite(str(out / "preview.png"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    row = {
        "source_mesh": str(mesh_path),
        "source_mesh_sha256": file_hash(mesh_path),
        "faces_before": original_faces,
        "faces": len(mesh.faces),
        "vertices": len(mesh.vertices),
        "registered_views": len(views),
        "fit": data["fit"],
        "heldout": data["heldout"],
        "constraint": f"fit-mask-face-culling, support >= {support_threshold}, tolerance 1 pixel"
        if constrain
        else "none",
        "camera_sha256": file_hash(root / "train/sparse/0/images.bin"),
        "bounds": mesh.bounds.tolist(),
        "watertight": bool(mesh.is_watertight),
        "assessment": "Qualitative fit-view preview only. No held-out poses or geometric ground truth.",
        "scale": "SfM arbitrary units; not metric calibrated",
        "preview_max_side": 400,
    }
    write_json(out / "review.json", row)
    return row


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Inspect real reconstructed assets using their fit-only COLMAP cameras."
    )
    parser.add_argument("root", type=Path)
    parser.add_argument("mesh", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--constrain", action="store_true")
    parser.add_argument("--support", type=float, default=0.95)
    args = parser.parse_args()
    print(json.dumps(preview(args.root, args.mesh, args.out, args.constrain, args.support)))

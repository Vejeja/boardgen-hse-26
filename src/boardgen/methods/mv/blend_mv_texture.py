import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pymeshlab
import trimesh
from scipy.spatial import cKDTree

from boardgen.evaluation.benchmark3d import file_hash, write_json
from boardgen.evaluation.benchmark_surface import render_geometry


def area_weighted_normals(mesh):
    triangles = mesh.vertices[mesh.faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals = np.zeros_like(mesh.vertices)
    for corner in range(3):
        np.add.at(normals, mesh.faces[:, corner], cross)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-15)
    return normals


def sample_pixels(image, coordinates, interpolation, border=0):
    count = len(coordinates)
    if not count:
        return np.empty((0, 3)) if image.ndim == 3 else np.empty(0)
    padded = np.zeros(((count + 255) // 256 * 256, 2), dtype="float32")
    padded[:count] = coordinates
    values = cv2.remap(
        image,
        padded[:, 0].reshape(-1, 256),
        padded[:, 1].reshape(-1, 256),
        interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border,
    )
    return values.reshape(-1, 3)[:count] if image.ndim == 3 else values.ravel()[:count]


def raster_atlas(mesh, colors, size):
    uv = mesh.visual.uv.copy()
    uv[:, 1] = 1 - uv[:, 1]
    xy = uv * (size - 1)
    image = np.broadcast_to(np.median(colors, axis=0), (size, size, 3)).astype("float32").copy()
    valid = np.zeros((size, size), dtype="uint8")
    for face in mesh.faces:
        p = xy[face]
        lo = np.maximum(0, np.floor(p.min(0)).astype(int))
        hi = np.minimum(size - 1, np.ceil(p.max(0)).astype(int))
        matrix = np.vstack([p.T, np.ones(3)])
        if abs(np.linalg.det(matrix)) < 1e-8:
            continue
        xx, yy = np.meshgrid(np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1))
        x, y = xx.ravel(), yy.ravel()
        bary = np.column_stack([x, y, np.ones(len(x))]) @ np.linalg.inv(matrix).T
        keep = np.all(bary >= -1e-6, axis=1)
        if keep.any():
            image[y[keep], x[keep]] = bary[keep] @ colors[face]
            valid[y[keep], x[keep]] = 1
        else:
            x, y = np.rint(p.mean(0)).astype(int).clip(0, size - 1)
            image[y, x] = colors[face].mean(0)
            valid[y, x] = 1

    distance, labels = cv2.distanceTransformWithLabels(
        1 - valid, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL
    )
    band = (valid == 0) & (distance <= 4)
    image[band] = image[valid > 0][labels[band] - 1]
    return np.clip(image, 0, 255).astype("uint8")


def main():
    parser = argparse.ArgumentParser(
        description="Coherent photo colour on fixed vertices, baked through the existing UV atlas.\n\nVisible views are weighted by smooth incidence, rather than independently chosen\nfor each small triangle. Only fit images are used. Geometry/normals/UV stay fixed."
    )
    parser.add_argument("root", type=Path)
    parser.add_argument("--incidence-power", type=float, default=10)
    parser.add_argument(
        "--mask-margin", type=int, default=0, help="Inset photo support masks by this many pixels"
    )
    parser.add_argument(
        "--bright-quantile",
        type=float,
        default=1.0,
        help="Optional upper foreground luminance quantile for background leakage rejection",
    )
    args = parser.parse_args()
    if not 0 <= args.mask_margin <= 8 or not 0.8 <= args.bright_quantile <= 1:
        parser.error("mask-margin must be 0..8 and bright-quantile 0.8..1")
    root = args.root.resolve()
    started = time.perf_counter()
    result = json.loads((root / "result.json").read_text())
    canonical = trimesh.load(result["source"], force="mesh")
    photo = trimesh.load(root / "photo.glb", force="mesh")
    camera_data = json.loads((root / "cameras.json").read_text())
    spec = (root / "colmap/sparse/cameras.txt").read_text().split()
    w, h = int(spec[2]), int(spec[3])
    fx, fy, cx, cy = map(float, spec[4:8])
    k = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(canonical.vertices, canonical.faces))
    ms.meshing_decimation_quadric_edge_collapse(targetfacenum=5000, preservenormal=True)
    proxy = trimesh.Trimesh(
        ms.current_mesh().vertex_matrix(), ms.current_mesh().face_matrix(), process=False
    )
    total = np.zeros((len(canonical.vertices), 3))
    weights = np.zeros(len(canonical.vertices))
    samples = []
    for camera in camera_data:
        image = cv2.imread(str(root / "colmap/images" / camera["name"]))[:, :, ::-1]
        mask = (
            cv2.imread(str(root / "colmap/images" / (Path(camera["name"]).stem + ".mask.png")), 0)
            > 128
        )
        if args.mask_margin:
            width = 2 * args.mask_margin + 1
            mask = cv2.erode(mask.astype("uint8"), np.ones((width, width), "uint8")) > 0
        if not mask.any():
            raise ValueError(f"Photo support mask is empty: {camera['name']}")
        lum = image.mean(2)
        median = np.median(lum[mask])
        mad = np.median(np.abs(lum[mask] - median))

        bright_limit = median + max(60, 6 * mad)
        if args.bright_quantile < 1:
            bright_limit = min(bright_limit, float(np.quantile(lum[mask], args.bright_quantile)))
        rotation = np.array(camera["rotation"])
        translation = np.array(camera["translation"])
        pos = -rotation.T @ translation
        directions = pos - canonical.vertices
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        incidence = np.einsum("ij,ij->i", canonical.vertex_normals, directions)
        projected = canonical.vertices @ rotation.T + translation
        uv = projected[:, :2] / projected[:, 2:3]
        uv *= [fx, fy]
        uv += [cx, cy]
        x, y = np.rint(uv).astype(int).T
        inside = (projected[:, 2] > 0) & (x >= 0) & (x < w) & (y >= 0) & (y < h)
        ids = np.flatnonzero(inside)
        scale = 320 / max(w, h)
        sw, sh = round(w * scale), round(h * scale)
        small_k = np.diag([sw / w, sh / h, 1]) @ k
        _, depth = render_geometry(
            proxy, small_k @ np.column_stack([rotation, translation]), rotation, sw, sh
        )

        sample_depth = sample_pixels(
            depth.astype("float32"), uv * [sw / w, sh / h], cv2.INTER_NEAREST, float("inf")
        )
        visible = np.isfinite(sample_depth) & (projected[:, 2] <= sample_depth + 0.025)
        support = (
            mask[y[ids], x[ids]]
            & (lum[y[ids], x[ids]] <= bright_limit)
            & visible[ids]
            & (incidence[ids] > 0.15)
        )
        ids = ids[support]

        weight = np.maximum(incidence[ids], 0) ** args.incidence_power
        color = sample_pixels(image, uv[ids], cv2.INTER_LINEAR)
        total[ids] += color * weight[:, None]
        weights[ids] += weight
        samples.append(
            {
                "name": camera["name"],
                "supported_vertices": len(ids),
                "bright_limit": float(bright_limit),
            }
        )
        print(camera["name"], len(ids), flush=True)
    known = weights > 1e-8
    if not known.any():
        raise ValueError("No vertices have supported photograph colours")
    colors = np.zeros_like(total)
    colors[known] = total[known] / weights[known, None]

    unknown = np.flatnonzero(~known)
    if len(unknown):
        distance, nearest = cKDTree(canonical.vertices[known]).query(canonical.vertices[unknown])
        colors[unknown] = np.median(colors[known], axis=0)
        close = distance < 0.025
        colors[unknown[close]] = colors[known][nearest[close]]
    distance, original_ids = cKDTree(canonical.vertices).query(photo.vertices)
    if distance.max() > 1e-7:
        raise ValueError("photo surface differs from frozen geometry")
    np.save(root / "blended-vertex-colours.npy", colors)
    size = max(2048, photo.visual.material.baseColorTexture.width)
    atlas = raster_atlas(photo, colors[original_ids], size)
    cv2.imwrite(str(root / "blended-atlas.png"), atlas[:, :, ::-1])
    from PIL import Image

    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=Image.fromarray(atlas),
        baseColorFactor=[255, 255, 255, 255],
        metallicFactor=0,
        roughnessFactor=0.8,
    )
    photo.visual.material = material

    photo.vertex_normals = area_weighted_normals(canonical)[original_ids]
    photo.export(root / "photo-blended.glb")
    write_json(
        root / "blending.json",
        {
            "seconds": time.perf_counter() - started,
            "method": f"Depth-tested smooth incidence^{args.incidence_power:g} multiview vertex colours baked into fixed UV; no new lighting",
            "incidence_power": args.incidence_power,
            "mask_margin_pixels": args.mask_margin,
            "bright_quantile": args.bright_quantile,
            "known_vertices": int(known.sum()),
            "total_vertices": len(known),
            "unknown_vertices": len(unknown),
            "atlas_size": size,
            "visibility_proxy_faces": len(proxy.faces),
            "depth_tolerance_height_units": 0.025,
            "views": samples,
            "source_sha256": result["source_sha256"],
            "photo_sha256": file_hash(root / "photo-blended.glb"),
            "limitation": "Approximate cameras and geometry; no invented underside detail. Raw baked shadows persist.",
        },
    )
    print(root / "photo-blended.glb", flush=True)


if __name__ == "__main__":
    main()

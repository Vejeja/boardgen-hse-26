from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import mapbox_earcut
import networkx as nx
import numpy as np
import pymeshlab
import trimesh
from PIL import Image
from scipy.spatial import cKDTree

from boardgen.evaluation.benchmark3d import file_hash, write_json


def close_small_holes(mesh, max_edges):
    topology = trimesh.Trimesh(mesh.vertices.copy(), mesh.faces.copy(), process=False)
    topology.merge_vertices()
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(topology.vertices, topology.faces))
    ms.meshing_close_holes(maxholesize=max_edges, selfintersection=True)
    repaired = ms.current_mesh()
    original_faces = {tuple(sorted(f)) for f in topology.faces}
    added = np.array([f for f in repaired.face_matrix() if tuple(sorted(f)) not in original_faces])
    if not len(added):
        return mesh, 0

    if not np.array_equal(repaired.vertex_matrix()[: len(topology.vertices)], topology.vertices):
        raise ValueError("Hole closure moved observed vertices")
    positions = repaired.vertex_matrix()[added].reshape(-1, 3)
    _, nearest = cKDTree(mesh.vertices).query(positions)
    vertices = np.vstack([mesh.vertices, positions])
    faces = np.vstack([mesh.faces, np.arange(len(positions)).reshape(-1, 3) + len(mesh.vertices)])
    visual = trimesh.visual.TextureVisuals(
        uv=np.vstack([mesh.visual.uv, mesh.visual.uv[nearest]]), material=mesh.visual.material
    )
    return trimesh.Trimesh(vertices, faces, visual=visual, process=False), len(added)


def close_base_holes(mesh, ceiling, max_edges=1500):
    if not np.isfinite(ceiling) or not 3 <= max_edges <= 5000:
        raise ValueError("Invalid base closure bounds")
    topology = trimesh.Trimesh(mesh.vertices.copy(), mesh.faces.copy(), process=False)
    topology.merge_vertices()
    counts = np.bincount(topology.edges_unique_inverse, minlength=len(topology.edges_unique))
    indices = np.flatnonzero(counts == 1)
    _, first_indices = np.unique(topology.edges_unique_inverse, return_index=True)
    edges = topology.edges[first_indices[indices]]
    graph = nx.Graph()
    graph.add_edges_from(edges)
    loops = nx.cycle_basis(graph)
    uses = {}
    for loop in loops:
        for a, b in zip(loop, np.roll(loop, -1), strict=True):
            edge = tuple(sorted([int(a), int(b)]))
            uses[edge] = uses.get(edge, 0) + 1
    directed_boundary = set(map(tuple, edges.tolist()))
    added, rejected, closed = [], 0, 0
    for ring in loops:
        if len(ring) > max_edges or topology.vertices[ring, 1].max() > ceiling:
            rejected += 1
            continue

        if any(
            uses[tuple(sorted([int(a), int(b)]))] != 1
            for a, b in zip(ring, np.roll(ring, -1), strict=True)
        ):
            rejected += 1
            continue
        if (ring[0], ring[1]) not in directed_boundary:
            ring = ring[::-1]
        positions = topology.vertices[ring]
        center = positions.mean(axis=0)
        _, singular, basis = np.linalg.svd(positions - center, full_matrices=False)
        if singular[1] < 1e-10 or singular[2] > singular[1] * 0.5:
            rejected += 1
            continue
        coordinates = np.ascontiguousarray((positions - center) @ basis[:2].T)
        cap = mapbox_earcut.triangulate_float64(
            coordinates, np.array([len(ring)], dtype=np.uint32)
        ).reshape(-1, 3)
        if len(cap) != len(ring) - 2:
            rejected += 1
            continue
        area = (
            abs(
                np.sum(
                    coordinates[:, 0] * np.roll(coordinates[:, 1], -1)
                    - coordinates[:, 1] * np.roll(coordinates[:, 0], -1)
                )
            )
            / 2
        )
        triangles = coordinates[cap]
        a, b = triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
        cap_area = np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]).sum() / 2
        if area <= 1e-12 or not np.isclose(cap_area, area, rtol=1e-5, atol=1e-12):
            rejected += 1
            continue

        oriented = {
            (int(a), int(b)) for face in cap for a, b in zip(face, np.roll(face, -1), strict=True)
        }
        if (0, 1) in oriented:
            cap = cap[:, ::-1]

        cap_mesh = trimesh.Trimesh(positions, cap, process=False)
        cap_counts = np.bincount(
            cap_mesh.edges_unique_inverse, minlength=len(cap_mesh.edges_unique)
        )
        boundary = {tuple(e) for e in cap_mesh.edges_unique[cap_counts == 1]}
        expected = {tuple(sorted([i, (i + 1) % len(ring)])) for i in range(len(ring))}
        if boundary != expected:
            rejected += 1
            continue
        added.append(positions[cap].reshape(-1, 3))
        closed += 1
    report = {
        "added_faces": int(sum(len(v) // 3 for v in added)),
        "closed_caps": closed,
        "rejected_caps": rejected,
        "ceiling_y": float(ceiling),
        "max_hole_edges": max_edges,
        "scope": "Complete simple boundaries in base band only; body and original UV unchanged",
    }
    if not added:
        return mesh, report
    positions = np.vstack(added)
    _, nearest = cKDTree(mesh.vertices).query(positions)
    if mesh.visual.kind == "texture" and mesh.visual.uv is not None:
        visual = trimesh.visual.TextureVisuals(
            uv=np.vstack([mesh.visual.uv, mesh.visual.uv[nearest]]),
            material=mesh.visual.material,
        )
    else:
        colours = mesh.visual.vertex_colors
        visual = trimesh.visual.ColorVisuals(vertex_colors=np.vstack([colours, colours[nearest]]))
    triangles = positions.reshape(-1, 3, 3)
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-20)
    result = trimesh.Trimesh(
        vertices=np.vstack([mesh.vertices, positions]),
        vertex_normals=np.vstack([mesh.vertex_normals, np.repeat(normals, 3, axis=0)]),
        faces=np.vstack(
            [
                mesh.faces,
                np.arange(len(positions)).reshape(-1, 3) + len(mesh.vertices),
            ]
        ),
        visual=visual,
        process=False,
    )
    return result, report


def negative_space_faces(mesh, transform, projection, mask, height_range):
    centers = trimesh.transformations.transform_points(
        mesh.triangles_center, np.linalg.inv(transform)
    )
    hp = np.column_stack([centers, np.ones(len(centers))]) @ projection.T
    xy = np.rint(hp[:, :2] / np.maximum(hp[:, 2:], 1e-9)).astype(int)
    h, w = mask.shape
    valid = (hp[:, 2] > 0) & (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
    inside = np.zeros(len(centers), bool)
    indices = np.flatnonzero(valid)
    inside[indices] = mask[xy[indices, 1], xy[indices, 0]]
    height = mesh.triangles_center[:, 1]
    return inside & (height > height_range[0]) & (height < height_range[1])


def flatten_supplemental_base(scene):
    bodies = [mesh for name, mesh in scene.geometry.items() if "body" in name]
    bases = [
        (name, mesh) for name, mesh in scene.geometry.items() if "inferred_boundary_closure" in name
    ]
    if not bodies or not bases:
        raise ValueError(
            "Flat underside repair requires a separate native body and supplemental base"
        )
    plane = min(mesh.bounds[0, 1] for mesh in bodies)
    minimum, maximum = scene.bounds[:, 1]
    if not minimum <= plane <= minimum + 0.15 * (maximum - minimum):
        raise ValueError("Inferred base plane is outside the lowest 15%")
    edits = []
    for name, base in bases:
        below = base.vertices[:, 1] < plane
        base.vertices[below, 1] = plane
        base.update_faces(base.nondegenerate_faces())
        base.remove_unreferenced_vertices()
        edits.append(
            {
                "part": name,
                "clamped_vertices": int(below.sum()),
                "plane_y_before_normalization": float(plane),
                "reason": "Unseen flat underside prior at the native body's minimum; body geometry and UV unchanged",
            }
        )
    return edits


def repair(
    source,
    output,
    trim_supplement=False,
    hole_edges=0,
    annotations=None,
    dataset=None,
    flat_base=False,
):
    output.mkdir(parents=True, exist_ok=False)
    result = json.loads((source / "result.json").read_text())
    scene = trimesh.load(source / "photo.glb", force="scene", process=False)
    log = []
    if flat_base:
        log.extend(flatten_supplemental_base(scene))
    exclusions = []
    if annotations:
        from boardgen.evaluation.benchmark_preview import read_colmap

        data = json.loads((dataset / "dataset.json").read_text())
        projections = {name: p for name, p, _, _ in read_colmap(dataset / "train")}
        for annotation in annotations:
            view = annotation["view"]
            name = f"view_{view:04d}.png"
            if view not in data["fit"] or name not in projections:
                raise ValueError("Negative-space annotations must use fit photographs")
            path = dataset / "reference/images" / name
            image = Image.open(path)
            mask = np.zeros((image.height, image.width), "uint8")
            cv2.fillPoly(mask, [np.array(annotation["polygon"], dtype="int32")], 1)
            annotation = {**annotation, "photograph_sha256": file_hash(path)}
            exclusions.append((annotation, mask.astype(bool), projections[name]))
    for name, mesh in list(scene.geometry.items()):
        old_faces = len(mesh.faces)
        if trim_supplement and "supplement" in name:
            scene.delete_geometry(name)
            log.append(
                {
                    "part": name,
                    "removed_faces": old_faces,
                    "reason": "Silhouette supplement can bridge observed negative spaces",
                }
            )
            continue
        if exclusions and ("body" in name or "supplement" in name):
            remove = np.zeros(len(mesh.faces), bool)
            for annotation, mask, projection in exclusions:
                remove |= negative_space_faces(
                    mesh,
                    np.array(result["transform"]),
                    projection,
                    mask,
                    annotation["normalized_height_range"],
                )
            mesh.update_faces(~remove)
            mesh.remove_unreferenced_vertices()
            log.append(
                {
                    "part": name,
                    "removed_faces": int(remove.sum()),
                    "reason": "Manually verified empty space above the base in fit photographs",
                }
            )
        if hole_edges and "body" in name:
            repaired, added = close_small_holes(mesh, hole_edges)
            scene.geometry[name] = repaired
            log.append({"part": name, "added_faces": added, "max_hole_edges": hole_edges})
    lo, hi = scene.bounds
    normalization = np.eye(4)
    normalization[:3, :3] /= hi[1] - lo[1]
    normalization[:3, 3] = np.array([-(lo[0] + hi[0]) / 2, -lo[1], -(lo[2] + hi[2]) / 2]) / (
        hi[1] - lo[1]
    )
    scene.apply_transform(normalization)
    result["transform"] = (normalization @ np.array(result["transform"])).tolist()
    scene.export(output / "photo.glb")
    history = list(result.get("repair_history", []))
    if result.get("localized_repairs"):
        history.append(
            {
                "source_mesh_sha256": file_hash(source / "photo.glb"),
                "edits": result["localized_repairs"],
                "annotations": result.get("photographic_negative_space_annotations", []),
            }
        )
    result.update(
        photo_sha256=file_hash(output / "photo.glb"),
        repair_source=str(source.resolve()),
        repair_source_sha256=file_hash(source / "photo.glb"),
        localized_repairs=log,
        repair_history=history,
        photographic_negative_space_annotations=[a for a, _, _ in exclusions],
        assessment="Observed mesh repair candidate. Original body positions and photo UV faces retained; small boundary closure is an interpolation prior. Silhouette supplements may be removed to preserve negative spaces. Requires visual comparison, including actual source photos.",
    )
    write_json(output / "result.json", result)
    return log


def main():
    parser = argparse.ArgumentParser(
        description="Create reversible mesh repair candidates; preserve photos and original UV charts."
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--trim-supplement", action="store_true")
    parser.add_argument("--hole-edges", type=int, default=0)
    parser.add_argument(
        "--flat-base",
        action="store_true",
        help="Flatten only the supplemental underside to the native body's lower bound",
    )
    parser.add_argument(
        "--annotations", type=Path, help="JSON list of verified empty-space photo regions"
    )
    parser.add_argument(
        "--dataset", type=Path, help="Orbit dataset containing the annotated fit photos"
    )
    args = parser.parse_args()
    if not 0 <= args.hole_edges <= 100:
        parser.error("hole-edges must be between 0 and 100")
    if bool(args.annotations) != bool(args.dataset):
        parser.error("annotations and dataset must be supplied together")
    print(
        repair(
            args.source.resolve(),
            args.output.resolve(),
            args.trim_supplement,
            args.hole_edges,
            json.loads(args.annotations.read_text()) if args.annotations else None,
            args.dataset.resolve() if args.dataset else None,
            args.flat_base,
        )
    )


if __name__ == "__main__":
    main()

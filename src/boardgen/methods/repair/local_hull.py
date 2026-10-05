import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import trimesh
from PIL import Image

from boardgen.evaluation.benchmark3d import file_hash, write_json
from boardgen.evaluation.benchmark_preview import read_colmap
from boardgen.methods.mv.blend_mv_texture import area_weighted_normals
from boardgen.methods.orbit.hull_bounds import frustum_bounds
from boardgen.reconstruction.recon3d import carve_visual_hull_calibrated


def weld_normals(mesh):
    _, ids = np.unique(np.round(mesh.vertices, 9), axis=0, return_inverse=True)
    positions = np.zeros((ids.max() + 1, 3))
    positions[ids] = mesh.vertices
    welded = trimesh.Trimesh(positions, ids[mesh.faces], process=False)
    mesh.vertex_normals = area_weighted_normals(welded)[ids]


def flatten_native_bottom(body, plane_y):
    minimum, maximum = body.bounds[:, 1]
    if not np.isfinite(plane_y) or not minimum <= plane_y <= minimum + 0.15 * (maximum - minimum):
        raise ValueError("Inferred bottom plane must remain within the lowest 15% of native height")
    before = body.vertices.copy()
    below = body.vertices[:, 1] < plane_y
    body.vertices[below, 1] = plane_y
    if not np.array_equal(before[~below], body.vertices[~below]):
        raise ValueError("Bottom repair altered vertices above the inferred plane")
    metadata = {
        "plane_y": plane_y,
        "clamped_native_vertices": int(below.sum()),
        "vertices_above_plane_unchanged": True,
        "native_UV_preserved": True,
        "assumption": "Explicit unseen flat-bottom prior, estimated from the visible outer base rim. This edits only the lowest native region; it is not observed underside ground truth.",
    }
    body.update_faces(body.nondegenerate_faces())
    body.remove_unreferenced_vertices()
    return metadata


def main():
    ap = argparse.ArgumentParser(
        description="Preserve the native Spirula body/UV; supplement only missing outer contours."
    )
    ap.add_argument("root", type=Path)
    ap.add_argument("run", type=Path)
    ap.add_argument("--grid", type=int, default=192)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--consensus", type=float, default=0.97)
    ap.add_argument(
        "--conservative-bounds",
        action="store_true",
        help="derive carving volume from fit-view frusta instead of incomplete trained geometry",
    )
    ap.add_argument(
        "--include-internal",
        action="store_true",
        help="also supplement missing raster foreground inside the predicted outline; fit-mask hull still constrains every patch",
    )
    ap.add_argument(
        "--base-band",
        type=float,
        default=0,
        help="retain the full fit-mask hull in the lowest fraction of the carving volume for an observed thin base",
    )
    ap.add_argument(
        "--close-base",
        action="store_true",
        help="close boundary loops only in the lower hull band; inferred underside closure, not observed ground truth",
    )
    ap.add_argument(
        "--flat-inferred-bottom",
        action="store_true",
        help="clamp only supplemental base underside to the native body's estimated minimum Y; explicit unseen-bottom prior",
    )
    ap.add_argument(
        "--base-plane-y",
        type=float,
        help="explicit unseen-bottom prior: clamp native vertices below this world-Y plane; preserve all vertices above it and native UV",
    )
    a = ap.parse_args()
    if not 0.9 <= a.consensus <= 1:
        ap.error("consensus must be between 0.9 and 1")
    if not 0 <= a.base_band <= 0.15:
        ap.error("base-band must be between 0 and 0.15")
    if a.close_base and not a.base_band:
        ap.error("close-base requires a nonzero base-band")
    if a.flat_inferred_bottom and not a.close_base:
        ap.error("flat-inferred-bottom requires close-base")
    root = a.root.resolve()
    run = a.run.resolve()
    out = a.output or run / "local-hull"
    out.mkdir(exist_ok=False)
    started = time.perf_counter()
    body = trimesh.load(run / "constrained.glb", force="mesh")
    before = body.vertices.copy()
    base_plane_edit = None
    if a.base_plane_y is not None:
        base_plane_edit = flatten_native_bottom(body, a.base_plane_y)
    views = read_colmap(root / "train")
    cal = json.loads((root / "calibration.json").read_text())
    params = {f"view_{c['index']:04d}.png": c for c in cal["camera_parameters"]}
    angles = np.array([params[v[0]]["yaw_deg"] % 360 for v in views])
    selected = sorted(
        {
            int(np.argmin(abs((angles - angle + 180) % 360 - 180)))
            for angle in np.linspace(0, 360, 32, endpoint=False)
        }
    )
    views = [views[i] for i in selected]
    masks = np.stack([cv2.imread(str(root / "reference/masks" / v[0]), 0) > 128 for v in views])
    lo, hi = body.bounds
    pad = (hi - lo) * 0.035
    bounds = np.column_stack([lo - pad, hi + pad]).tolist()
    if a.conservative_bounds:
        bounds = frustum_bounds(masks, [v[1] for v in views]).tolist()
    verts, faces = carve_visual_hull_calibrated(
        masks,
        [v[1] for v in views],
        grid=a.grid,
        consensus=a.consensus,
        bounds=bounds,
        samples_per_axis=1,
    )
    hull = trimesh.Trimesh(verts, faces, process=False)

    if hull.volume < 0:
        hull.invert()
    hull.export(out / "full-hull.ply")
    votes = np.zeros(len(verts), int)
    ph = np.column_stack([verts, np.ones(len(verts))])
    bh = np.column_stack([body.vertices, np.ones(len(body.vertices))])

    from boardgen.methods.orbit.geometry_proxy import proxy

    geometry = proxy(body, 18000)
    bv, bf = geometry.vertices, geometry.faces
    bh = np.column_stack([bv, np.ones(len(bv))])
    for (_name, proj, w, h), mask in zip(views, masks, strict=True):
        scale = min(1, 256 / max(w, h))
        ww, hh = round(w * scale), round(h * scale)
        p = np.diag([ww / w, hh / h, 1]) @ proj
        hp = bh @ p.T
        uv = np.rint(hp[:, :2] / np.maximum(hp[:, 2:], 1e-9)).clip(-10000, 10000).astype("int32")
        pred = np.zeros((hh, ww), "uint8")
        for triangle in uv[bf[(hp[bf, 2] > 0).all(1)]]:
            cv2.fillConvexPoly(pred, triangle, 1)
        pred = cv2.dilate(pred, np.ones((3, 3), "uint8"))
        external = (1 - pred).astype("uint8")
        cv2.floodFill(external, None, (0, 0), 2)
        target = cv2.resize(mask.astype("uint8"), (ww, hh), interpolation=cv2.INTER_NEAREST) > 0
        missing = target & ((pred == 0) if a.include_internal else (external == 2))
        hp = ph @ p.T
        xy = np.rint(hp[:, :2] / np.maximum(hp[:, 2:], 1e-9)).astype(int)
        valid = (
            (hp[:, 2] > 0) & (xy[:, 0] >= 0) & (xy[:, 0] < ww) & (xy[:, 1] >= 0) & (xy[:, 1] < hh)
        )
        ids = np.flatnonzero(valid)
        votes[ids] += missing[xy[ids, 1], xy[ids, 0]]
    keep = (votes >= 2)[hull.faces].sum(1) >= 2
    base_limit = bounds[1][0] + a.base_band * (bounds[1][1] - bounds[1][0])
    base = None
    base_closure = None
    if a.base_band:
        base_faces = (hull.vertices[hull.faces, 1] <= base_limit).all(1)
        if a.close_base:
            import pymeshlab

            base = hull.copy()
            base.update_faces(base_faces)
            base.remove_unreferenced_vertices()
            if not len(base.faces):
                raise ValueError("The selected hull band contains no base faces")
            ms = pymeshlab.MeshSet()
            ms.add_mesh(pymeshlab.Mesh(base.vertices, base.faces))
            ms.meshing_close_holes(maxholesize=10000, selfintersection=True)
            closed = trimesh.Trimesh(
                ms.current_mesh().vertex_matrix(), ms.current_mesh().face_matrix(), process=False
            )
            base_closure = {
                "faces_before": len(base.faces),
                "faces_after": len(closed.faces),
                "watertight": bool(closed.is_watertight),
                "max_hole_edges": 10000,
                "assumption": "Unseen base boundary closure is a geometric prior. Original body and UV are preserved; this is not recovered underside ground truth or a watertight Boolean union.",
            }
            base = closed
            if a.flat_inferred_bottom:
                below = base.vertices[:, 1] < lo[1]
                base_closure["inferred_bottom_plane_y"] = float(lo[1])
                base_closure["clamped_supplement_vertices"] = int(below.sum())
                base.vertices[below, 1] = lo[1]
                base.merge_vertices()
                base.update_faces(base.nondegenerate_faces())
                base.remove_unreferenced_vertices()
                base_closure["assumption"] += (
                    " Supplemental underside is clamped to the native minimum-Y estimate; no original body vertices are changed."
                )
            base.export(out / "base-closure.ply")
            keep &= ~base_faces
        else:
            keep |= base_faces
    hull.update_faces(keep)
    hull.remove_unreferenced_vertices()
    hull.export(out / "patch.ply")
    colors = []
    for name, _, _, _ in views:
        rgb = cv2.imread(str(root / "reference/images" / name))[:, :, ::-1]
        mask = cv2.imread(str(root / "reference/masks" / name), 0) > 220
        colors.append(np.median(rgb[mask], axis=0))
    colour = np.median(colors, axis=0).astype("uint8")
    weld_normals(body)
    hull.visual = trimesh.visual.TextureVisuals(
        uv=np.full((len(hull.vertices), 2), 0.5),
        material=trimesh.visual.material.PBRMaterial(
            baseColorTexture=Image.new("RGB", (32, 32), tuple(colour)),
            metallicFactor=0,
            roughnessFactor=0.85,
        ),
    )

    if len(hull.faces):
        hull.vertex_normals = area_weighted_normals(hull)
    if base is not None:
        base.visual = trimesh.visual.TextureVisuals(
            uv=np.full((len(base.vertices), 2), 0.5), material=hull.visual.material
        )
        base.vertex_normals = area_weighted_normals(base)
    if base_plane_edit is None and not np.array_equal(before, body.vertices):
        raise ValueError("Hull supplement altered body vertices")
    combined_bounds = (
        np.array(
            [np.minimum(body.bounds[0], hull.bounds[0]), np.maximum(body.bounds[1], hull.bounds[1])]
        )
        if len(hull.faces)
        else body.bounds.copy()
    )
    if base is not None:
        combined_bounds[0] = np.minimum(combined_bounds[0], base.bounds[0])
        combined_bounds[1] = np.maximum(combined_bounds[1], base.bounds[1])
    center = combined_bounds.mean(0)
    shift = np.array([-center[0], -combined_bounds[0, 1], -center[2]])
    scale = 1 / (combined_bounds[1, 1] - combined_bounds[0, 1])
    transform = np.eye(4)
    transform[:3, :3] *= scale
    transform[:3, 3] = shift * scale
    body.apply_transform(transform)
    hull.apply_transform(transform)
    if base is not None:
        base.apply_transform(transform)
    scene = trimesh.Scene()
    scene.add_geometry(body, geom_name="Spirula_body_native_UV")
    if len(hull.faces):
        scene.add_geometry(hull, geom_name="local_outline_supplement")
    if base is not None:
        scene.add_geometry(base, geom_name="observed_base_with_inferred_boundary_closure")
    scene.export(out / "photo.glb")
    write_json(
        out / "result.json",
        {
            "seconds": time.perf_counter() - started,
            "body": str(run / "constrained.glb"),
            "body_sha256": file_hash(run / "constrained.glb"),
            "body_faces": len(body.faces),
            "patch_faces": len(hull.faces),
            "body_vertices_unchanged_before_uniform_normalization": base_plane_edit is None,
            "inferred_base_plane_edit": base_plane_edit,
            "grid": a.grid,
            "consensus": a.consensus,
            "bounds_source": "fit-view foreground frusta"
            if a.conservative_bounds
            else "native body bounds with 3.5% margin",
            "views": [v[0] for v in views],
            "hull_boundaries": bounds,
            "minimum_missing_contour_views": 2,
            "include_internal_missing_foreground": a.include_internal,
            "base_band": a.base_band,
            "base_band_upper_y": base_limit if a.base_band else None,
            "base_band_note": "Optional fit-silhouette hull surface at the observed base; no analytic circle or generated relief. Unseen underside and silhouette-only concavities are not recovered ground truth.",
            "base_closure": base_closure,
            "patch_colour_rgb": colour.tolist(),
            "patch_colour_note": "Plain median plastic colour, not recovered photographic texture; native body UV/photo preserved",
            "transform": transform.tolist(),
            "photo_sha256": file_hash(out / "photo.glb"),
            "faces": len(body.faces)
            + len(hull.faces)
            + (len(base.faces) if base is not None else 0),
            "assessment": "Localized contour supplementation only; concavities, gaps and original surface preserved. Patch is a separate surface, not a watertight Boolean union. No floor flattening or axis stretching.",
        },
    )
    print(
        out / "photo.glb",
        len(body.faces),
        "+",
        len(hull.faces),
        "seconds",
        time.perf_counter() - started,
        flush=True,
    )


if __name__ == "__main__":
    main()

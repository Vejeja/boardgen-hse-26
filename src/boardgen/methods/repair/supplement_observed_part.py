import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import trimesh
from PIL import Image

from boardgen.evaluation.benchmark3d import file_hash, write_json
from boardgen.evaluation.benchmark_preview import read_colmap
from boardgen.reconstruction.recon3d import carve_visual_hull_calibrated


def main():
    parser = argparse.ArgumentParser(
        description="Recover a localized missing part from calibrated fit silhouettes, preserving the body."
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("annotation", type=Path)
    args = parser.parse_args()
    source, dataset, output = args.source.resolve(), args.dataset.resolve(), args.output.resolve()
    annotation = json.loads(args.annotation.read_text())
    name = f"view_{annotation['view']:04d}.png"
    data = json.loads((dataset / "dataset.json").read_text())
    if annotation["view"] not in data["fit"]:
        raise ValueError("Part localization must use a fit photograph")
    views = read_colmap(dataset / "train")
    projection = next(v[1] for v in views if v[0] == name)
    image = Image.open(dataset / "reference/images" / name)
    roi = np.zeros((image.height, image.width), "uint8")
    cv2.fillPoly(roi, [np.array(annotation["polygon"], "int32")], 1)
    result = json.loads((source / "result.json").read_text())
    transform = np.array(result["transform"])
    scene = trimesh.load(source / "photo.glb", force="scene", process=False)
    world = scene.to_geometry()
    world.apply_transform(np.linalg.inv(transform))
    hp = np.column_stack([world.vertices, np.ones(len(world.vertices))]) @ projection.T
    xy = np.rint(hp[:, :2] / np.maximum(hp[:, 2:], 1e-9)).astype(int)
    valid = (
        (hp[:, 2] > 0)
        & (xy[:, 0] >= 0)
        & (xy[:, 0] < image.width)
        & (xy[:, 1] >= 0)
        & (xy[:, 1] < image.height)
    )
    ids = np.flatnonzero(valid)
    ids = ids[roi[xy[ids, 1], xy[ids, 0]] > 0]
    if len(ids) < 10:
        raise ValueError("Part localization has too few existing vertices")
    height = np.ptp(world.vertices[:, 1])
    lo, hi = world.vertices[ids].min(0), world.vertices[ids].max(0)
    lo -= height * np.array([0.06, 0.03, 0.06])
    hi += height * np.array([0.06, 0.18, 0.06])
    cal = json.loads((dataset / "calibration.json").read_text())
    params = {f"view_{c['index']:04d}.png": c for c in cal["camera_parameters"]}
    angles = np.array([params[v[0]]["yaw_deg"] for v in views])
    selected = sorted(
        {
            int(np.argmin(abs((angles - angle + 180) % 360 - 180)))
            for angle in np.linspace(0, 360, 32, endpoint=False)
        }
    )
    views = [views[i] for i in selected]
    masks = np.stack([cv2.imread(str(dataset / "reference/masks" / v[0]), 0) > 128 for v in views])
    vertices, faces = carve_visual_hull_calibrated(
        masks,
        [v[1] for v in views],
        grid=256,
        consensus=0.97,
        bounds=np.column_stack([lo, hi]).tolist(),
        samples_per_axis=1,
    )
    hull = trimesh.Trimesh(vertices, faces, process=False)
    if hull.volume < 0:
        hull.invert()
    hp = np.column_stack([hull.triangles_center, np.ones(len(hull.faces))]) @ projection.T
    xy = np.rint(hp[:, :2] / np.maximum(hp[:, 2:], 1e-9)).astype(int)
    valid = (
        (hp[:, 2] > 0)
        & (xy[:, 0] >= 0)
        & (xy[:, 0] < image.width)
        & (xy[:, 1] >= 0)
        & (xy[:, 1] < image.height)
    )
    keep = np.zeros(len(hull.faces), bool)
    ids = np.flatnonzero(valid)
    keep[ids] = roi[xy[ids, 1], xy[ids, 0]] > 0
    hull.update_faces(keep)
    hull.remove_unreferenced_vertices()
    if not len(hull.faces):
        raise ValueError("No observed silhouette part survived")
    hull.visual = trimesh.visual.TextureVisuals(
        uv=np.full((len(hull.vertices), 2), 0.5),
        material=trimesh.visual.material.PBRMaterial(
            baseColorTexture=Image.new("RGB", (32, 32), tuple(result["patch_colour_rgb"])),
            metallicFactor=0,
            roughnessFactor=0.85,
        ),
    )
    hull.apply_transform(transform)
    scene.add_geometry(hull, geom_name="localized_observed_hand_supplement")
    lo, hi = scene.bounds
    normalization = np.eye(4)
    normalization[:3, :3] /= hi[1] - lo[1]
    normalization[:3, 3] = np.array([-(lo[0] + hi[0]) / 2, -lo[1], -(lo[2] + hi[2]) / 2]) / (
        hi[1] - lo[1]
    )
    scene.apply_transform(normalization)
    output.mkdir(parents=True, exist_ok=False)
    scene.export(output / "photo.glb")
    result.update(
        transform=(normalization @ transform).tolist(),
        localized_part={
            "annotation": annotation,
            "photograph_sha256": file_hash(dataset / "reference/images" / name),
            "fit_views": [v[0] for v in views],
            "grid": 256,
            "consensus": 0.97,
            "faces": len(hull.faces),
            "assessment": "Localized silhouette-derived hand surface; native body/UV retained. Thickness and hidden surfaces are silhouette assumptions.",
        },
        repair_source=str(source),
        repair_source_sha256=file_hash(source / "photo.glb"),
        photo_sha256=file_hash(output / "photo.glb"),
    )
    write_json(output / "result.json", result)
    print(output, len(hull.faces), flush=True)


if __name__ == "__main__":
    main()

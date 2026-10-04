import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pymeshlab
import trimesh
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from boardgen.camera.camera_audit import read_model
from boardgen.evaluation.benchmark3d import file_hash, run_command, write_json
from boardgen.studio.environment import TOOLS as TOOL_ROOT


def camera_pose(parameters, k, width, height):
    yaw, elevation, roll, log_distance, dx, dy = parameters
    az, el = np.radians([yaw, elevation])
    distance = np.exp(log_distance)
    center = np.array([0.0, 0.5, 0.0])
    direction = np.array([np.sin(az) * np.cos(el), np.sin(el), np.cos(az) * np.cos(el)])
    position = center + distance * direction
    forward = -direction
    right = np.cross(forward, [0, 1, 0])
    right /= np.linalg.norm(right)
    rotation = np.stack([right, np.cross(forward, right), forward])
    rotation = Rotation.from_euler("z", roll, degrees=True).as_matrix() @ rotation
    translation = -rotation @ position
    translation += [dx * width / k[0, 0] * distance, dy * height / k[1, 1] * distance, 0]
    return rotation, translation


def silhouette(vertices, faces, projection, width, height):
    hp = np.column_stack([vertices, np.ones(len(vertices))]) @ projection.T
    xy = np.rint(hp[:, :2] / np.maximum(hp[:, 2:3], 1e-9)).clip(-10000, 10000).astype("int32")
    predicted = np.zeros((height, width), "uint8")
    for triangle in xy[faces[(hp[faces, 2] > 0).all(1)]]:
        cv2.fillConvexPoly(predicted, triangle, 1)
    return predicted > 0


def fit_camera(mesh, target, k, nominal):
    height, width = target.shape
    ys, _ = np.where(target)
    distance = k[1, 1] / (ys.max() - ys.min())
    initial = np.array([nominal, 20, 0, np.log(distance), 0, 0], dtype=float)
    bounds = [
        (nominal - 18, nominal + 18),
        (8, 36),
        (-6, 6),
        (initial[3] - 0.35, initial[3] + 0.35),
        (-0.12, 0.12),
        (-0.12, 0.12),
    ]
    evaluations = 0
    for resolution in [128, 256]:
        scale = resolution / max(width, height)
        w, h = round(width * scale), round(height * scale)
        mask = cv2.resize(target.astype("uint8"), (w, h), interpolation=cv2.INTER_NEAREST) > 0
        outside = cv2.distanceTransform((~mask).astype("uint8"), cv2.DIST_L2, 3)
        small_k = np.diag([w / width, h / height, 1]) @ k

        def loss(parameters, small_k=small_k, w=w, h=h, mask=mask, outside=outside):
            rot, trans = camera_pose(parameters, k, width, height)
            p = small_k @ np.column_stack([rot, trans])
            pred = silhouette(mesh.vertices, mesh.faces, p, w, h)
            union = (pred | mask).sum()
            iou = (pred & mask).sum() / max(1, union)
            exterior = outside[pred].mean() / max(w, h) if pred.any() else 1
            return float(1 - iou + 0.4 * exterior)

        start_loss = loss(initial)
        result = minimize(
            loss,
            initial,
            method="Powell",
            bounds=bounds,
            options={"maxfev": 320, "maxiter": 8, "xtol": 0.0008, "ftol": 0.0002},
        )
        evaluations += result.nfev
        if result.fun < start_loss:
            initial = result.x
    rot, trans = camera_pose(initial, k, width, height)
    pred = silhouette(mesh.vertices, mesh.faces, k @ np.column_stack([rot, trans]), width, height)
    iou = (pred & target).sum() / (pred | target).sum()
    return initial, rot, trans, pred, float(iou), evaluations


def restore_uv(canonical, obj_path, output):
    lines = obj_path.read_text().splitlines()
    vertices = np.array(
        [[float(x) for x in line.split()[1:4]] for line in lines if line.startswith("v ")]
    )
    uv = np.array(
        [[float(x) for x in line.split()[1:3]] for line in lines if line.startswith("vt ")]
    )
    pairs = np.array(
        [
            [[int(i) - 1 for i in c.split("/")[:2]] for c in line.split()[1:]]
            for line in lines
            if line.startswith("f ")
        ]
    )
    if (
        vertices.shape != canonical.vertices.shape
        or np.abs(vertices - canonical.vertices).max() > 1e-6
    ):
        raise ValueError("texture tool changed vertex coordinates or order")
    actual, expected = np.sort(pairs[:, :, 0], axis=1), np.sort(canonical.faces, axis=1)
    if not np.array_equal(
        actual[np.lexsort(actual.T[::-1])], expected[np.lexsort(expected.T[::-1])]
    ):
        raise ValueError("texture tool changed surface triangles")
    used_materials = {line for line in lines if line.startswith("usemtl ")}
    if len(used_materials) != 1:
        raise ValueError("multiple atlases require explicit material remapping")
    material = trimesh.load(obj_path, force="mesh").visual.material
    atlas_size = list(material.image.size)
    material = material.to_pbr()

    material.metallicFactor = 0.0
    unique, inverse = np.unique(pairs.reshape(-1, 2), axis=0, return_inverse=True)
    textured = trimesh.Trimesh(
        canonical.vertices[unique[:, 0]],
        inverse.reshape(-1, 3),
        vertex_normals=canonical.vertex_normals[unique[:, 0]],
        process=False,
        visual=trimesh.visual.TextureVisuals(uv=uv[unique[:, 1]], material=material),
    )
    textured.export(output)
    return {
        "faces": len(textured.faces),
        "max_original_coordinate_error": 0,
        "triangle_index_sets_identical": True,
        "uv_vertex_splits": len(unique) - len(canonical.vertices),
        "atlas_size": atlas_size,
        "metallic_factor": 0.0,
    }


def extend_foreground(rgb, mask, radius=8):
    distance, labels = cv2.distanceTransformWithLabels(
        (~mask).astype("uint8"), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL
    )
    result = rgb.copy()
    band = (~mask) & (distance <= radius)
    result[band] = rgb[mask][labels[band] - 1]
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Bake fit photographs onto frozen MV geometry with independent bounded cameras.\n\nNo SfM orbit is reused: its phase jumps are unsuitable for photo projection.\nApproximate quarter-turn anchors initialize poses; silhouette fitting refines\ncamera placement only. This is not independent camera calibration or a benchmark."
    )
    parser.add_argument("object", nargs="?", default="custom")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--views", type=Path, required=True)
    parser.add_argument("--focal-px", type=float)
    parser.add_argument("--texture-size", type=int, default=4096)
    parser.add_argument("--tools", type=Path, default=TOOL_ROOT / "openmvs")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    obj = args.object
    selection = json.loads(args.views.read_text())
    indices = [v["view"] for v in selection["views"]]
    dataset = args.dataset.resolve()
    out = args.output
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    data = json.loads((dataset / "dataset.json").read_text())
    assert set(indices) <= set(data["fit"]), "held-out photographs must not enter texture fitting"
    assert not set(indices) & set(data["heldout"])
    source = args.source.resolve()
    mesh = trimesh.load(source, force="mesh")
    mesh.export(out / "mesh.ply")
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(mesh.vertices, mesh.faces))
    ms.meshing_decimation_quadric_edge_collapse(targetfacenum=2500, preservenormal=True)
    proxy = trimesh.Trimesh(
        ms.current_mesh().vertex_matrix(), ms.current_mesh().face_matrix(), process=False
    )
    if (dataset / "train/sparse/0/cameras.bin").exists():
        cameras, _, _ = read_model(dataset / "train/sparse/0")
        c = next(iter(cameras.values()))
    else:
        width, height = data["output_size"]
        focal = args.focal_px or max(width, height) * 1.65
        c = {
            "width": width,
            "height": height,
            "fx": focal,
            "fy": focal,
            "cx": width / 2,
            "cy": height / 2,
        }

    k = np.array([[c["fx"], 0, c["cx"]], [0, c["fy"], c["cy"]], [0, 0, 1]])
    folder = out / "colmap/images"
    folder.mkdir(parents=True)
    sparse = out / "colmap/sparse"
    sparse.mkdir()
    x0, y0, x1, y1 = data["crop_xyxy"]
    w, h = x1 - x0, y1 - y0
    native_k = np.diag([w / c["width"], h / c["height"], 1]) @ k
    (sparse / "cameras.txt").write_text(
        f"1 PINHOLE {w} {h} {native_k[0, 0]} {native_k[1, 1]} {native_k[0, 2]} {native_k[1, 2]}\n"
    )
    (sparse / "points3D.txt").write_text(
        "# No triangulation; frozen generated mesh is supplied to TextureMesh.\n"
    )
    if file_hash(Path(data["source_video"])) != data["source_video_sha256"]:
        raise ValueError("original video hash changed")
    cap = cv2.VideoCapture(data["source_video"])
    records, lines, tiles, colors = [], [], [], []
    for image_id, index in enumerate(indices, 1):
        name = f"view_{index:04d}.png"
        rgb = cv2.imread(str(dataset / "reference/images" / name))
        alpha = cv2.imread(str(dataset / "reference/masks" / name), 0)
        target = alpha > 128
        pose_started = time.perf_counter()
        params, rot, trans, pred, iou, evaluations = fit_camera(
            proxy,
            target,
            k,
            next(v["nominal_angle"] for v in selection["views"] if v["view"] == index),
        )
        q = Rotation.from_matrix(rot).as_quat()
        lines.append(
            f"{image_id} {q[3]} {q[0]} {q[1]} {q[2]} {trans[0]} {trans[1]} {trans[2]} 1 {name}\n\n"
        )
        cap.set(cv2.CAP_PROP_POS_FRAMES, data["source_frame_indices"][index])
        ok, frame = cap.read()
        if not ok:
            raise ValueError("original frame decode failed")
        native = frame[y0:y1, x0:x1]
        if native.shape[:2] != (h, w):
            raise ValueError("crop no longer matches recorded dataset")
        native_mask = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_LINEAR) >= 128
        native = extend_foreground(native, native_mask)
        cv2.imwrite(str(folder / name), native)
        cv2.imwrite(str(folder / f"view_{index:04d}.mask.png"), native_mask.astype("uint8") * 255)
        colors.append(np.median(native[native_mask], axis=0))
        preview = rgb.copy()
        contours, _ = cv2.findContours(
            pred.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(preview, contours, -1, (60, 230, 70), 1)
        preview = cv2.resize(preview, (240, round(240 * h / w)))
        canvas = np.full((360, 240, 3), 25, dtype="uint8")
        canvas[: min(320, preview.shape[0])] = preview[:320]
        cv2.putText(
            canvas,
            f"view{index}: mask IoU {iou:.3f}",
            (5, 345),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (240, 240, 240),
            1,
        )
        tiles.append(canvas)
        records.append(
            {
                "index": index,
                "name": name,
                "nominal_yaw": next(
                    v["nominal_angle"] for v in selection["views"] if v["view"] == index
                ),
                "camera_parameters": params.tolist(),
                "rotation": rot.tolist(),
                "translation": trans.tolist(),
                "silhouette_iou": iou,
                "evaluations": evaluations,
                "seconds": time.perf_counter() - pose_started,
                "image_sha256": file_hash(folder / name),
                "source_alpha_sha256": file_hash(dataset / "reference/masks" / name),
            }
        )
        write_json(out / "cameras.json", records)
        print(obj, index, "IoU", round(iou, 3), "yaw", round(params[0], 1), flush=True)
    cap.release()
    pose_seconds = time.perf_counter() - started
    (sparse / "images.txt").write_text("".join(lines))
    cv2.imwrite(
        str(out / "camera-overlays.jpg"),
        np.vstack([np.hstack(tiles[i : i + 4]) for i in range(0, len(tiles), 4)]),
    )
    stages = {}
    for name, binary, flags, expected in [
        (
            "import",
            "InterfaceCOLMAP",
            [
                "-i",
                str(out / "colmap"),
                "-o",
                str(out / "scene.mvs"),
                "--image-folder",
                str(folder),
            ],
            ["scene.mvs"],
        ),
        (
            "texture",
            "TextureMesh",
            [
                "-i",
                str(out / "scene.mvs"),
                "-m",
                str(out / "mesh.ply"),
                "-o",
                str(out / "textured.obj"),
                "--export-type",
                "obj",
                "--decimate",
                "1",
                "--close-holes",
                "0",
                "--resolution-level",
                "0",
                "--max-texture-size",
                str(args.texture_size),
                "--virtual-face-images",
                "3",
                "--cost-smoothness-ratio",
                "0.01",
                "--global-seam-leveling",
                "0",
                "--local-seam-leveling",
                "0",
                "--sharpness-weight",
                "0",
                "--ignore-mask-label",
                "0",
                "--empty-color",
                str(
                    sum(
                        int(v) << shift
                        for v, shift in zip(
                            np.median(colors, axis=0)[::-1], [16, 8, 0], strict=True
                        )
                    )
                ),
            ],
            ["textured.obj", "textured.mtl"],
        ),
    ]:
        stages[name] = run_command(
            out / name,
            [str(args.tools / binary), "-w", str(out), "--max-threads", "4", *flags],
            [str(out / p) for p in expected],
            timeout=300,
        )
        if stages[name]["status"] != "success":
            raise RuntimeError(f"{name} failed; see {out / name / 'process.log'}")
    geometry = restore_uv(mesh, out / "textured.obj", out / "photo.glb")
    write_json(
        out / "result.json",
        {
            "object": obj,
            "source": str(source),
            "source_sha256": file_hash(source),
            "source_video_sha256": data["source_video_sha256"],
            "texture_views": indices,
            "heldout_used": [],
            "pose_and_preparation_seconds": pose_seconds,
            "stages": stages,
            "total_texture_pipeline_seconds": time.perf_counter() - started,
            "geometry": geometry,
            "photo_sha256": file_hash(out / "photo.glb"),
            "assessment": "Fixed MV geometry, native RGB pixels, BiRefNet masks, approximate camera fit to silhouettes. Baked illumination remains; no claim of calibrated poses or ground-truth internal detail. Unseen underside uses object median colour.",
        },
    )
    print(out / "photo.glb", flush=True)


if __name__ == "__main__":
    main()

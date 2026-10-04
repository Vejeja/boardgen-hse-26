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
from boardgen.evaluation.benchmark3d import file_hash, write_json
from boardgen.methods.mv.texture_mv import silhouette


def main():
    parser = argparse.ArgumentParser(
        description="Register the unchanged MV mesh to a physically constrained photo orbit.\n\nOnly global rigid orientation, translation and uniform scale enter registration;\ngeometry stays frozen. All photos used here belong to the original fit split."
    )
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--yaw-range", type=float, default=18)
    parser.add_argument("--yaw-search", type=int, default=0)
    args = parser.parse_args()
    if not 0 < args.yaw_range <= 180 or not 0 <= args.yaw_search <= 36:
        parser.error("yaw-range must be in (0,180]; yaw-search between 0 and 36")
    old = args.old.resolve()
    out = args.output.resolve()
    out.mkdir(exist_ok=False)
    started = time.perf_counter()
    calroot = args.dataset.resolve()
    cams, views, points = read_model(calroot / "train/sparse/0")
    cam = next(iter(cams.values()))
    w, h = cam["width"], cam["height"]
    k = np.array([[cam["fx"], 0, cam["cx"]], [0, cam["fy"], cam["cy"]], [0, 0, 1]])
    oldcams = json.loads((old / "cameras.json").read_text())
    indices = [c["index"] for c in oldcams]
    by_index = {int(v["name"][5:9]): v for v in views}
    selected = [by_index[i] for i in indices]
    result = json.loads((old / "result.json").read_text())
    mesh = trimesh.load(result["source"], force="mesh")
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(mesh.vertices, mesh.faces))
    ms.meshing_decimation_quadric_edge_collapse(targetfacenum=1800, preservenormal=True)
    proxy = ms.current_mesh()
    verts = proxy.vertex_matrix()
    faces = proxy.face_matrix()
    p = np.array(list(points.values()))
    lo, hi = np.percentile(p, [1, 99], axis=0)
    scale = (hi - lo)[1] * 1.1
    offset = (lo + hi) / 2 - scale * np.array([0, 0.5, 0])
    initial = np.r_[np.log(scale), 0.0, 0.0, 0.0, offset]
    bounds = [
        (np.log(scale * 0.75), np.log(scale * 1.4)),
        (-8, 8),
        (-args.yaw_range, args.yaw_range),
        (-5, 5),
        *[(v - 0.025, v + 0.025) for v in offset],
    ]
    targets = [cv2.imread(str(calroot / "reference/masks" / v["name"]), 0) > 128 for v in selected]

    def transform(x):
        return np.exp(x[0]), Rotation.from_euler("xyz", x[1:4], degrees=True).as_matrix(), x[4:7]

    evaluations = []
    for resolution in [128, 256]:
        sw, sh = round(w * resolution / h), resolution
        small_k = np.diag([sw / w, sh / h, 1]) @ k
        masks = [
            cv2.resize(m.astype("uint8"), (sw, sh), interpolation=cv2.INTER_NEAREST) > 0
            for m in targets
        ]

        def loss(x, masks=masks, small_k=small_k, sw=sw, sh=sh):
            s, rot, offset = transform(x)
            v = verts @ (s * rot).T + offset
            losses = []
            for image, mask in zip(selected, masks, strict=True):
                pred = silhouette(
                    v, faces, small_k @ np.column_stack([image["r"], image["t"]]), sw, sh
                )
                losses.append(1 - (pred & mask).sum() / max(1, (pred | mask).sum()))
            return np.mean(losses)

        if args.yaw_search and resolution == 128:
            trials = []
            for yaw in np.linspace(
                -args.yaw_range, args.yaw_range, args.yaw_search, endpoint=False
            ):
                trial = initial.copy()
                trial[2] = yaw
                trials.append((loss(trial), trial))
            initial = min(trials, key=lambda item: item[0])[1]
        opt = minimize(
            loss,
            initial,
            method="Powell",
            bounds=bounds,
            options={"maxfev": 450, "maxiter": 7, "xtol": 0.0001, "ftol": 0.0001},
        )
        if opt.fun < loss(initial):
            initial = opt.x
        evaluations.append(
            {"resolution": resolution, "loss": float(loss(initial)), "nfev": opt.nfev}
        )
        print("registration", evaluations[-1], flush=True)
    s, rot, offset = transform(initial)
    records = []
    tiles = []
    (out / "colmap/sparse").mkdir(parents=True)
    import shutil

    shutil.copytree(old / "colmap/images", out / "colmap/images")
    for path in (out / "colmap/images").glob("*.png"):
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image.shape[:2] != (h, w):
            interpolation = cv2.INTER_NEAREST if path.name.endswith(".mask.png") else cv2.INTER_AREA
            cv2.imwrite(str(path), cv2.resize(image, (w, h), interpolation=interpolation))
    shutil.copy2(old / "photo.glb", out / "photo.glb")
    (out / "colmap/sparse/cameras.txt").write_text(
        f"1 PINHOLE {w} {h} {cam['fx']} {cam['fy']} {cam['cx']} {cam['cy']}\n"
    )
    for image, mask, oldcam in zip(selected, targets, oldcams, strict=True):
        r = image["r"] @ rot
        t = (image["r"] @ offset + image["t"]) / s
        pred = silhouette(verts, faces, k @ np.column_stack([r, t]), w, h)
        iou = (pred & mask).sum() / (pred | mask).sum()
        record = oldcam.copy()
        record.update(
            rotation=r.tolist(),
            translation=t.tolist(),
            silhouette_iou=float(iou),
            camera_method="One global similarity registration to constrained fit-only orbit; no free silhouette yaw per photo",
        )
        records.append(record)
        rgb = cv2.imread(str(old / "colmap/images" / image["name"]))
        contours, _ = cv2.findContours(
            pred.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(rgb, contours, -1, (40, 220, 40), 1)
        rgb = cv2.resize(rgb, (198, 260))
        cv2.putText(
            rgb,
            f"{image['name']}: {iou:.3f}",
            (2, 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            (0, 220, 0),
            1,
        )
        tiles.append(rgb)
    cv2.imwrite(
        str(out / "camera-overlays.jpg"),
        np.vstack([np.hstack(tiles[i : i + 4]) for i in range(0, len(tiles), 4)]),
    )
    seconds = time.perf_counter() - started
    write_json(out / "cameras.json", records)
    result.update(
        total_texture_pipeline_seconds=seconds,
        pose_and_preparation_seconds=seconds,
        assessment="Frozen MV geometry, globally registered to fit-only physical orbit. Approximate generated shape can still disagree internally with photos; baked shadows remain.",
        uv_reused_from=str(old / "photo.glb"),
    )
    write_json(out / "result.json", result)
    write_json(
        out / "registration.json",
        {
            "seconds": seconds,
            "parameters": initial.tolist(),
            "evaluations": evaluations,
            "orbit_calibration_sha256": file_hash(calroot / "calibration.json"),
            "mean_iou": float(np.mean([r["silhouette_iou"] for r in records])),
            "geometry_preserved": True,
        },
    )
    print(out, seconds, flush=True)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from boardgen.evaluation.benchmark3d import file_hash, write_json


def read_model(sparse: Path):
    cameras, images, points = {}, [], {}
    with (sparse / "cameras.bin").open("rb") as f:
        for _ in range(struct.unpack("<Q", f.read(8))[0]):
            cid, kind, w, h = struct.unpack("<iiQQ", f.read(24))
            if kind == 0:
                focal, cx, cy = struct.unpack("<ddd", f.read(24))
                fx = fy = focal
            elif kind == 1:
                fx, fy, cx, cy = struct.unpack("<dddd", f.read(32))
            else:
                raise ValueError("audit supports undistorted pinhole cameras only")
            cameras[cid] = {
                "kind": kind,
                "width": w,
                "height": h,
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy,
            }
    with (sparse / "images.bin").open("rb") as f:
        for _ in range(struct.unpack("<Q", f.read(8))[0]):
            iid, qw, qx, qy, qz, tx, ty, tz, cid = struct.unpack("<idddddddi", f.read(64))
            name = bytearray()
            while (byte := f.read(1)) != b"\0":
                if not byte:
                    raise ValueError("truncated COLMAP image name")
                name.extend(byte)
            count = struct.unpack("<Q", f.read(8))[0]
            observations = np.frombuffer(
                f.read(count * 24), dtype=[("xy", "<f8", (2,)), ("pid", "<i8")]
            )
            if len(observations) != count:
                raise ValueError("truncated COLMAP observations")
            images.append(
                {
                    "id": iid,
                    "name": name.decode(),
                    "cid": cid,
                    "r": Rotation.from_quat([qx, qy, qz, qw]).as_matrix(),
                    "t": np.array([tx, ty, tz]),
                    "observations": observations,
                }
            )
    with (sparse / "points3D.bin").open("rb") as f:
        for _ in range(struct.unpack("<Q", f.read(8))[0]):
            row = struct.unpack("<QdddBBBdQ", f.read(51))
            points[row[0]] = np.array(row[1:4])
            f.seek(row[-1] * 8, 1)
    return cameras, sorted(images, key=lambda i: i["name"]), points


def orbit_diagnostics(centers: np.ndarray) -> dict:
    if len(centers) < 4:
        return {"status": "insufficient cameras"}
    origin = centers.mean(0)
    _, _, vt = np.linalg.svd(centers - origin, full_matrices=False)
    xy = (centers - origin) @ vt[:2].T
    coefs = np.linalg.lstsq(
        np.column_stack([2 * xy, np.ones(len(xy))]), np.sum(xy**2, axis=1), rcond=None
    )[0]
    radius = np.linalg.norm(xy - coefs[:2], axis=1)
    phase = np.unwrap(np.arctan2(xy[:, 1] - coefs[1], xy[:, 0] - coefs[0]))
    return {
        "radius_cv": float(radius.std() / radius.mean()),
        "plane_rms_over_radius": float(
            np.sqrt(np.mean(((centers - origin) @ vt[2]) ** 2)) / radius.mean()
        ),
        "estimated_sweep_deg": float(np.ptp(phase) * 180 / np.pi),
        "phase_deg": ((phase - phase[0]) * 180 / np.pi).tolist(),
        "assessment": "SfM-derived motion consistency, not independent pose ground truth.",
    }


def audit(sparse: Path, out: Path, expected_names: set[str] | None = None) -> dict:
    cameras, images, points = read_model(sparse)
    per_view, errors, centers = [], [], []
    for image in images:
        camera = cameras[image["cid"]]
        obs = image["observations"]
        centers.append(-image["r"].T @ image["t"])
        valid = np.array([pid in points for pid in obs["pid"]], dtype=bool)
        xyz = np.array([points[pid] for pid in obs["pid"][valid]])
        if not len(xyz):
            continue
        cam = xyz @ image["r"].T + image["t"]
        uv = cam[:, :2] / cam[:, 2:3]
        uv *= [camera["fx"], camera["fy"]]
        uv += [camera["cx"], camera["cy"]]
        error = np.linalg.norm(uv - obs["xy"][valid], axis=1)
        errors.extend(error.tolist())
        per_view.append(
            {
                "name": image["name"],
                "observations": len(error),
                "reprojection_mean_px": float(error.mean()),
                "reprojection_p95_px": float(np.percentile(error, 95)),
            }
        )
    names = {i["name"] for i in images}
    expected = names if expected_names is None else expected_names
    warnings = []
    for camera in cameras.values():
        camera["fx_over_fy"] = camera["fx"] / camera["fy"]
        if abs(camera["fx_over_fy"] - 1) > 0.02:
            warnings.append(
                "Anisotropic focal estimate: inspect shape under square-pixel, uniform-resize assumption."
            )
    if expected_names is not None and names != expected_names:
        warnings.append("The model does not register exactly the expected fit views.")
    orbit = orbit_diagnostics(np.array(centers))
    if orbit.get("estimated_sweep_deg", 0) < 320:
        warnings.append(
            "Estimated motion covers less than 320 degrees; frame count alone does not establish a full orbit."
        )
    row = {
        "registered_views": len(images),
        "points": len(points),
        "missing_fit_views": sorted(expected - names),
        "unexpected_views": sorted(names - expected),
        "cameras": cameras,
        "reprojection_mean_px": float(np.mean(errors)) if errors else None,
        "reprojection_p95_px": float(np.percentile(errors, 95)) if errors else None,
        "per_view": per_view,
        "orbit": orbit,
        "warnings": warnings,
        "hashes": {n: file_hash(sparse / n) for n in ("cameras.bin", "images.bin", "points3D.bin")},
        "assessment": "Reprojection checks training correspondences. Low residual does not certify correct proportions.",
    }
    write_json(out, row)
    return row


def main():
    parser = argparse.ArgumentParser(
        description="Audit pinhole COLMAP intrinsics, tracked reprojections and turntable coverage."
    )
    parser.add_argument("root", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    data = json.loads((args.root / "dataset.json").read_text())
    row = audit(args.root / "train/sparse/0", args.out, {f"view_{i:04d}.png" for i in data["fit"]})
    print(
        json.dumps(
            {k: row[k] for k in ("registered_views", "cameras", "reprojection_mean_px", "warnings")}
        )
    )


if __name__ == "__main__":
    main()

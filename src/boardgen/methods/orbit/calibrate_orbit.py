import argparse
import json
import shutil
import struct
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

from boardgen.camera.camera_audit import read_model
from boardgen.evaluation.benchmark3d import file_hash, write_json


def rotations(yaw, elevation, roll):
    az = yaw
    direction = np.column_stack(
        [
            np.sin(az) * np.cos(elevation),
            np.full(len(az), np.sin(elevation)),
            np.cos(az) * np.cos(elevation),
        ]
    )
    forward = -direction
    right = np.cross(forward, [0, 1, 0])
    right /= np.linalg.norm(right, axis=1)[:, None]
    result = np.stack([right, np.cross(forward, right), forward], axis=1)
    return np.einsum("ab,nbc->nac", Rotation.from_euler("z", roll).as_matrix(), result)


def repeat_period(root, fit, candidates):
    images, masks = {}, {}
    for i in fit:
        name = f"view_{i:04d}.png"
        images[i] = cv2.resize(
            cv2.imread(str(root / "reference/images" / name)), (120, 160)
        ).astype(float)
        masks[i] = cv2.resize(cv2.imread(str(root / "reference/masks" / name), 0), (120, 160)) > 128
    rows = []
    for offset in candidates:
        values = []
        for i in fit:
            if i + offset not in images:
                continue
            a, b = masks[i], masks[i + offset]
            overlap = a & b
            values.append(
                1
                - overlap.sum() / max(1, (a | b).sum())
                + abs(images[i] - images[i + offset])[overlap].mean() / 255
            )
        rows.append({"offset": offset, "pairs": len(values), "loss": float(np.median(values))})
    return min(rows, key=lambda r: r["loss"])["offset"], rows


def triangulate(rs, ts, k, image_ids, xy):
    ps = k @ np.concatenate([rs[image_ids], ts[image_ids, :, None]], axis=2)
    a = np.stack(
        [xy[:, 0, None] * ps[:, 2] - ps[:, 0], xy[:, 1, None] * ps[:, 2] - ps[:, 1]], axis=1
    ).reshape(-1, 4)
    _, _, vt = np.linalg.svd(a)
    p = vt[-1, :3] / vt[-1, 3]
    cp = np.einsum("nab,b->na", rs[image_ids], p) + ts[image_ids]
    pred = cp[:, :2] / cp[:, 2:] * [k[0, 0], k[1, 1]] + [k[0, 2], k[1, 2]]
    return p, np.linalg.norm(pred - xy, axis=1), cp[:, 2]


def write_model(folder, camera, images, points, observations, colors):
    folder.mkdir(parents=True)
    with (folder / "cameras.bin").open("wb") as f:
        f.write(
            struct.pack(
                "<QiiQQddd",
                1,
                1,
                0,
                camera["width"],
                camera["height"],
                camera["fx"],
                camera["cx"],
                camera["cy"],
            )
        )
    obs_by_image = [[] for _ in images]
    tracks = [[] for _ in points]
    for im, pid, xy in observations:
        tracks[pid].append((images[im]["id"], len(obs_by_image[im])))
        obs_by_image[im].append((*xy, pid + 1))
    with (folder / "images.bin").open("wb") as f:
        f.write(struct.pack("<Q", len(images)))
        for i, im in enumerate(images):
            q = Rotation.from_matrix(im["r"]).as_quat()
            f.write(struct.pack("<idddddddi", im["id"], q[3], *q[:3], *im["t"], 1))
            f.write(im["name"].encode() + b"\0")
            f.write(struct.pack("<Q", len(obs_by_image[i])))
            for x, y, pid in obs_by_image[i]:
                f.write(struct.pack("<ddq", x, y, pid))
    with (folder / "points3D.bin").open("wb") as f:
        f.write(struct.pack("<Q", len(points)))
        for i, p in enumerate(points):
            f.write(struct.pack("<QdddBBBdQ", i + 1, *p, *map(int, colors[i]), 0.0, len(tracks[i])))
            for track in tracks[i]:
                f.write(struct.pack("<ii", *track))


def main():
    ap = argparse.ArgumentParser(
        description="Constrained turntable bundle adjustment using fit-only SIFT observations.\n\nA physical full-turn period is estimated from repeat-view image agreement, not\nfrom a generated mesh. Short SfM tracks are re-triangulated in this orbit; its\noriginal (potentially folded) poses and point coordinates are discarded."
    )
    ap.add_argument("object", help="object identifier for provenance")
    ap.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="fresh fit-only SfM dataset",
    )
    ap.add_argument(
        "--front-view",
        type=int,
        required=True,
        help="approximate front anchor in sampled view indices",
    )
    ap.add_argument(
        "--period-range",
        required=True,
        type=int,
        nargs=2,
        metavar=("MIN", "MAX"),
        help="inclusive period candidate range in sampled view indices",
    )
    ap.add_argument(
        "--focal-px", type=float, required=True, help="initial focal length in native crop pixels"
    )
    ap.add_argument(
        "--yaw-slack-deg",
        type=float,
        default=float(np.degrees(0.06)),
        help="maximum per-view angular deviation from the repeated-view orbit",
    )
    ap.add_argument(
        "--shift-bound-px",
        type=float,
        default=20,
        help="maximum per-view image shift for a moving or off-center rotation axis",
    )
    ap.add_argument("--max-nfev", type=int, default=70)
    ap.add_argument("--rotation-direction", type=int, choices=[-1, 1], default=1)
    ap.add_argument(
        "--principal-point",
        type=float,
        nargs=2,
        help="optical center in cropped/resized pixels; default is the crop center",
    )
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    front = args.front_view
    focal = args.focal_px
    candidates = range(args.period_range[0], args.period_range[1] + 1)
    root = args.dataset.resolve()
    if args.front_view is not None:
        front = args.front_view
    if args.period_range is not None:
        low, high = args.period_range
        if low <= 0 or high < low:
            ap.error("period range must be positive and increasing")
        candidates = range(low, high + 1)
    if args.focal_px is not None:
        if args.focal_px <= 0:
            ap.error("focal length must be positive")
        focal = args.focal_px
    if not 0 < args.yaw_slack_deg <= 45 or args.shift_bound_px <= 0 or args.max_nfev < 1:
        ap.error("positive camera bounds required; yaw slack must not exceed 45 degrees")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    data = json.loads((root / "dataset.json").read_text())
    fit = data["fit"]
    period, period_rows = repeat_period(root, fit, candidates)
    cams, images, _ = read_model(root / "train/sparse/0")
    cam = next(iter(cams.values()))
    w, h = cam["width"], cam["height"]
    cx, cy = args.principal_point if args.principal_point else (w / 2, h / 2)
    if not np.isfinite([cx, cy]).all():
        ap.error("principal point must be finite")
    indices = np.array([int(im["name"][5:9]) for im in images])
    n = len(images)
    nominal = args.rotation_direction * (indices - front) * 2 * np.pi / period
    rs = rotations(nominal, np.radians(24), np.radians(-1))
    ts = np.tile([0.0, 0.0, 1.0], (n, 1))
    k = np.array([[focal, 0, cx], [0, focal, cy], [0, 0, 1.0]])
    masks = {im["name"]: cv2.imread(str(root / "reference/masks" / im["name"]), 0) for im in images}
    tracks = {}
    for i, im in enumerate(images):
        for obs in im["observations"]:
            xy = obs["xy"]
            pid = int(obs["pid"])
            x, y = np.rint(xy).astype(int)
            if pid >= 0 and 0 <= x < w and 0 <= y < h and masks[im["name"]][y, x] > 220:
                tracks.setdefault(pid, []).append((i, xy))
    points = []
    observations = []
    colors = []
    rgb = {}
    for track in tracks.values():
        segments = []
        segment = []
        for obs in track:
            if segment and indices[obs[0]] - indices[segment[-1][0]] > 3:
                segments.append(segment)
                segment = []
            segment.append(obs)
        segments.append(segment)
        for segment in segments:
            if len(segment) < 3:
                continue
            segment = segment[:16]
            ids = np.array([a[0] for a in segment])
            xy = np.array([a[1] for a in segment])
            if indices[ids[-1]] - indices[ids[0]] < 3:
                continue
            p, err, depth = triangulate(rs, ts, k, ids, xy)
            if (
                np.median(err) > 3.5
                or err.max() > 9
                or depth.min() < 0.3
                or np.linalg.norm(p) > 0.5
            ):
                continue
            pi = len(points)
            points.append(p)
            observations.extend((int(i), pi, uv) for i, uv in zip(ids, xy, strict=True))
            im = images[ids[len(ids) // 2]]
            filename = im["name"]
            if filename not in rgb:
                rgb[filename] = cv2.imread(str(root / "reference/images" / filename))[:, :, ::-1]
            x, y = np.rint(xy[len(xy) // 2]).astype(int)
            colors.append(rgb[filename][y, x])
    points = np.array(points)
    colors = np.array(colors)
    oi = np.array([o[0] for o in observations])
    pi = np.array([o[1] for o in observations])
    target = np.array([o[2] for o in observations])
    m = len(oi)
    if len(points) < 100:
        raise ValueError(f"Only {len(points)} reliable short tracks")
    print(args.object, "period", period, "points", len(points), "observations", m, flush=True)

    base = 3 + 3 * n
    x0 = np.r_[
        np.log(focal),
        np.radians(24),
        np.radians(-1),
        np.column_stack([nominal, np.zeros((n, 2))]).ravel(),
        points.ravel(),
    ]
    gaps = np.diff(indices)
    angular_speed = args.rotation_direction * 2 * np.pi / period

    def unpack(x):
        c = x[3:base].reshape(n, 3)
        r = rotations(c[:, 0], x[1], x[2])
        t = np.column_stack([c[:, 1] / np.exp(x[0]), c[:, 2] / np.exp(x[0]), np.ones(n)])
        return c, r, t, x[base:].reshape(-1, 3)

    def residual(x):
        c, r, t, p = unpack(x)
        cp = np.einsum("nab,nb->na", r[oi], p[pi]) + t[oi]
        pred = cp[:, :2] / np.maximum(cp[:, 2:], 1e-6) * np.exp(x[0]) + [cx, cy]
        return np.r_[
            (pred - target).ravel(),
            (np.diff(c[:, 0]) / gaps - angular_speed) * 80,
            (c[:, 0] - nominal) * 0.3,
            c[:, 1:].ravel() * 0.08,
            (x[0] - np.log(focal)) * 0.5,
            (x[1] - np.radians(24)) * 0.5,
            x[2] * 0.2,
        ]

    sparsity = lil_matrix((2 * m + (n - 1) + n + 2 * n + 3, len(x0)), dtype=int)
    for j, (iv, pv) in enumerate(zip(oi, pi, strict=True)):
        sparsity[2 * j : 2 * j + 2, :3] = 1
        sparsity[2 * j : 2 * j + 2, 3 + 3 * iv : 6 + 3 * iv] = 1
        sparsity[2 * j : 2 * j + 2, base + 3 * pv : base + 3 * pv + 3] = 1
    row = 2 * m
    for i in range(n - 1):
        sparsity[row + i, 3 + 3 * i] = 1
        sparsity[row + i, 6 + 3 * i] = 1
    row += n - 1
    for i in range(n):
        sparsity[row + i, 3 + 3 * i] = 1
    row += n
    for i in range(n):
        sparsity[row + 2 * i : row + 2 * i + 2, 4 + 3 * i : 6 + 3 * i] = 1
    row += 2 * n
    sparsity[row : row + 3, :3] = 1
    lower = np.full(len(x0), -np.inf)
    upper = np.full(len(x0), np.inf)
    lower[:3] = [np.log(focal * 0.6), np.radians(10), np.radians(-8)]
    upper[:3] = [np.log(focal * 1.6), np.radians(40), np.radians(8)]
    for i in range(n):
        lower[3 + 3 * i : 6 + 3 * i] = [
            nominal[i] - np.radians(args.yaw_slack_deg),
            -args.shift_bound_px,
            -args.shift_bound_px,
        ]
        upper[3 + 3 * i : 6 + 3 * i] = [
            nominal[i] + np.radians(args.yaw_slack_deg),
            args.shift_bound_px,
            args.shift_bound_px,
        ]
    result = least_squares(
        residual,
        x0,
        jac_sparsity=sparsity.tocsr(),
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=0.8,
        x_scale="jac",
        max_nfev=args.max_nfev,
        ftol=1e-5,
        verbose=0,
    )
    c, r, t, p = unpack(result.x)
    err = np.linalg.norm(residual(result.x)[: 2 * m].reshape(-1, 2), axis=1)
    valid = err < 3.0
    support = np.bincount(pi[valid], minlength=len(p))
    valid &= support[pi] >= 3
    kept = np.unique(pi[valid])
    mapping = {int(old): j for j, old in enumerate(kept)}
    observations = [(int(oi[j]), mapping[int(pi[j])], target[j]) for j in np.flatnonzero(valid)]
    for i, im in enumerate(images):
        im["r"], im["t"] = r[i], t[i]
    cam = {"width": w, "height": h, "fx": float(np.exp(result.x[0])), "cx": cx, "cy": cy}
    write_model(out / "train/sparse/0", cam, images, p[kept], observations, colors[kept])
    for section in ["train/images", "train/masks", "reference"]:
        (out / section).parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(root / section, out / section)
    data.update(
        derived_from=str(root),
        camera_method="Fit-only constrained orbit BA from short SIFT tracks; RGB repeat period; no generated geometry",
        camera_source_hash=file_hash(root / "train/sparse/0/images.bin"),
        camera_initialization_provenance=data.get("camera_provenance"),
        cameras="SIMPLE_PINHOLE, calibrated fit-only constrained orbit; arbitrary nonmetric scale",
        camera_provenance={
            "run": "fit-only constrained orbit bundle adjustment",
            "hashes": {
                name: file_hash(out / "train/sparse/0" / name)
                for name in ("cameras.bin", "images.bin", "points3D.bin")
            },
        },
    )
    write_json(out / "dataset.json", data)
    write_json(
        out / "calibration.json",
        {
            "seconds": time.perf_counter() - started,
            "object": args.object,
            "period_frames": period,
            "rotation_direction": args.rotation_direction,
            "repeat_period_screen": period_rows,
            "camera": cam,
            "optimization_bounds": {
                "yaw_slack_deg": args.yaw_slack_deg,
                "shift_bound_px": args.shift_bound_px,
                "max_nfev": args.max_nfev,
                "maximum_yaw_correction_deg": float(np.max(np.abs(np.degrees(c[:, 0] - nominal)))),
                "maximum_image_shift_px": float(np.max(np.abs(c[:, 1:]))),
                "converged": bool(result.success),
                "nfev": int(result.nfev),
            },
            "elevation_deg": float(np.degrees(result.x[1])),
            "roll_deg": float(np.degrees(result.x[2])),
            "registered_views": n,
            "initial_points": len(points),
            "retained_points": len(kept),
            "observations": len(observations),
            "median_reprojection_px": float(np.median(err[valid])),
            "p95_reprojection_px": float(np.percentile(err[valid], 95)),
            "rejected_observations": int((~valid).sum()),
            "max_neighbor_angle_deg": float(np.max(np.degrees(np.diff(c[:, 0]) / gaps))),
            "min_neighbor_angle_deg": float(np.min(np.degrees(np.diff(c[:, 0]) / gaps))),
            "camera_parameters": [
                {
                    "index": int(index),
                    "yaw_deg": float(np.degrees(a[0])),
                    "dx_px": float(a[1]),
                    "dy_px": float(a[2]),
                }
                for index, a in zip(indices, c, strict=True)
            ],
            "assessment": "Constrained camera estimate, not metrologic calibration; manual approximate front anchor fixes yaw gauge. No held-out data or MV mesh used.",
        },
    )
    print(
        out,
        "seconds",
        time.perf_counter() - started,
        "focal",
        cam["fx"],
        "elevation",
        np.degrees(result.x[1]),
        "reproj",
        np.median(err[valid]),
        flush=True,
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from boardgen.evaluation.benchmark3d import file_hash, write_json
from boardgen.methods.orbit.calibrate_orbit import write_model


def local_tracks(root: Path, focal: float) -> dict:
    if not np.isfinite(focal) or focal <= 0:
        raise ValueError("Focal initialization must be finite and positive")
    data = json.loads((root / "dataset.json").read_text())
    fit = sorted(data["fit"])
    target = root / "train/sparse/0"
    if target.exists():
        raise ValueError("Track initialization must use a fresh dataset")
    sift = cv2.SIFT_create(nfeatures=12000, contrastThreshold=0.005, edgeThreshold=20)
    features, images = [], []
    for index in fit:
        name = f"view_{index:04d}.png"
        rgb = cv2.imread(str(root / "reference/images" / name))
        alpha = cv2.imread(str(root / "reference/masks" / name), 0)
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        mask = cv2.erode((alpha > 220).astype(np.uint8) * 255, np.ones((3, 3), np.uint8))
        kp, desc = sift.detectAndCompute(gray, mask)
        features.append((kp, desc, rgb))
        images.append(
            {"id": len(images) + 1, "name": name, "r": np.eye(3), "t": np.array([0.0, 0.0, 1.0])}
        )
    chains = {}
    matcher = cv2.BFMatcher()
    counts = []
    for i in range(len(fit) - 1):
        ka, da, _ = features[i]
        kb, db, _ = features[i + 1]
        if da is None or db is None or len(da) < 8 or len(db) < 8:
            counts.append(0)
            continue
        forward = [m for m, n in matcher.knnMatch(da, db, k=2) if m.distance < 0.8 * n.distance]
        reverse = {
            (m.trainIdx, m.queryIdx)
            for m, n in matcher.knnMatch(db, da, k=2)
            if m.distance < 0.8 * n.distance
        }
        matches = [m for m in forward if (m.queryIdx, m.trainIdx) in reverse]
        if len(matches) >= 12:
            _, inliers = cv2.findFundamentalMat(
                np.float32([ka[m.queryIdx].pt for m in matches]),
                np.float32([kb[m.trainIdx].pt for m in matches]),
                cv2.FM_RANSAC,
                1.5,
                0.999,
            )
            if inliers is not None:
                matches = [m for m, keep in zip(matches, inliers.ravel(), strict=True) if keep]
        counts.append(len(matches))
        for match in matches:
            node = (i, match.queryIdx)
            chain = chains.get(node, [node])
            chain.append((i + 1, match.trainIdx))
            chains[node] = chain
            chains[(i + 1, match.trainIdx)] = chain
    unique = {id(chain): chain for chain in chains.values()}.values()
    tracks = [
        [(i, np.array(features[i][0][key].pt)) for i, key in chain[:16]]
        for chain in unique
        if len(chain) >= 3 and fit[chain[-1][0]] - fit[chain[0][0]] >= 3
    ]

    grays = [cv2.cvtColor(item[2], cv2.COLOR_BGR2GRAY) for item in features]
    masks = [cv2.imread(str(root / "reference/masks" / im["name"]), 0) for im in images]
    for start in range(0, len(fit) - 2, 3):
        seeds = cv2.goodFeaturesToTrack(
            grays[start],
            maxCorners=350,
            qualityLevel=0.005,
            minDistance=5,
            mask=(masks[start] > 220).astype(np.uint8) * 255,
        )
        if seeds is None:
            continue
        active = seeds.astype(np.float32)
        trails = [[(start, xy[0].copy())] for xy in active]
        for end in range(start + 1, min(len(fit), start + 8)):
            if not len(active):
                break
            nxt, forward, error = cv2.calcOpticalFlowPyrLK(
                grays[end - 1], grays[end], active, None, winSize=(21, 21), maxLevel=3
            )
            back, backward, _ = cv2.calcOpticalFlowPyrLK(
                grays[end], grays[end - 1], nxt, None, winSize=(21, 21), maxLevel=3
            )
            xy = np.rint(nxt[:, 0]).astype(int)
            h, w = masks[end].shape
            valid = (
                (forward.ravel() > 0)
                & (backward.ravel() > 0)
                & (np.linalg.norm(back[:, 0] - active[:, 0], axis=1) < 0.8)
                & (error.ravel() < 30)
                & (xy[:, 0] >= 0)
                & (xy[:, 0] < w)
                & (xy[:, 1] >= 0)
                & (xy[:, 1] < h)
            )
            ids = np.flatnonzero(valid)
            valid[ids] &= masks[end][xy[ids, 1], xy[ids, 0]] > 220
            for trail, point, keep in zip(trails, nxt[:, 0], valid, strict=True):
                if keep:
                    trail.append((end, point.copy()))
                elif len(trail) >= 3 and fit[trail[-1][0]] - fit[start] >= 3:
                    tracks.append(trail)
            trails = [trail for trail, keep in zip(trails, valid, strict=True) if keep]
            active = nxt[valid]
        tracks.extend(
            trail for trail in trails if len(trail) >= 3 and fit[trail[-1][0]] - fit[start] >= 3
        )
    if len(tracks) < 100:
        raise ValueError(f"Only {len(tracks)} local tracks; initialization not usable")
    observations, colors = [], []
    per_view = np.zeros(len(fit), int)
    for pid, chain in enumerate(tracks):
        for i, xy in chain:
            observations.append((i, pid, xy))
            per_view[i] += 1
        i, xy = chain[len(chain) // 2]
        x, y = np.rint(xy).astype(int)
        colors.append(features[i][2][y, x, ::-1])
    h, w = features[0][2].shape[:2]
    write_model(
        target,
        {"width": w, "height": h, "fx": focal, "cx": w / 2, "cy": h / 2},
        images,
        np.zeros((len(tracks), 3)),
        observations,
        colors,
    )
    provenance = {
        "method": "fit-only mutual local SIFT with adjacent-pair fundamental RANSAC plus forward/backward-checked local corner flow",
        "tracks": len(tracks),
        "pair_matches": counts,
        "per_view_observations": dict(zip(map(str, fit), map(int, per_view), strict=True)),
        "placeholder_poses": True,
        "assessment": "Initialization only. Points and poses require constrained orbit BA before training; no held-out pixels used.",
    }
    data["camera_provenance"] = {
        "run": "local-tracks",
        "hashes": {n: file_hash(target / n) for n in ["cameras.bin", "images.bin", "points3D.bin"]},
    }
    data["cameras"] = "local track initialization only; not training cameras"
    write_json(root / "dataset.json", data)
    write_json(root / "local-tracks.json", provenance)
    return provenance


def main():
    parser = argparse.ArgumentParser(
        description="Build fit-only local SIFT tracks for constrained turntable bundle adjustment.\n\nThe exported poses/points are initialization placeholders, never training poses.\nThis fallback avoids requiring global SfM registration on repetitive plastic.\nOnly a subsequent calibrated and audited orbit model may be used for training."
    )
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--focal", type=float, required=True)
    args = parser.parse_args()
    print(json.dumps(local_tracks(args.dataset, args.focal)))


if __name__ == "__main__":
    main()

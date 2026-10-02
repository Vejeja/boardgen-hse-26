import glob
import itertools
import os

import cv2
import numpy as np
from PIL import Image
from skimage import measure

from boardgen.camera.camera_model import TurntableCameraModel
from boardgen.common.logger import get_logger

logger = get_logger()


def load_masks(d, return_paths=False, alpha_threshold=128):
    paths = sorted(
        p for p in glob.glob(os.path.join(d, "*.png")) if not os.path.basename(p).startswith("_")
    )
    arr = []
    for p in paths:
        with Image.open(p) as im:
            a = np.array(im.split()[-1] if im.mode in ("RGBA", "LA") else im.convert("L"))
        threshold = 1 if a.max() <= 1 else alpha_threshold
        m = (a >= threshold).astype(np.uint8)
        arr.append(m)
    if not arr:
        raise RuntimeError("no masks")
    stack = np.stack(arr, 0)
    if return_paths:
        return stack, paths
    return stack


def carve_visual_hull(m, fov_deg=45.0, grid=128, consensus=0.85, y_offset=1.2, samples_per_axis=2):
    if grid < 2:
        raise ValueError("grid must contain at least two samples per axis")
    N, H, W = m.shape
    import math

    fx = (W / 2) / math.tan(math.radians(fov_deg / 2))
    fy = fx
    cx = (W - 1) / 2
    cy = (H - 1) / 2
    S = 1.0
    xs = np.linspace(-S / 2, S / 2, grid)
    ys = np.linspace(-S / 2, S / 2, grid)
    zs = np.linspace(0, S, grid)
    step = S / (grid - 1) if grid > 1 else S
    thetas = np.linspace(0, 2 * np.pi, N, endpoint=False)
    occ = np.zeros((grid, grid, grid), np.float32)
    max_ratio = 0.0
    samples = max(1, int(samples_per_axis))
    if samples <= 1:
        offsets = [(0.0, 0.0, 0.0)]
    else:
        lin = np.linspace(-0.35, 0.35, samples)
        offsets = [(dx, dy, dz) for dx, dy, dz in itertools.product(lin, repeat=3)]
    num_offsets = float(len(offsets))
    for zi, z in enumerate(zs):
        accum = np.zeros((grid, grid), np.float32)
        for dx, dy, dz in offsets:
            X, Y = np.meshgrid(xs + dx * step, ys + dy * step, indexing="xy")
            Z = np.full_like(X, z + dz * step)
            P = np.stack([X, Y, Z], -1).reshape(-1, 3)
            votes = np.zeros((P.shape[0],), np.int32)
            for v in range(N):
                th = thetas[v]
                c = np.cos(-th)
                s = np.sin(-th)
                R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], np.float32)
                pc = P @ R.T
                Xc = pc[:, 0]
                Yc = pc[:, 1] + y_offset
                Zc = pc[:, 2]
                valid = Yc > 1e-3
                u = np.rint(fx * (Xc / Yc) + cx).astype(np.int32)
                vv = np.rint(fy * (-Zc / Yc) + cy).astype(np.int32)
                inside = valid & (u >= 0) & (u < W) & (vv >= 0) & (vv < H)
                if not np.any(inside):
                    continue
                idx = np.where(inside)[0]
                if idx.size:
                    votes_idx = (m[v][vv[idx], u[idx]] > 0).astype(np.int32)
                    if votes_idx.size:
                        votes[idx] += votes_idx
            if votes.size:
                ratios = votes.reshape(grid, grid) / float(N)
                accum += ratios
        if num_offsets > 0:
            accum /= num_offsets
        occ[zi, :, :] = accum
        mr = float(accum.max())
        if mr > max_ratio:
            max_ratio = mr
    if grid >= 16:
        blur = np.empty_like(occ)
        for zi in range(grid):
            blur[zi] = cv2.GaussianBlur(occ[zi], (3, 3), 0)
        if grid > 2:
            occ_sm = blur.copy()
            occ_sm[1:-1] = (blur[:-2] + blur[1:-1] + blur[2:]) / 3.0
            occ_sm[0] = (blur[0] + blur[1]) * 0.5
            occ_sm[-1] = (blur[-1] + blur[-2]) * 0.5
        else:
            occ_sm = blur
        occ = occ_sm
    max_ratio = float(occ.max()) if occ.size else 0.0
    if max_ratio < consensus or not np.any(occ):
        raise RuntimeError(f"visual hull empty (max_ratio={max_ratio:.3f}, required={consensus})")
    verts, faces, _, _ = measure.marching_cubes(
        occ, level=consensus, spacing=(step, step, step), allow_degenerate=False
    )
    V = np.stack([verts[:, 2] - S / 2, verts[:, 1] - S / 2, verts[:, 0]], 1)
    return V, faces.astype(np.int32)


def carve_visual_hull_calibrated(
    m,
    projection_matrices,
    grid=128,
    consensus=0.85,
    bounds=((-0.5, 0.5), (-0.5, 0.5), (0.0, 1.0)),
    samples_per_axis=2,
):
    if grid < 2:
        raise ValueError("grid must contain at least two samples per axis")
    n_views, H, W = m.shape
    if len(projection_matrices) < n_views:
        raise ValueError(f"need at least {n_views} projection matrices")
    x_bounds, y_bounds, z_bounds = bounds
    xs = np.linspace(float(x_bounds[0]), float(x_bounds[1]), grid)
    ys = np.linspace(float(y_bounds[0]), float(y_bounds[1]), grid)
    zs = np.linspace(float(z_bounds[0]), float(z_bounds[1]), grid)
    step = max(
        (float(x_bounds[1]) - float(x_bounds[0])) / max(1, grid - 1),
        (float(y_bounds[1]) - float(y_bounds[0])) / max(1, grid - 1),
        (float(z_bounds[1]) - float(z_bounds[0])) / max(1, grid - 1),
    )
    occ = np.zeros((grid, grid, grid), np.float32)
    samples = max(1, int(samples_per_axis))
    if samples <= 1:
        offsets = [(0.0, 0.0, 0.0)]
    else:
        lin = np.linspace(-0.35, 0.35, samples)
        offsets = [(dx, dy, dz) for dx, dy, dz in itertools.product(lin, repeat=3)]
    p_mats = [np.asarray(p, dtype=np.float64) for p in projection_matrices[:n_views]]
    num_offsets = float(len(offsets))
    for zi, z in enumerate(zs):
        accum = np.zeros((grid, grid), np.float32)
        for dx, dy, dz in offsets:
            X, Y = np.meshgrid(xs + dx * step, ys + dy * step, indexing="xy")
            Z = np.full_like(X, z + dz * step)
            pts = np.stack([X, Y, Z, np.ones_like(X)], -1).reshape(-1, 4)
            votes = np.zeros((pts.shape[0],), np.int32)
            for view_idx, proj in enumerate(p_mats):
                hp = pts @ proj.T
                depth = hp[:, 2]
                valid = depth > 1e-8
                u = np.rint(hp[:, 0] / np.maximum(depth, 1e-12)).astype(np.int32)
                vv = np.rint(hp[:, 1] / np.maximum(depth, 1e-12)).astype(np.int32)
                inside = valid & (u >= 0) & (u < W) & (vv >= 0) & (vv < H)
                if not np.any(inside):
                    continue
                idx = np.where(inside)[0]
                votes[idx] += (m[view_idx][vv[idx], u[idx]] > 0).astype(np.int32)
            accum += votes.reshape(grid, grid) / float(n_views)
        occ[zi, :, :] = accum / num_offsets
    if grid >= 16:
        blur = np.empty_like(occ)
        for zi in range(grid):
            blur[zi] = cv2.GaussianBlur(occ[zi], (3, 3), 0)
        if grid > 2:
            occ_sm = blur.copy()
            occ_sm[1:-1] = (blur[:-2] + blur[1:-1] + blur[2:]) / 3.0
            occ_sm[0] = (blur[0] + blur[1]) * 0.5
            occ_sm[-1] = (blur[-1] + blur[-2]) * 0.5
        else:
            occ_sm = blur
        occ = occ_sm
    max_ratio = float(occ.max()) if occ.size else 0.0
    if max_ratio < consensus or not np.any(occ):
        raise RuntimeError(
            f"calibrated visual hull empty (max_ratio={max_ratio:.3f}, required={consensus})"
        )
    spacing = (
        (float(z_bounds[1]) - float(z_bounds[0])) / (grid - 1),
        (float(y_bounds[1]) - float(y_bounds[0])) / (grid - 1),
        (float(x_bounds[1]) - float(x_bounds[0])) / (grid - 1),
    )
    verts, faces, _, _ = measure.marching_cubes(
        occ, level=consensus, spacing=spacing, allow_degenerate=False
    )
    V = np.stack(
        [
            verts[:, 2] + float(x_bounds[0]),
            verts[:, 1] + float(y_bounds[0]),
            verts[:, 0] + float(z_bounds[0]),
        ],
        1,
    )
    return V, faces.astype(np.int32)


def reconstruct(masks, cameras=None, config=None):
    cfg = config or {}
    if cameras is None:
        return carve_visual_hull(
            masks,
            fov_deg=float(cfg.get("fov_deg", 45.0)),
            grid=int(cfg.get("grid_res", 128)),
            consensus=float(cfg.get("consensus", 0.85)),
            y_offset=float(cfg.get("y_offset", 1.2)),
            samples_per_axis=int(cfg.get("samples_per_axis", 2)),
        )
    if isinstance(cameras, TurntableCameraModel):
        p_mats = cameras.projection_matrices()
    else:
        p_mats = cameras
    return carve_visual_hull_calibrated(
        masks,
        p_mats,
        grid=int(cfg.get("grid_res", 128)),
        consensus=float(cfg.get("consensus", 0.85)),
        bounds=cfg.get("bounds", ((-0.5, 0.5), (-0.5, 0.5), (0.0, 1.0))),
        samples_per_axis=int(cfg.get("samples_per_axis", 2)),
    )

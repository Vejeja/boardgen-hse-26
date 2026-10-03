from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import trimesh

from boardgen.camera.camera_audit import read_model
from boardgen.evaluation.benchmark3d import file_hash, write_json


def render_geometry(mesh, projection, rotation, width, height):
    hp = np.column_stack([mesh.vertices, np.ones(len(mesh.vertices))]) @ projection.T
    xy = hp[:, :2] / np.maximum(hp[:, 2:3], 1e-12)
    normals = mesh.vertex_normals @ rotation.T
    depth = np.full((height, width), np.inf)
    result = np.zeros((height, width, 3))
    for face in mesh.faces:
        if np.any(hp[face, 2] <= 0):
            continue
        p = xy[face]
        lo = np.maximum(0, np.floor(p.min(0)).astype(int))
        hi = np.minimum([width - 1, height - 1], np.ceil(p.max(0)).astype(int))
        if np.any(hi < lo):
            continue
        matrix = np.vstack([p.T, np.ones(3)])
        if abs(np.linalg.det(matrix)) < 1e-10:
            continue
        xx, yy = np.meshgrid(np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1))
        x, y = xx.ravel(), yy.ravel()
        bary = np.column_stack([x, y, np.ones(len(x))]) @ np.linalg.inv(matrix).T
        weights = bary / hp[face, 2]
        z = 1 / np.maximum(weights.sum(1), 1e-12)
        keep = np.all(bary >= -1e-8, axis=1) & (z < depth[y, x])
        if not keep.any():
            continue
        w = weights[keep]
        w /= np.maximum(w.sum(1, keepdims=True), 1e-12)
        n = w @ normals[face]
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        result[y[keep], x[keep]] = n
        depth[y[keep], x[keep]] = z[keep]
    return result, depth


def lighting_basis(normals):
    x, y, z = np.moveaxis(normals, -1, 0)
    return np.stack(
        [np.ones_like(x), x, y, z, x * y, x * z, y * z, x * x - y * y, 3 * z * z - 1], -1
    )


def fit_lighting(basis, target):
    coeff = np.zeros(9)
    weights = np.ones(len(target))
    for _ in range(4):
        a = basis * np.sqrt(weights[:, None])
        b = target * np.sqrt(weights)
        coeff = np.linalg.solve(a.T @ a + np.eye(9) * 0.01, a.T @ b)
        residual = np.abs(basis @ coeff - target)
        weights = np.minimum(1, 0.03 / np.maximum(residual, 1e-9))
    return coeff


def metrics(pred, target, predicted_mask, target_mask):
    if not target_mask.any():
        raise ValueError("empty foreground cannot be scored")
    overlap = predicted_mask & target_mask
    union = predicted_mask | target_mask
    iou = overlap.sum() / max(1, union.sum())
    recall = overlap.sum() / max(1, target_mask.sum())
    rmse = float(np.sqrt(np.mean((pred[target_mask] - target[target_mask]) ** 2)))
    eroded = cv2.erode(overlap.astype("uint8"), np.ones((7, 7), "uint8")) > 0
    pg = np.stack([cv2.Sobel(pred, cv2.CV_64F, 1, 0), cv2.Sobel(pred, cv2.CV_64F, 0, 1)], -1)
    tg = np.stack([cv2.Sobel(target, cv2.CV_64F, 1, 0), cv2.Sobel(target, cv2.CV_64F, 0, 1)], -1)
    p, t = pg[eroded].ravel(), tg[eroded].ravel()
    pn, tn = np.linalg.norm(p), np.linalg.norm(t)
    norm = pn * tn
    correlation = float(p @ t / norm) if norm > 1e-12 else float(pn < 1e-12 and tn < 1e-12)
    std = max(0.03, float(target[target_mask].std()))
    nrmse = min(2.0, rmse / std)
    return {
        "iou": float(iou),
        "foreground_recall": float(recall),
        "linear_shading_rmse": rmse,
        "shading_nrmse": nrmse,
        "interior_gradient_correlation": correlation,
        "diagnostic_loss": float(0.4 * (1 - iou) + 0.4 * nrmse / 2 + 0.2 * (1 - correlation) / 2),
    }


def assess(
    root: Path, mesh_path: Path, validation: Path, out: Path, max_side=200, views_per_split=6
):
    if out.exists():
        raise ValueError("choose a fresh assessment directory")
    data = json.loads((root / "dataset.json").read_text())
    cameras, fit_images, _ = read_model(root / "train/sparse/0")
    val = json.loads(validation.read_text())
    heldout_names = {f"view_{i:04d}.png" for i in data["heldout"]}
    if any(i["name"] not in heldout_names for i in val["views"]):
        raise ValueError("validation poses must belong to the heldout set")
    valid_views = [i for i in val["views"] if i["status"] == "accepted"]
    if not valid_views:
        raise ValueError("no accepted validation poses")
    mesh = trimesh.load(mesh_path, force="mesh")
    samples = []
    for split, images in [("fit", fit_images), ("validation", valid_views)]:
        for idx in np.linspace(0, len(images) - 1, min(views_per_split, len(images))).astype(int):
            image = images[idx]
            camera = cameras[image["cid"]] if split == "fit" else val["camera"]
            r, t = np.array(image["r"]), np.array(image["t"])
            w, h = camera["width"], camera["height"]
            s = min(1.0, max_side / max(w, h))
            ww, hh = round(w * s), round(h * s)
            k = np.array(
                [[camera["fx"], 0, camera["cx"]], [0, camera["fy"], camera["cy"]], [0, 0, 1]]
            )
            p = np.diag([ww / w, hh / h, 1]) @ k @ np.column_stack([r, t])
            normals, depth = render_geometry(mesh, p, r, ww, hh)
            name = image["name"]
            rgb = cv2.cvtColor(cv2.imread(str(root / "reference/images" / name)), cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (ww, hh), interpolation=cv2.INTER_AREA) / 255.0
            linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
            gray = linear @ [0.2126, 0.7152, 0.0722]
            mask = (
                cv2.resize(
                    cv2.imread(str(root / "reference/masks" / name), 0),
                    (ww, hh),
                    interpolation=cv2.INTER_NEAREST,
                )
                >= 220
            )
            samples.append((split, name, lighting_basis(normals), np.isfinite(depth), gray, mask))
    train = [i for i in samples if i[0] == "fit"]
    a = np.concatenate([basis[mask & pm][::4] for _, _, basis, pm, _, mask in train])
    b = np.concatenate([target[mask & pm][::4] for _, _, _, pm, target, mask in train])
    if not len(a):
        raise ValueError("mesh has no supported fit pixels")
    coeff = fit_lighting(a, b)
    rows, tiles = [], []
    from boardgen.evaluation.benchmark_preview import thumbnail

    for split, name, basis, pm, target, mask in samples:
        pred = np.clip(basis @ coeff, 0, 1) * pm
        row = metrics(pred, target, pm, mask)
        row.update(split=split, name=name)
        rows.append(row)

        def display(x):
            srgb = np.where(
                x <= 0.0031308, 12.92 * x, 1.055 * np.maximum(x, 0) ** (1 / 2.4) - 0.055
            )
            return np.repeat(np.clip(srgb * 255, 0, 255).astype("uint8")[:, :, None], 3, axis=2)

        tile = np.hstack(
            [thumbnail(display(target * mask), 160, 200), thumbnail(display(pred), 160, 200)]
        )
        cv2.putText(
            tile, f"{split} {name}", (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 255, 0), 1
        )
        tiles.append(tile)
    out.mkdir(parents=True)
    cv2.imwrite(str(out / "shading.png"), cv2.cvtColor(np.vstack(tiles), cv2.COLOR_RGB2BGR))
    summary = {
        split: {
            key: float(np.mean([r[key] for r in rows if r["split"] == split]))
            for key in rows[0]
            if key not in {"split", "name"}
        }
        for split in ["fit", "validation"]
    }
    result = {
        "mesh_sha256": file_hash(mesh_path),
        "validation_sha256": file_hash(validation),
        "lighting_coefficients": coeff.tolist(),
        "summary": summary,
        "views": rows,
        "renderer": "texture-free normals; shared camera-space degree2 diffuse lighting fitted on fit pixels",
        "assessment": "Diagnostic proxy; ignores specularity and cast shadows; not geometric ground truth",
        "settings": {"max_side": max_side, "views_per_split": views_per_split},
        "faces": len(mesh.faces),
    }
    write_json(out / "metrics.json", result)
    return result


def main():
    p = argparse.ArgumentParser(
        description="Texture-free surface diagnostics under shared camera-space diffuse lighting.\n\nThis is an approximate shading model, not geometric ground truth: it ignores\nspecularity, cast shadows and ambient occlusion. Report all terms, not just score."
    )
    p.add_argument("root", type=Path)
    p.add_argument("mesh", type=Path)
    p.add_argument("validation", type=Path)
    p.add_argument("out", type=Path)
    args = p.parse_args()
    print(json.dumps(assess(args.root, args.mesh, args.validation, args.out)["summary"]))


if __name__ == "__main__":
    main()

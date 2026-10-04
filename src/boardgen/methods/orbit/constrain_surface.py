import sys
import time
from pathlib import Path

import cv2
import numpy as np
import trimesh

from boardgen.evaluation.benchmark3d import write_json
from boardgen.evaluation.benchmark_preview import read_colmap


def constrain(root, source, out, support_threshold=0.94):
    start = time.perf_counter()
    mesh = trimesh.load(source, force="mesh")
    before = len(mesh.faces)
    ph = np.column_stack([mesh.vertices, np.ones(len(mesh.vertices))])
    views = read_colmap(root / "train")
    support = np.zeros(len(ph))
    for name, proj, w, h in views:
        hp = ph @ proj.T
        xy = np.rint(hp[:, :2] / np.maximum(hp[:, 2:], 1e-9)).astype(int)
        valid = (hp[:, 2] > 0) & (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
        ids = np.flatnonzero(valid)
        mask = cv2.imread(str(root / "reference/masks" / name), 0) > 128
        mask = cv2.dilate(mask.astype("uint8"), np.ones((3, 3), "uint8"))
        support[ids] += mask[xy[ids, 1], xy[ids, 0]]
    keep = (support / len(views) >= support_threshold)[mesh.faces].all(1)
    mesh.update_faces(keep)
    mesh.remove_unreferenced_vertices()
    mesh.export(out)
    row = {
        "faces_before": before,
        "faces": len(mesh.faces),
        "mask_support": support_threshold,
        "tolerance": "1 pixel dilated fit-only mattes",
        "seconds": time.perf_counter() - start,
        "geometry_changed": "Triangle removal only; retained body vertices and UV unchanged",
    }
    write_json(out.with_suffix(".json"), row)
    print(row, flush=True)
    return mesh


if __name__ == "__main__":
    constrain(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))

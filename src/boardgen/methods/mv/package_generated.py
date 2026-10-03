import argparse
import time
from pathlib import Path

import numpy as np
import pymeshlab as ml
import trimesh

from boardgen.evaluation.benchmark3d import file_hash, write_json


def package(source: Path, out: Path, faces: int = 120000):
    if out.with_suffix(".glb").exists():
        raise ValueError("fresh output required")
    started = time.perf_counter()
    source_mesh = trimesh.load(source, force="mesh")
    parts = sorted(
        source_mesh.split(only_watertight=False), key=lambda m: len(m.faces), reverse=True
    )
    if not parts:
        raise ValueError("empty generated surface")
    mesh = parts[0].copy()
    offset = -mesh.bounds.mean(axis=0)
    offset[1] = -mesh.bounds[0, 1]
    scale = 1.0 / np.ptp(mesh.vertices[:, 1])
    mesh.apply_translation(offset)
    mesh.apply_scale(scale)
    out.parent.mkdir(parents=True, exist_ok=True)
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=np.zeros((len(mesh.vertices), 2)),
        material=trimesh.visual.material.PBRMaterial(
            baseColorFactor=[160, 160, 160, 255], metallicFactor=0, roughnessFactor=0.7
        ),
    )
    mesh.export(out.with_name(out.name + "-full.glb"))
    if len(mesh.faces) > faces:
        ms = ml.MeshSet()
        ms.add_mesh(ml.Mesh(vertex_matrix=mesh.vertices, face_matrix=mesh.faces))
        ms.meshing_decimation_quadric_edge_collapse(
            targetfacenum=faces, preservenormal=True, preservetopology=True
        )
        simplified = trimesh.Trimesh(
            ms.current_mesh().vertex_matrix(), ms.current_mesh().face_matrix(), process=False
        )
        simplified.visual = trimesh.visual.TextureVisuals(
            uv=np.zeros((len(simplified.vertices), 2)), material=mesh.visual.material
        )
    else:
        simplified = mesh

    review_offset = -simplified.bounds.mean(axis=0)
    review_offset[1] = -simplified.bounds[0, 1]
    review_scale = 1.0 / np.ptp(simplified.vertices[:, 1])
    simplified.apply_translation(review_offset)
    simplified.apply_scale(review_scale)
    simplified.export(out.with_suffix(".glb"))
    result = {
        "source": str(source.resolve()),
        "source_sha256": file_hash(source),
        "components": [
            {
                "faces": len(part.faces),
                "watertight": bool(part.is_watertight),
                "bounds": part.bounds.tolist(),
            }
            for part in parts
        ],
        "operation": "Largest connected component only; translation and uniform scale, no stretching, filling or carving; full source retained.",
        "translation_before_scale": offset.tolist(),
        "uniform_scale": float(scale),
        "normalization": "Y-up, base0, height1; source orientation retained",
        "full_faces": len(mesh.faces),
        "review_faces": len(simplified.faces),
        "review_normalization": {
            "translation_before_scale": review_offset.tolist(),
            "uniform_scale": float(review_scale),
        },
        "review_watertight": bool(simplified.is_watertight),
        "review_components": len(simplified.split(only_watertight=False)),
        "review_sha256": file_hash(out.with_suffix(".glb")),
        "full_sha256": file_hash(out.with_name(out.name + "-full.glb")),
        "seconds": time.perf_counter() - started,
    }
    write_json(out.with_suffix(".json"), result)
    print(out.with_suffix(".glb"), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Keep the connected figure, normalize uniformly, and export a review mesh.\n\nDetached generated sheets/specks are excluded, but connected surface defects\nremain. The source is retained; this is not silhouette carving or hole filling."
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--faces", type=int, default=120000)
    args = parser.parse_args()
    package(args.source, args.out, args.faces)

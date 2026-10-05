from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import trimesh
from PIL import ImageEnhance

from boardgen.common.figure_names import figure_names
from boardgen.evaluation.benchmark3d import file_hash, write_json
from boardgen.methods.repair.repair_figure_mesh import close_base_holes

DEFAULT_OBJECTS = [3, 4, 6, 13, 15, 17, 18, 22, 25, 26]


def boundary_count(mesh, ceiling):
    topology = trimesh.Trimesh(mesh.vertices.copy(), mesh.faces.copy(), process=False)
    topology.merge_vertices()
    counts = np.bincount(topology.edges_unique_inverse, minlength=len(topology.edges_unique))
    edges = topology.edges_unique[counts == 1]
    return int((topology.vertices[edges, 1].max(axis=1) <= ceiling).sum())


def enhance_textures(scene, factor):
    textures = 0
    for mesh in scene.geometry.values():
        if mesh.visual.kind != "texture":
            continue
        material = copy.deepcopy(mesh.visual.material)
        attribute = "baseColorTexture" if hasattr(material, "baseColorTexture") else "image"
        image = getattr(material, attribute, None)
        if image is None or min(image.size) < 128:
            continue
        updated = ImageEnhance.Contrast(image.convert("RGB")).enhance(factor)
        if "A" in image.getbands():
            updated.putalpha(image.getchannel("A"))
        setattr(material, attribute, updated)
        mesh.visual.material = material
        textures += 1
    return textures


def run(package, output, objects, fraction=0.12, contrast=1.10):
    if not 0 < fraction <= 0.12 or not 1 <= contrast <= 1.20:
        raise ValueError("Repair is limited to the bottom 12%; contrast must be 1–1.2")
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "source_package": str(package),
        "base_fraction": fraction,
        "contrast": contrast,
        "objects": {},
    }
    names = figure_names()
    for number in objects:
        obj = f"obj_{number:02d}"
        meta = json.loads((package / obj / "asset/asset.json").read_text())
        gallery = json.loads((package / "gallery" / f"comparison-{obj}.json").read_text())
        variants = {}
        report["objects"][obj] = {"name": names[obj]["name"], "variants": variants}
        for kind in ["native", "fusion"]:
            source = package / obj / "asset" / f"{kind}.glb"
            sha = file_hash(source)
            existing = next(
                (entry for entry in gallery["candidates"] if entry["mesh_sha256"] == sha), None
            )
            provenance = (
                package / "gallery" / existing["coordinate_provenance"]
                if existing
                else Path(meta["refinement"]["source_workspace"]) / f"{kind}-local-hull/result.json"
            )
            result = json.loads(provenance.read_text())
            if result["photo_sha256"] != sha:
                raise ValueError(f"Mismatched original provenance: {obj}/{kind}")
            scene = trimesh.load(source, force="scene", process=False)
            if any(
                not np.allclose(scene.graph[node][0], np.eye(4))
                for node in scene.graph.nodes_geometry
            ):
                raise ValueError("Repair requires geometry in the exported model coordinates")
            low, high = scene.bounds[:, 1]
            ceiling = low + fraction * (high - low)
            repairs = []
            for name, mesh in list(scene.geometry.items()):
                before = boundary_count(mesh, ceiling)
                repaired, details = close_base_holes(mesh, ceiling)
                assert np.array_equal(repaired.vertices[: len(mesh.vertices)], mesh.vertices)
                assert np.array_equal(repaired.faces[: len(mesh.faces)], mesh.faces)
                if mesh.visual.kind == "texture" and mesh.visual.uv is not None:
                    assert np.array_equal(repaired.visual.uv[: len(mesh.vertices)], mesh.visual.uv)
                scene.geometry[name] = repaired
                repairs.append(
                    {
                        "part": name,
                        "base_boundary_edges_before": before,
                        "base_boundary_edges_after": boundary_count(repaired, ceiling),
                        **details,
                    }
                )
            added = sum(item["added_faces"] for item in repairs)
            entry = {
                "source": str(source),
                "source_sha256": sha,
                "source_provenance": str(provenance),
                "added_faces": added,
                "repairs": repairs,
                "original_vertices_faces_uv_preserved": True,
                "original_metrics": meta["validation"][kind]["validation"],
            }
            variants[kind] = entry
            if added:
                dest = output / obj / kind
                dest.mkdir(parents=True)
                scene.export(dest / "photo.glb")
                result.update(
                    photo_sha256=file_hash(dest / "photo.glb"),
                    base_boundary_repair=repairs,
                    repair_source=str(source),
                    repair_source_sha256=sha,
                    assessment="Base-boundary interpolation only. Original geometry, UV and cameras unchanged; not a new measured scan. Remaining open boundaries and sculpt errors may persist.",
                )
                write_json(dest / "result.json", result)
                entry["candidate"] = str(dest)
            if kind == "native" and enhance_textures(scene, contrast):
                dest = output / obj / "native-contrast"
                dest.mkdir(parents=True)
                scene.export(dest / "photo.glb")
                enhanced = {
                    **result,
                    "photo_sha256": file_hash(dest / "photo.glb"),
                    "repair_source": str(source),
                    "repair_source_sha256": sha,
                    "base_boundary_repair": repairs,
                    "texture_adjustment": {
                        "method": "Pillow ImageEnhance.Contrast",
                        "factor": contrast,
                        "alpha_and_uv_preserved": True,
                    },
                    "assessment": "Native photographic atlas with mild contrast; base caps are an interpolation prior. Contrast does not restore missing geometric detail.",
                }
                write_json(dest / "result.json", enhanced)
                entry["contrast_candidate"] = str(dest)
            print(
                obj,
                kind,
                "added",
                added,
                "remaining base edges",
                sum(r["base_boundary_edges_after"] for r in repairs),
                flush=True,
            )
            write_json(output / "repair-report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Create bounded base-cap and mild native-texture candidates without retraining."
    )
    parser.add_argument("package", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--objects", nargs="+", type=int, default=DEFAULT_OBJECTS)
    args = parser.parse_args()
    run(args.package.resolve(), args.output.resolve(), args.objects)


if __name__ == "__main__":
    main()

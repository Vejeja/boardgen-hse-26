import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from boardgen.evaluation.benchmark3d import file_hash, write_json


def replace(source: Path, dataset: Path, output: Path):
    provenance = json.loads((source / "result.json").read_text())
    generation = provenance["generation_provenance"]
    data = json.loads((dataset / "dataset.json").read_text())
    samples, photographs = [], []
    for view in generation["inputs"]:
        index = view["view"]
        if index not in data["fit"]:
            raise ValueError("Plastic colour must use fit photographs only")
        name = f"view_{index:04d}.png"
        photo = dataset / "reference/images" / name
        mask_path = dataset / "reference/masks" / name
        rgb = np.asarray(Image.open(photo).convert("RGB"))
        mask = np.asarray(Image.open(mask_path).convert("L")) > 128
        if rgb.shape[:2] != mask.shape or not mask.any():
            raise ValueError("Colour photograph and support mask differ or are empty")
        samples.append(np.median(rgb[mask], axis=0))
        photographs.append(
            {"name": name, "photo_sha256": file_hash(photo), "mask_sha256": file_hash(mask_path)}
        )
    colour = np.rint(np.median(samples, axis=0)).astype("uint8")
    scene = trimesh.load(source / "photo.glb", force="scene", process=False)
    for mesh in scene.geometry.values():
        normals = mesh.vertex_normals.copy()
        mesh.visual = trimesh.visual.TextureVisuals(
            uv=mesh.visual.uv.copy(),
            material=trimesh.visual.material.PBRMaterial(
                baseColorFactor=[*colour.tolist(), 255], metallicFactor=0, roughnessFactor=0.8
            ),
        )
        mesh.vertex_normals = normals
    output.mkdir(parents=True, exist_ok=False)
    scene.export(output / "photo.glb")
    shutil.copy2(source / "photo.glb", output / "photographic-original.glb")
    provenance.update(
        photo_sha256=file_hash(output / "photo.glb"),
        appearance={
            "method": "uniform_observed_plastic_colour",
            "median_srgb": colour.tolist(),
            "fit_photographs": photographs,
            "source_mesh_sha256": file_hash(source / "photo.glb"),
            "geometry_unchanged": True,
            "limitation": "Uniform captured colour, not measured albedo. Photographic detail and baked lighting removed; generated surface detail remains approximate.",
        },
    )
    write_json(output / "result.json", provenance)
    return colour.tolist()


def main():
    parser = argparse.ArgumentParser(
        description="Replace a distorted photographic atlas with a measured uniform plastic colour.\n\nGeometry stays fixed. This deliberately removes photographic surface detail;\nthe original textured candidate remains available for comparison."
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(replace(args.source.resolve(), args.dataset.resolve(), args.output.resolve()))


if __name__ == "__main__":
    main()

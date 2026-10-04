import json

import numpy as np
import trimesh
from PIL import Image

from boardgen.methods.repair.use_observed_plastic_material import replace


def test_plain_plastic_keeps_geometry_and_original_comparison(tmp_path):
    source, dataset, output = tmp_path / "source", tmp_path / "dataset", tmp_path / "output"
    source.mkdir()
    for part in ["images", "masks"]:
        (dataset / "reference" / part).mkdir(parents=True)
    (dataset / "dataset.json").write_text(json.dumps({"fit": [0]}))
    Image.new("RGB", (12, 12), (190, 180, 160)).save(dataset / "reference/images/view_0000.png")
    Image.new("L", (12, 12), 255).save(dataset / "reference/masks/view_0000.png")
    mesh = trimesh.creation.box()
    mesh.visual = trimesh.visual.TextureVisuals(uv=np.zeros((len(mesh.vertices), 2)))
    mesh.export(source / "photo.glb")
    (source / "result.json").write_text(
        json.dumps({"generation_provenance": {"inputs": [{"view": 0}]}})
    )
    assert replace(source, dataset, output) == [190, 180, 160]
    original = trimesh.load(source / "photo.glb", force="mesh", process=False)
    cleaned = trimesh.load(output / "photo.glb", force="mesh", process=False)
    np.testing.assert_array_equal(cleaned.vertices, original.vertices)
    np.testing.assert_array_equal(cleaned.faces, original.faces)
    np.testing.assert_array_equal(cleaned.visual.uv, original.visual.uv)
    assert (output / "photographic-original.glb").read_bytes() == (
        source / "photo.glb"
    ).read_bytes()
    assert cleaned.visual.material.baseColorTexture is None

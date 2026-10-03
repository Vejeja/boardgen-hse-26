import numpy as np
import trimesh

from boardgen.methods.mv.package_generated import package


def test_component_cleanup_preserves_proportions_and_source(tmp_path):
    body = trimesh.creation.box(extents=[2, 4, 1]).subdivide()
    body.apply_translation([3, 7, -2])
    sheet = trimesh.creation.box(extents=[100, 0.01, 100])
    sheet.apply_translation([0, -20, 0])
    source = tmp_path / "raw.glb"
    trimesh.util.concatenate([body, sheet]).export(source)
    original = source.read_bytes()
    out = tmp_path / "figure"
    result = package(source, out, faces=1000)
    mesh = trimesh.load(out.with_suffix(".glb"), force="mesh")
    np.testing.assert_allclose(mesh.extents, [0.5, 1, 0.25])
    np.testing.assert_allclose(mesh.bounds.mean(axis=0), [0, 0.5, 0])
    assert len(mesh.faces) == len(body.faces)
    assert result["review_components"] == 1
    assert result["review_watertight"]
    assert source.read_bytes() == original

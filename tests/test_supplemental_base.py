import numpy as np
import trimesh

from boardgen.methods.repair.repair_figure_mesh import flatten_supplemental_base


def test_flatten_supplement_preserves_body_and_upper_base():
    body = trimesh.creation.box(extents=[0.4, 0.9, 0.4])
    body.apply_translation([0, 0.55, 0])
    body.visual = trimesh.visual.TextureVisuals(uv=np.zeros((len(body.vertices), 2)))
    base = trimesh.creation.box(extents=[0.6, 0.2, 0.6])
    base.apply_translation([0, 0.1, 0])
    scene = trimesh.Scene()
    scene.add_geometry(body, geom_name="native_body")
    scene.add_geometry(base, geom_name="inferred_boundary_closure")
    original, uv = body.vertices.copy(), body.visual.uv.copy()
    upper = base.vertices[base.vertices[:, 1] >= 0.1].copy()
    edits = flatten_supplemental_base(scene)
    np.testing.assert_array_equal(body.vertices, original)
    np.testing.assert_array_equal(body.visual.uv, uv)
    np.testing.assert_array_equal(base.vertices[base.vertices[:, 1] > 0.1 + 1e-8], upper)
    assert np.isclose(base.bounds[0, 1], 0.1)
    assert edits[0]["clamped_vertices"] == 4

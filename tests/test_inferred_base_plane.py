import importlib.util
from pathlib import Path

import numpy as np
import pytest
import trimesh

SOURCE = Path(__file__).resolve().parents[1] / "src/boardgen/methods/repair/local_hull.py"
SPEC = importlib.util.spec_from_file_location("base_prior", SOURCE)
HULL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HULL)


def test_bottom_prior_preserves_visible_body_and_uv():
    body = trimesh.creation.box()
    body.apply_translation([0, 0.5, 0])
    uv = np.arange(len(body.vertices) * 2).reshape(-1, 2) / 20
    body.visual = trimesh.visual.TextureVisuals(uv=uv.copy())
    original = body.vertices.copy()
    result = HULL.flatten_native_bottom(body, 0.1)
    assert np.array_equal(body.vertices[original[:, 1] >= 0.1], original[original[:, 1] >= 0.1])
    assert np.array_equal(body.visual.uv, uv)
    assert body.bounds[0, 1] == pytest.approx(0.1)
    assert result["clamped_native_vertices"] == 4


@pytest.mark.parametrize("plane", [0.16, float("nan"), -0.01])
def test_unbounded_bottom_prior_is_rejected_without_mutation(plane):
    body = trimesh.creation.box()
    body.apply_translation([0, 0.5, 0])
    original = body.vertices.copy()
    with pytest.raises(ValueError, match="lowest 15%"):
        HULL.flatten_native_bottom(body, plane)
    assert np.array_equal(body.vertices, original)

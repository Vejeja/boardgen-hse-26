import numpy as np
import trimesh
from PIL import Image

from boardgen.evaluation.benchmark_surface import (
    fit_lighting,
    lighting_basis,
    metrics,
    render_geometry,
)


def test_surface_renderer_ignores_baked_texture_and_resolves_occlusion():
    front = np.array([[1, 1, 1], [5, 1, 1], [1, 5, 1]], float)
    mesh = trimesh.Trimesh(
        vertices=np.vstack([front, front * 2]), faces=[[0, 1, 2], [3, 4, 5]], process=False
    )
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=np.zeros((6, 2)), image=Image.new("RGB", (4, 4), "red")
    )
    projection = np.column_stack([np.eye(3), np.zeros(3)])
    normals, depth = render_geometry(mesh, projection, np.eye(3), 8, 8)
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=np.ones((6, 2)), image=Image.new("RGB", (4, 4), "blue")
    )
    mesh.faces = mesh.faces[::-1]
    other, distance = render_geometry(mesh, projection, np.eye(3), 8, 8)
    np.testing.assert_allclose(normals, other)
    np.testing.assert_allclose(depth, distance)
    assert depth[2, 2] == 1
    assert not np.isfinite(depth[7, 7])


def test_shading_proxy_penalizes_flat_geometry_with_identical_silhouette():
    y, x = np.mgrid[-1:1:32j, -1:1:32j]
    mask = x * x + y * y < 0.9
    target = 0.15 + 0.08 * x + 0.08 * y * y
    good = metrics(target, target, mask, mask)
    flat = metrics(np.full_like(target, target[mask].mean()), target, mask, mask)
    assert flat["iou"] == good["iou"] == 1
    assert flat["linear_shading_rmse"] > good["linear_shading_rmse"]
    assert flat["diagnostic_loss"] > good["diagnostic_loss"]


def test_shared_lighting_recovers_known_diffuse_field():
    rng = np.random.default_rng(4)
    normals = rng.normal(size=(500, 3))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    basis = lighting_basis(normals)
    expected = np.array([0.3, 0.03, -0.04, 0.06, 0.005, 0, 0, 0.008, 0.004])
    recovered = fit_lighting(basis, basis @ expected)
    np.testing.assert_allclose(recovered, expected, atol=1e-4)

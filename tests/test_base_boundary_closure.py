import numpy as np
import trimesh
from PIL import Image

from boardgen.methods.repair.repair_figure_mesh import close_base_holes


def open_box(center_y):
    mesh = trimesh.creation.box(extents=[0.1, 0.04, 0.1])
    mesh.apply_translation([0, center_y, 0])

    mesh.update_faces(mesh.face_normals[:, 1] < 0.9)
    return mesh


def test_only_complete_base_caps_are_added_and_original_uv_is_retained():
    mesh = trimesh.util.concatenate([open_box(0.03), open_box(0.5)])
    uv = np.arange(len(mesh.vertices) * 2).reshape(-1, 2) / 40
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=Image.new("RGB", (8, 8), (175, 170, 155))
    )
    mesh.visual = trimesh.visual.TextureVisuals(uv=uv.copy(), material=material)
    vertices, faces = mesh.vertices.copy(), mesh.faces.copy()
    normals = mesh.vertex_normals.copy()
    repaired, report = close_base_holes(mesh, ceiling=0.10)
    assert report["closed_caps"] == 1
    assert report["rejected_caps"] == 1
    assert report["added_faces"] == 2
    assert np.array_equal(repaired.vertices[: len(vertices)], vertices)
    assert np.array_equal(repaired.faces[: len(faces)], faces)
    assert np.array_equal(repaired.visual.uv[: len(vertices)], uv)
    assert np.array_equal(repaired.vertex_normals[: len(vertices)], normals)
    assert repaired.visual.material is material
    assert repaired.vertices[len(vertices) :, 1].max() <= 0.10


def test_a_boundary_spanning_base_and_sculpt_is_rejected_whole():
    mesh = open_box(0.08)
    mesh.vertices[mesh.vertices[:, 1] > 0.08, 1] = [0.07, 0.07, 0.15, 0.15]
    repaired, report = close_base_holes(mesh, ceiling=0.10)
    assert report["added_faces"] == 0
    assert repaired is mesh


def test_base_loops_touching_at_one_vertex_are_closed_independently():
    first, second = open_box(0.03), open_box(0.03)
    second.apply_translation([0.1, 0, 0.1])
    mesh = trimesh.util.concatenate([first, second])
    repaired, report = close_base_holes(mesh, ceiling=0.1)
    assert report["closed_caps"] == 2
    assert report["added_faces"] == 4
    assert np.array_equal(repaired.faces[: len(mesh.faces)], mesh.faces)


def test_nonmanifold_edges_elsewhere_do_not_prevent_local_base_repair():
    first, second = open_box(0.03), trimesh.creation.box()
    second.apply_translation([0.5, 1, 0])
    second.faces = np.vstack([second.faces, second.faces[:1]])
    mesh = trimesh.util.concatenate([first, second])
    repaired, report = close_base_holes(mesh, ceiling=0.1)
    assert report["closed_caps"] == 1
    assert report["added_faces"] == 2
    assert np.array_equal(repaired.faces[: len(mesh.faces)], mesh.faces)

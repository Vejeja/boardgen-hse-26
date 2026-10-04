import numpy as np
import trimesh

from boardgen.methods.repair.repair_figure_mesh import negative_space_faces


def test_photographic_exclusion_preserves_base_and_vertices():
    vertices = np.array(
        [[2, 0.4, 1], [3, 0.4, 1], [2, 0.5, 1], [2, 0.04, 1], [3, 0.04, 1], [2, 0.06, 1]]
    )
    mesh = trimesh.Trimesh(vertices, [[0, 1, 2], [3, 4, 5]], process=False)
    projection = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    before = mesh.vertices.copy()
    remove = negative_space_faces(mesh, np.eye(4), projection, np.ones((10, 10), bool), [0.12, 0.6])
    assert remove.tolist() == [True, False]
    assert np.array_equal(mesh.vertices, before)

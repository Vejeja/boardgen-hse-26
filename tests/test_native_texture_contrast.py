import numpy as np
import trimesh
from PIL import Image

from boardgen.methods.repair.repair_figure_bases import enhance_textures


def test_native_contrast_preserves_uv_alpha_and_original_material():
    mesh = trimesh.creation.box()
    pixels = np.zeros((128, 128, 4), dtype=np.uint8)
    pixels[:, :64] = [70, 90, 100, 100]
    pixels[:, 64:] = [200, 180, 160, 220]
    image = Image.fromarray(pixels)
    material = trimesh.visual.material.PBRMaterial(baseColorTexture=image, metallicFactor=0)
    uv = np.arange(len(mesh.vertices) * 2).reshape(-1, 2) / 20
    mesh.visual = trimesh.visual.TextureVisuals(uv=uv.copy(), material=material)
    vertices, faces, normals = mesh.vertices.copy(), mesh.faces.copy(), mesh.vertex_normals.copy()
    scene = trimesh.Scene(mesh)
    assert enhance_textures(scene, 1.1) == 1
    updated = np.asarray(mesh.visual.material.baseColorTexture)
    assert np.array_equal(updated[:, :, 3], pixels[:, :, 3])
    assert not np.array_equal(updated[:, :, :3], pixels[:, :, :3])
    assert np.array_equal(np.asarray(material.baseColorTexture), pixels)
    assert mesh.visual.material.metallicFactor == 0
    assert np.array_equal(mesh.vertices, vertices)
    assert np.array_equal(mesh.faces, faces)
    assert np.array_equal(mesh.vertex_normals, normals)
    assert np.array_equal(mesh.visual.uv, uv)


def test_small_flat_colour_patches_are_not_enhanced():
    mesh = trimesh.creation.box()
    material = trimesh.visual.material.PBRMaterial(baseColorTexture=Image.new("RGB", (32, 32)))
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=np.zeros((len(mesh.vertices), 2)), material=material
    )
    assert enhance_textures(trimesh.Scene(mesh), 1.1) == 0
    assert mesh.visual.material is material

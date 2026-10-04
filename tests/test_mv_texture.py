import cv2
import numpy as np
import trimesh
from PIL import Image

from boardgen.methods.mv.blend_mv_texture import raster_atlas, sample_pixels
from boardgen.methods.mv.texture_mv import camera_pose, extend_foreground, restore_uv


def test_pixel_sampling_handles_more_than_opencv_short_limit():
    image = np.full((4, 5, 3), [30, 90, 150], dtype="uint8")
    xy = np.tile([2.0, 1.0], (60000, 1))
    samples = sample_pixels(image, xy, cv2.INTER_LINEAR)
    np.testing.assert_array_equal(samples, np.broadcast_to([30, 90, 150], (60000, 3)))


def test_empty_and_degenerate_uvs_keep_plain_colour():
    mesh = trimesh.Trimesh([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]], process=False)
    mesh.visual = trimesh.visual.TextureVisuals(uv=np.zeros((3, 2)))
    colors = np.tile([50, 70, 90], (3, 1))
    atlas = raster_atlas(mesh, colors, 32)
    np.testing.assert_array_equal(atlas, np.broadcast_to([50, 70, 90], atlas.shape))


def test_matte_padding_preserves_every_observed_rgb_pixel():
    rgb = np.full((16, 16, 3), 230, dtype="uint8")
    mask = np.zeros((16, 16), dtype=bool)
    mask[4:12, 4:12] = True
    rgb[mask] = [20, 40, 80]
    padded = extend_foreground(rgb, mask)
    np.testing.assert_array_equal(padded[mask], rgb[mask])
    np.testing.assert_array_equal(padded[3, 8], [20, 40, 80])


def test_camera_fit_rotates_without_mirroring_or_stretching():
    k = np.array([[1000, 0, 200], [0, 1000, 260], [0, 0, 1]])
    for yaw in [0, 90, 180, 270]:
        r, t = camera_pose([yaw, 20, 3, np.log(3), 0.02, -0.03], k, 400, 520)
        np.testing.assert_allclose(r.T @ r, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(r), 1)
        assert (r @ [0, 0.5, 0] + t)[2] > 0


def test_photo_export_preserves_surface_and_uses_diffuse_material(tmp_path):
    canonical = trimesh.Trimesh([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]], process=False)
    Image.new("RGB", (8, 8), (20, 80, 160)).save(tmp_path / "atlas.png")
    (tmp_path / "atlas.mtl").write_text("newmtl photo\nmap_Kd atlas.png\n")
    obj = tmp_path / "textured.obj"
    obj.write_text(
        "mtllib atlas.mtl\nv 0 0 0\nv 1 0 0\nv 0 1 0\n"
        "vt 0 0\nvt 1 0\nvt 0 1\nusemtl photo\nf 1/1 2/2 3/3\n"
    )
    output = tmp_path / "photo.glb"
    report = restore_uv(canonical, obj, output)
    result = trimesh.load(output, force="mesh", process=False)
    np.testing.assert_allclose(result.vertices, canonical.vertices)
    np.testing.assert_array_equal(result.faces, canonical.faces)
    assert report["triangle_index_sets_identical"]
    assert result.visual.material.metallicFactor == 0
    np.testing.assert_array_equal(
        np.asarray(result.visual.material.baseColorTexture)[..., :3],
        np.broadcast_to([20, 80, 160], (8, 8, 3)),
    )

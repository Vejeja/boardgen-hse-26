import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from boardgen.evaluation.figure_comparison import camera_matrices, model_to_world


def test_clip_projection_matches_original_pixels_after_model_normalization():
    camera = {"width": 640, "height": 480, "fx": 900, "fy": 850, "cx": 310, "cy": 255}
    r = Rotation.from_euler("xyz", [10, 25, -3], degrees=True).as_matrix()
    t = np.array([0.1, -0.2, 2])
    transform = np.eye(4)
    transform[:3, :3] = 2.3 * Rotation.from_euler("y", 12, degrees=True).as_matrix()
    transform[:3, 3] = [0.2, 0.8, -0.1]
    points = np.array([[0.1, 0.2, 0.3, 1], [-0.2, 0.1, 0, 1]])
    normalized = points @ transform.T
    projection, view = camera_matrices(camera, r, t)
    clip = normalized @ (projection @ view @ model_to_world(transform)).T
    ndc = clip[:, :2] / clip[:, 3:4]
    screen = (ndc * [1, -1] + 1) * [camera["width"] / 2, camera["height"] / 2]
    cv = points[:, :3] @ r.T + t
    expected = cv[:, :2] / cv[:, 2:] * [900, 850] + [310, 255]
    np.testing.assert_allclose(screen, expected, atol=1e-10)


@pytest.mark.parametrize("scale", [[1, 2, 1], [-1, 1, 1], [0, 1, 1]])
def test_comparison_rejects_axis_stretching_and_reflection(scale):
    with pytest.raises(ValueError, match="similarity"):
        model_to_world(np.diag([*scale, 1]))

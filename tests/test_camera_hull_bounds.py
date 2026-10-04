import numpy as np
import pytest

from boardgen.methods.orbit import hull_bounds as AUDIT
from boardgen.methods.orbit.calibrate_orbit import rotations


def test_frusta_bound_the_complete_observed_box():
    corners = np.array(
        [[x, y, z, 1] for x in [-0.2, 0.2] for y in [-0.3, 0.3] for z in [-0.15, 0.15]]
    )
    k = np.array([[150, 0, 64], [0, 150, 64], [0, 0, 1]])
    projections, masks = [], []
    for r in rotations(np.radians([0, 90, 180, 270]), 0, 0):
        p = k @ np.column_stack([r, [0, 0, 1]])
        projected = corners @ p.T
        xy = projected[:, :2] / projected[:, 2:]
        low, high = np.floor(xy.min(0)).astype(int), np.ceil(xy.max(0)).astype(int)
        mask = np.zeros((128, 128), bool)
        mask[low[1] : high[1] + 1, low[0] : high[0] + 1] = True
        projections.append(p)
        masks.append(mask)
    bounds = AUDIT.frustum_bounds(masks, projections)
    assert (bounds[:, 0] <= corners[:, :3].min(0)).all()
    assert (bounds[:, 1] >= corners[:, :3].max(0)).all()

    assert (np.diff(bounds, axis=1).ravel() < [0.6, 0.9, 0.5]).all()


def test_single_view_does_not_invent_depth_bounds():
    mask = np.zeros((64, 64), bool)
    mask[20:45, 20:45] = True
    projection = np.array([[100, 0, 32, 32], [0, 100, 32, 32], [0, 0, 1, 1]])
    with pytest.raises(ValueError, match="bounded common volume"):
        AUDIT.frustum_bounds([mask], [projection])

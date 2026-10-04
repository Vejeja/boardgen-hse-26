import numpy as np
from scipy.optimize import linprog


def frustum_bounds(masks, projections, padding_px=2):
    inequalities, limits = [], []
    for mask, p in zip(masks, projections, strict=True):
        yy, xx = np.nonzero(mask)
        if not len(xx):
            raise ValueError("Empty foreground cannot define a camera hull")
        for axis, low, high in [(0, xx.min(), xx.max()), (1, yy.min(), yy.max())]:
            for row in [(low - padding_px) * p[2] - p[axis], p[axis] - (high + padding_px) * p[2]]:
                inequalities.append(row[:3])
                limits.append(-row[3])
        inequalities.append(-p[2, :3])
        limits.append(p[2, 3] - 1e-6)
    bounds = np.zeros((3, 2))
    for axis in range(3):
        for j, sign in enumerate([1, -1]):
            cost = np.eye(3)[axis] * sign
            result = linprog(cost, A_ub=inequalities, b_ub=limits, bounds=[(None, None)] * 3)
            if not result.success:
                raise ValueError("Fit foreground frusta do not define a bounded common volume")
            bounds[axis, j] = result.x[axis]
    if not np.isfinite(bounds).all() or (np.diff(bounds, axis=1) <= 0).any():
        raise ValueError("Degenerate camera hull volume")
    margin = np.diff(bounds, axis=1).ravel() * 0.02
    return bounds + np.column_stack([-margin, margin])

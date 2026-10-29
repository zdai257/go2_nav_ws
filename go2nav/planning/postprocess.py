"""Path post-processing shared by planners: shortcutting, resampling, yaw assignment."""

from __future__ import annotations

import numpy as np

from ..core.types import wrap_angle


def shortcut(xy: np.ndarray, costmap) -> np.ndarray:
    """Greedy line-of-sight pruning: from each kept waypoint jump to the farthest visible one."""
    xy = np.asarray(xy, dtype=float)
    if len(xy) <= 2:
        return xy
    out = [xy[0]]
    i = 0
    n = len(xy)
    while i < n - 1:
        j = n - 1
        while j > i + 1 and not costmap.segment_free(xy[i], xy[j]):
            j -= 1
        out.append(xy[j])
        i = j
    return np.array(out)


def densify(path: np.ndarray, step: float) -> np.ndarray:
    """Resample a (N,2|3) polyline at ~``step`` metres; yaw (if present) interpolated on the circle."""
    path = np.asarray(path, dtype=float)
    if len(path) < 2 or step <= 0:
        return path
    out = [path[0]]
    for a, b in zip(path[:-1], path[1:]):
        seg = np.linalg.norm(b[:2] - a[:2])
        n = max(1, int(np.ceil(seg / step)))
        t = np.linspace(0.0, 1.0, n + 1)[1:, None]
        pts = a[None, :] + t * (b - a)[None, :]
        if path.shape[1] >= 3:
            pts[:, 2] = wrap_angle(a[2] + t[:, 0] * wrap_angle(b[2] - a[2]))
        out.extend(pts)
    return np.array(out)


def assign_yaw(xy: np.ndarray, start_yaw: float, goal_yaw: float, mode: str = "tangent") -> np.ndarray:
    xy = np.asarray(xy, dtype=float)
    n = len(xy)
    yaw = np.zeros(n)
    if mode == "goal":
        yaw[:] = goal_yaw
    else:
        if n >= 2:
            d = np.diff(xy, axis=0)
            yaw[:-1] = np.arctan2(d[:, 1], d[:, 0])
        yaw[-1] = goal_yaw
        if n == 1:
            yaw[0] = goal_yaw
    return np.column_stack([xy, yaw])

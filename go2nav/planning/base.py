"""Global planner interface.

Writing a new planner::

    from go2nav.planning import GlobalPlanner, PLANNERS

    @PLANNERS.register("my_planner")
    class MyPlanner(GlobalPlanner):
        def __init__(self, my_param=1.0, **kw):
            super().__init__(**kw)          # smoothing / densify / yaw options
            self.my_param = my_param

        def _plan(self, start, goal, costmap):
            ...                             # return (N,2) xy or (N,3) xy-yaw waypoints, info dict
            return waypoints, {"expansions": n}

Select it with ``--planner my_planner`` (or ``planner.name`` in YAML); params come
from ``planner.params.my_planner``. Files outside the package can be loaded with
``--plugin path/to/file.py``.
"""

from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..core.registry import Registry
from ..core.types import Pose2D, path_length
from ..perception.occupancy import Costmap2D
from .postprocess import assign_yaw, densify, shortcut

PLANNERS = Registry("planner")


@dataclass
class PlanResult:
    success: bool
    path: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))  # (N,3) x, y, yaw
    raw_path: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    planning_time: float = 0.0
    expansions: int = 0
    message: str = ""
    info: Dict[str, Any] = field(default_factory=dict)

    @property
    def length(self) -> float:
        return path_length(self.path)


class GlobalPlanner(ABC):
    """Base class. Subclasses implement ``_plan``; ``plan`` adds timing and post-processing.

    Args:
        smooth: greedy line-of-sight shortcutting of the raw path (grid / sampling planners)
        densify_step: resample the final path every ``densify_step`` metres (0 = off)
        yaw_mode: how yaw is assigned to xy-only paths: "tangent" (face the direction of
            travel, last pose = goal yaw) or "goal" (hold goal yaw everywhere - strafing)
        time_limit: soft wall-clock budget in seconds (planners should respect it)
        goal_search_radius: if the goal is inside an inflated obstacle, plan to the nearest
            free cell within this radius instead of failing
    """

    #: True for planners that already return feasible (x, y, yaw) sequences (e.g. Hybrid A*)
    outputs_yaw: bool = False
    registry_name: str = "?"

    def __init__(self, smooth: bool = True, densify_step: float = 0.1, yaw_mode: str = "tangent",
                 time_limit: float = 5.0, goal_search_radius: float = 1.0):
        self.smooth = smooth
        self.densify_step = densify_step
        self.yaw_mode = yaw_mode
        self.time_limit = time_limit
        self.goal_search_radius = goal_search_radius

    @property
    def name(self) -> str:
        return self.registry_name

    # ------------------------------------------------------------------ api
    def plan(self, start: Pose2D, goal: Pose2D, costmap: Costmap2D) -> PlanResult:
        t0 = time.perf_counter()
        self._deadline = t0 + self.time_limit
        goal_adj = self.project_goal(goal, costmap)
        if goal_adj is None:
            return PlanResult(False, message="goal is in collision and no free cell nearby",
                              planning_time=time.perf_counter() - t0)
        try:
            raw, info = self._plan(start, goal_adj, costmap)
        except TimeoutError:
            raw, info = None, {"message": "timeout"}
        dt = time.perf_counter() - t0
        if raw is None or len(raw) == 0:
            return PlanResult(False, planning_time=dt, expansions=info.get("expansions", 0),
                              message=info.get("message", "no path found"), info=info)
        raw = np.asarray(raw, dtype=float)
        path = self.postprocess(raw, start, goal_adj, costmap)
        return PlanResult(True, path=path, raw_path=raw, planning_time=time.perf_counter() - t0,
                          expansions=info.get("expansions", 0), message=info.get("message", "ok"), info=info)

    @abstractmethod
    def _plan(self, start: Pose2D, goal: Pose2D, costmap: Costmap2D) -> Tuple[Optional[np.ndarray], Dict]:
        """Return ((N,2) or (N,3) waypoints from start to goal, info) or (None, info) on failure."""

    # ------------------------------------------------------------- helpers
    def check_time(self) -> None:
        if time.perf_counter() > self._deadline:
            raise TimeoutError

    def postprocess(self, raw: np.ndarray, start: Pose2D, goal: Pose2D, costmap: Costmap2D) -> np.ndarray:
        if self.outputs_yaw and raw.shape[1] >= 3:
            return densify(raw, self.densify_step) if self.densify_step else raw
        xy = raw[:, :2]
        if self.smooth:
            xy = shortcut(xy, costmap)
        if self.densify_step:
            xy = densify(xy, self.densify_step)
        return assign_yaw(xy, start.yaw, goal.yaw, mode=self.yaw_mode)

    def project_goal(self, goal: Pose2D, costmap: Costmap2D) -> Optional[Pose2D]:
        if costmap.is_free_xy(goal.xy[None])[0]:
            return goal
        xy = nearest_free(costmap, goal.xy, self.goal_search_radius)
        return None if xy is None else Pose2D(xy[0], xy[1], goal.yaw)


def nearest_free(costmap: Costmap2D, xy: np.ndarray, radius: float) -> Optional[np.ndarray]:
    ij = costmap.world_to_grid(np.asarray(xy)[None])[0]
    r = int(math.ceil(radius / costmap.resolution))
    h, w = costmap.shape
    y0, y1, x0, x1 = max(0, ij[0] - r), min(h, ij[0] + r + 1), max(0, ij[1] - r), min(w, ij[1] + r + 1)
    sub = ~costmap.lethal[y0:y1, x0:x1]
    if not sub.any():
        return None
    ys, xs = np.nonzero(sub)
    d2 = (ys + y0 - ij[0]) ** 2 + (xs + x0 - ij[1]) ** 2
    k = int(np.argmin(d2))
    if d2[k] > r * r:
        return None
    return costmap.grid_to_world(np.array([[ys[k] + y0, xs[k] + x0]]))[0]


def escape_mask(costmap: Costmap2D, start_ij: Tuple[int, int], hard_clearance: float) -> np.ndarray:
    """Cells the robot may traverse to leave the inflated zone when it starts inside it.

    Flood fill from the start through lethal cells that still keep ``hard_clearance``
    to real obstacles; returns a boolean mask of those cells.
    """
    h, w = costmap.shape
    allowed = costmap.lethal & (costmap.clearance >= hard_clearance)
    mask = np.zeros((h, w), dtype=bool)
    sy, sx = start_ij
    if not (0 <= sy < h and 0 <= sx < w) or not costmap.lethal[sy, sx]:
        return mask
    stack = [(sy, sx)]
    mask[sy, sx] = True
    while stack:
        y, x = stack.pop()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and not mask[ny, nx] and allowed[ny, nx]:
                mask[ny, nx] = True
                stack.append((ny, nx))
    return mask

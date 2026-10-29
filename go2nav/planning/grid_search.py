"""Graph search on the 8-connected costmap grid: A*, Dijkstra, Greedy best-first, Lazy Theta*."""

from __future__ import annotations

import heapq
import math
from typing import Dict, Optional, Tuple

import numpy as np

from ..core.types import Pose2D
from ..perception.occupancy import Costmap2D
from .base import PLANNERS, GlobalPlanner, escape_mask

SQRT2 = math.sqrt(2.0)
NEIGHBORS = ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
             (1, 1, SQRT2), (1, -1, SQRT2), (-1, 1, SQRT2), (-1, -1, SQRT2))


class GridSearchPlanner(GlobalPlanner):
    """Best-first search template on the costmap grid.

    Edge cost = length * (1 + cost_weight * soft_cost(cell)). Subclasses customise
    ``heuristic`` and ``priority`` (f = g + w * h for A*).

    Args:
        cost_weight: penalty multiplier for the soft obstacle cost (0 = shortest path)
        heuristic_weight: w in f = g + w*h (1 = optimal A*, >1 = weighted A*)
        heuristic: "euclidean" | "octile" | "manhattan" | "zero"
        allow_diagonal_corner_cut: allow diagonal moves between two blocked orthogonal cells
        hard_clearance: min clearance [m] for escaping when the start lies in the inflated zone
    """

    def __init__(self, cost_weight: float = 2.0, heuristic_weight: float = 1.0, heuristic: str = "octile",
                 allow_diagonal_corner_cut: bool = False, hard_clearance: float = 0.15, **kw):
        super().__init__(**kw)
        self.cost_weight = cost_weight
        self.heuristic_weight = heuristic_weight
        self.heuristic_name = heuristic
        self.corner_cut = allow_diagonal_corner_cut
        self.hard_clearance = hard_clearance

    # ---- hooks -------------------------------------------------------------
    def heuristic(self, iy: int, ix: int, gy: int, gx: int) -> float:
        dy, dx = abs(iy - gy), abs(ix - gx)
        if self.heuristic_name == "euclidean":
            return math.hypot(dx, dy)
        if self.heuristic_name == "manhattan":
            return dx + dy
        if self.heuristic_name == "zero":
            return 0.0
        return (dx + dy) + (SQRT2 - 2.0) * min(dx, dy)  # octile

    def priority(self, g: float, h: float) -> float:
        return g + self.heuristic_weight * h

    # ---- search ------------------------------------------------------------
    def _blocked(self, costmap: Costmap2D, start_ij) -> np.ndarray:
        blocked = costmap.lethal.copy()
        esc = escape_mask(costmap, start_ij, self.hard_clearance)
        blocked[esc] = False
        blocked[start_ij[0], start_ij[1]] = False
        return blocked

    def _plan(self, start: Pose2D, goal: Pose2D, costmap: Costmap2D):
        s = tuple(costmap.world_to_grid(start.xy[None])[0])
        g = tuple(costmap.world_to_grid(goal.xy[None])[0])
        h_, w_ = costmap.shape
        if not (0 <= s[0] < h_ and 0 <= s[1] < w_ and 0 <= g[0] < h_ and 0 <= g[1] < w_):
            return None, {"message": "start or goal outside the map"}
        blocked = self._blocked(costmap, s)
        parents, n_exp, found = self._search(s, g, blocked, costmap.cost)
        if not found:
            return None, {"message": "no path", "expansions": n_exp}
        cells = [g]
        while cells[-1] != s:
            cells.append(parents[cells[-1]])
        cells.reverse()
        xy = costmap.grid_to_world(np.array(cells))
        xy[0], xy[-1] = start.xy, goal.xy
        return xy, {"expansions": n_exp}

    def _search(self, s, g, blocked, soft) -> Tuple[Dict, int, bool]:
        h_, w_ = blocked.shape
        cw = self.cost_weight
        gscore = {s: 0.0}
        parents = {s: s}
        closed = set()
        heap = [(self.priority(0.0, self.heuristic(*s, *g)), 0.0, s)]
        n_exp = 0
        while heap:
            _, gc, cur = heapq.heappop(heap)
            if cur in closed:
                continue
            closed.add(cur)
            n_exp += 1
            if cur == g:
                return parents, n_exp, True
            if (n_exp & 1023) == 0:
                self.check_time()
            cy, cx = cur
            for dy, dx, step in NEIGHBORS:
                ny, nx = cy + dy, cx + dx
                if ny < 0 or nx < 0 or ny >= h_ or nx >= w_ or blocked[ny, nx]:
                    continue
                if dy and dx and not self.corner_cut and (blocked[cy, nx] or blocked[ny, cx]):
                    continue
                nb = (ny, nx)
                if nb in closed:
                    continue
                ng = gc + step * (1.0 + cw * soft[ny, nx])
                if ng < gscore.get(nb, math.inf):
                    gscore[nb] = ng
                    parents[nb] = cur
                    heapq.heappush(heap, (self.priority(ng, self.heuristic(ny, nx, *g)), ng, nb))
        return parents, n_exp, False


@PLANNERS.register("astar")
class AStarPlanner(GridSearchPlanner):
    """A* (optimal for heuristic_weight=1 with an admissible heuristic)."""


@PLANNERS.register("dijkstra")
class DijkstraPlanner(GridSearchPlanner):
    """Uniform-cost search (A* with h = 0)."""

    def __init__(self, **kw):
        kw["heuristic"] = "zero"
        super().__init__(**kw)


@PLANNERS.register("greedy_bfs")
class GreedyBestFirstPlanner(GridSearchPlanner):
    """Greedy best-first: f = h. Fast, not optimal. Minimal example of a bespoke variant."""

    def priority(self, g: float, h: float) -> float:
        return h


@PLANNERS.register("theta_star")
class LazyThetaStarPlanner(GridSearchPlanner):
    """Lazy Theta* (Nash et al. 2010): any-angle paths, parents may skip cells with line of sight.

    Soft costs are ignored for the any-angle edges; obstacles are respected via the
    inflated lethal layer.
    """

    def __init__(self, **kw):
        kw.setdefault("heuristic", "euclidean")
        kw.setdefault("smooth", False)
        super().__init__(**kw)

    @staticmethod
    def _los(blocked: np.ndarray, a, b) -> bool:
        (y0, x0), (y1, x1) = a, b
        n = max(abs(y1 - y0), abs(x1 - x0)) * 2 + 1
        ys = np.rint(np.linspace(y0, y1, n)).astype(int)
        xs = np.rint(np.linspace(x0, x1, n)).astype(int)
        return not blocked[ys, xs].any()

    def _search(self, s, g, blocked, soft):
        h_, w_ = blocked.shape
        gscore = {s: 0.0}
        parents = {s: s}
        closed = set()
        heap = [(self.heuristic(*s, *g), s)]
        n_exp = 0
        dist = lambda a, b: math.hypot(a[0] - b[0], a[1] - b[1])
        while heap:
            _, cur = heapq.heappop(heap)
            if cur in closed:
                continue
            n_exp += 1
            if (n_exp & 1023) == 0:
                self.check_time()
            # lazy line-of-sight repair
            par = parents[cur]
            if par != cur and not self._los(blocked, par, cur):
                best, best_p = math.inf, None
                cy, cx = cur
                for dy, dx, step in NEIGHBORS:
                    nb = (cy + dy, cx + dx)
                    if nb in closed and gscore[nb] + step < best:
                        best, best_p = gscore[nb] + step, nb
                if best_p is not None:
                    parents[cur], gscore[cur] = best_p, best
            closed.add(cur)
            if cur == g:
                return parents, n_exp, True
            cy, cx = cur
            par = parents[cur]
            for dy, dx, step in NEIGHBORS:
                ny, nx = cy + dy, cx + dx
                if ny < 0 or nx < 0 or ny >= h_ or nx >= w_ or blocked[ny, nx]:
                    continue
                if dy and dx and (blocked[cy, nx] or blocked[ny, cx]):
                    continue
                nb = (ny, nx)
                if nb in closed:
                    continue
                ng = gscore[par] + dist(par, nb)
                if ng < gscore.get(nb, math.inf):
                    gscore[nb] = ng
                    parents[nb] = par
                    heapq.heappush(heap, (ng + self.heuristic_weight * self.heuristic(ny, nx, *g), nb))
        return parents, n_exp, False

"""RRT and RRT* (Karaman & Frazzoli 2011) in continuous (x, y) on the inflated costmap."""

from __future__ import annotations

import math
import time

import numpy as np

from ..core.types import Pose2D
from ..perception.occupancy import Costmap2D
from .base import PLANNERS, GlobalPlanner, escape_mask


@PLANNERS.register("rrt_star")
class RRTStarPlanner(GlobalPlanner):
    """
    Args:
        max_iterations: sampling iterations
        step: max edge length (steer distance) [m]
        goal_bias: probability of sampling the goal
        goal_tolerance: connect to goal when within this distance [m]
        gamma: rewiring radius constant (r = min(gamma * sqrt(log n / n), rewire_max))
        rewire: False -> plain RRT
        stop_on_first: return the first solution (anytime behaviour otherwise)
        seed: RNG seed (None = random)
    """

    def __init__(self, max_iterations: int = 4000, step: float = 0.7, goal_bias: float = 0.1,
                 goal_tolerance: float = 0.3, gamma: float = 6.0, rewire_max: float = 2.0, rewire: bool = True,
                 stop_on_first: bool = False, seed=None, **kw):
        super().__init__(**kw)
        self.max_iterations = max_iterations
        self.step = step
        self.goal_bias = goal_bias
        self.goal_tolerance = goal_tolerance
        self.gamma = gamma
        self.rewire_max = rewire_max
        self.rewire = rewire
        self.stop_on_first = stop_on_first
        self.rng = np.random.default_rng(seed)

    def _plan(self, start: Pose2D, goal: Pose2D, costmap: Costmap2D):
        s_ij = tuple(costmap.world_to_grid(start.xy[None])[0])
        esc = escape_mask(costmap, s_ij, 0.15)
        free_cells = np.argwhere(~costmap.lethal)
        if len(free_cells) == 0:
            return None, {"message": "map has no free space"}

        def seg_free(a, b):
            if costmap.segment_free(a, b):
                return True
            if esc.any():  # allow edges that stay inside the start escape region
                n = max(2, int(np.linalg.norm(b - a) / (0.5 * costmap.resolution)) + 1)
                t = np.linspace(0, 1, n)[:, None]
                pts = (1 - t) * a + t * b
                ij = costmap.world_to_grid(pts)
                inb = costmap.grid.in_bounds(ij)
                if not inb.all():
                    return False
                return bool((~costmap.lethal[ij[:, 0], ij[:, 1]] | esc[ij[:, 0], ij[:, 1]]).all())
            return False

        N = self.max_iterations + 2
        P = np.zeros((N, 2))
        parent = np.full(N, -1, dtype=int)
        cost = np.zeros(N)
        P[0] = start.xy
        n = 1
        goal_idx = -1
        best_goal_cost = math.inf
        res = costmap.resolution
        for it in range(self.max_iterations):
            if (it & 63) == 0:
                self.check_time()
            if self.rng.random() < self.goal_bias:
                q = goal.xy.copy()
            else:
                c = free_cells[self.rng.integers(len(free_cells))]
                q = costmap.grid_to_world(c[None])[0] + self.rng.uniform(-0.5, 0.5, 2) * res
            d = np.linalg.norm(P[:n] - q, axis=1)
            near_i = int(np.argmin(d))
            if d[near_i] < 1e-6:
                continue
            new = P[near_i] + (q - P[near_i]) * min(1.0, self.step / d[near_i])
            if not seg_free(P[near_i], new):
                continue
            # choose parent
            best_p, best_c = near_i, cost[near_i] + np.linalg.norm(new - P[near_i])
            near = np.array([], dtype=int)
            if self.rewire:
                r = min(self.gamma * math.sqrt(math.log(n + 1) / (n + 1)), self.rewire_max)
                dn = np.linalg.norm(P[:n] - new, axis=1)
                near = np.nonzero(dn <= r)[0]
                for j in near[np.argsort(cost[near] + dn[near])]:
                    cj = cost[j] + dn[j]
                    if cj < best_c and seg_free(P[j], new):
                        best_p, best_c = j, cj
                        break
            P[n], parent[n], cost[n] = new, best_p, best_c
            new_i = n
            n += 1
            # rewire neighbours through the new node
            for j in near:
                cj = best_c + np.linalg.norm(P[j] - new)
                if cj + 1e-9 < cost[j] and seg_free(new, P[j]):
                    delta = cost[j] - cj
                    parent[j] = new_i
                    cost[j] = cj
                    # propagate cost decrease to descendants
                    stack = [j]
                    while stack:
                        u = stack.pop()
                        ch = np.nonzero(parent[:n] == u)[0]
                        cost[ch] -= delta
                        stack.extend(ch.tolist())
            # goal connection
            dg = np.linalg.norm(new - goal.xy)
            if dg <= self.goal_tolerance or (dg <= self.step and seg_free(new, goal.xy)):
                gc = best_c + dg
                if gc < best_goal_cost:
                    best_goal_cost, goal_idx = gc, new_i
                    if self.stop_on_first or not self.rewire:
                        break
        if goal_idx < 0:
            return None, {"message": "no path", "expansions": n}
        # re-evaluate best goal-connected node (costs may have dropped via rewiring)
        dg = np.linalg.norm(P[:n] - goal.xy, axis=1)
        cand = np.nonzero(dg <= max(self.goal_tolerance, 1e-9))[0]
        if len(cand):
            goal_idx = int(cand[np.argmin(cost[cand] + dg[cand])])
        chain = [goal.xy]
        i = goal_idx
        while i >= 0:
            chain.append(P[i])
            i = parent[i]
        return np.array(chain[::-1]), {"expansions": n, "tree_size": n, "cost": float(best_goal_cost)}


@PLANNERS.register("rrt")
class RRTPlanner(RRTStarPlanner):
    """Plain RRT (no rewiring, stops at first solution)."""

    def __init__(self, **kw):
        kw["rewire"] = False
        super().__init__(**kw)

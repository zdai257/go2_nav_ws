"""Hybrid A* in SE(2) = (x, y, yaw) with quadruped motion primitives.

Primitives (body frame, length ``step``):
  * forward arcs with curvatures {0, +-1/r for r in turn_radii}
  * reverse arcs (optional, penalised)
  * in-place rotation +-rotate_step (legged robots can turn on the spot)
  * lateral strafe +-step/2 (optional, penalised)

Collision checking uses the yaw-dependent multi-circle footprint from the costmap.
Heuristic = max(Euclidean, obstacle-aware holonomic distance from a 2D wavefront).
An analytic "rotate - straight - rotate" shot to the goal is tried periodically.
"""

from __future__ import annotations

import heapq
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..core.types import Pose2D, wrap_angle
from ..perception.occupancy import Costmap2D
from .base import PLANNERS, GlobalPlanner

_SHIFTS = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
           (1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)), (-1, -1, math.sqrt(2))]


def holonomic_cost_to_go(free: np.ndarray, goal_ij: Tuple[int, int], res: float, max_iter: int = 5000,
                         cell_cost: Optional[np.ndarray] = None) -> np.ndarray:
    """Vectorised 8-connected wavefront (Bellman-Ford relaxation) from the goal.

    Edge cost = length * cell_cost[dst] (cell_cost = 1 -> metres).
    """
    h, w = free.shape
    D = np.full((h, w), np.inf)
    if 0 <= goal_ij[0] < h and 0 <= goal_ij[1] < w:
        D[goal_ij] = 0.0
    mult = np.ones((h, w)) if cell_cost is None else cell_cost
    for _ in range(max_iter):
        old = D
        D = D.copy()
        for dy, dx, c in _SHIFTS:
            sl_dst = (slice(max(0, dy), h - max(0, -dy)), slice(max(0, dx), w - max(0, -dx)))
            src = old[max(0, -dy): h - max(0, dy), max(0, -dx): w - max(0, dx)] + c * res * mult[sl_dst]
            dst = D[sl_dst]
            np.minimum(dst, src, out=dst)
        D[~free] = np.inf
        D[goal_ij] = 0.0
        if np.array_equal(D, old):
            break
    return D


@PLANNERS.register("hybrid_astar")
class HybridAStarPlanner(GlobalPlanner):
    outputs_yaw = True

    def __init__(self, xy_resolution: float = 0.2, yaw_bins: int = 36, step: float = 0.4,
                 turn_radii=(0.5, 1.0), allow_reverse: bool = True, reverse_penalty: float = 2.0,
                 allow_rotate_in_place: bool = True, rotate_step_deg: float = 30.0, rotate_cost: float = 0.3,
                 allow_strafe: bool = False, strafe_penalty: float = 2.5, direction_change_penalty: float = 0.5,
                 cost_weight: float = 1.0, heuristic_weight: float = 1.5,
                 goal_xy_tol: float = 0.15, goal_yaw_tol_deg: float = 15.0,
                 analytic_every: int = 8, analytic_max_dist: float = 8.0, max_expansions: int = 40000,
                 samples_per_step: int = 5, start_escape_radius: float = 0.6, **kw):
        kw.setdefault("smooth", False)
        super().__init__(**kw)
        self.xy_res = xy_resolution
        self.yaw_bins = yaw_bins
        self.step = step
        self.cost_weight = cost_weight
        self.heuristic_weight = heuristic_weight
        self.goal_xy_tol = goal_xy_tol
        self.goal_yaw_tol = math.radians(goal_yaw_tol_deg)
        self.analytic_every = analytic_every
        self.analytic_max_dist = analytic_max_dist
        self.max_expansions = max_expansions
        self.start_escape_radius = start_escape_radius
        self.direction_change_penalty = direction_change_penalty
        self.primitives = self._build_primitives(step, turn_radii, allow_reverse, reverse_penalty,
                                                 allow_rotate_in_place, math.radians(rotate_step_deg), rotate_cost,
                                                 allow_strafe, strafe_penalty, samples_per_step)
        self._rel = np.concatenate([p[0] for p in self.primitives], axis=0)  # (M*K, 3)
        self._K = samples_per_step

    @staticmethod
    def _build_primitives(step, radii, rev, rev_pen, rot, rot_step, rot_cost, strafe, strafe_pen, K):
        prims = []  # (samples (K,3), cost, direction tag)
        s = np.linspace(1.0 / K, 1.0, K)
        curvs = [0.0] + [c for r in radii for c in (1.0 / r, -1.0 / r)]
        for direction, pen in ((1, 1.0), (-1, rev_pen)) if rev else ((1, 1.0),):
            for k in curvs:
                L = direction * step * s
                if abs(k) < 1e-9:
                    pts = np.stack([L, np.zeros(K), np.zeros(K)], axis=1)
                else:
                    th = k * L
                    pts = np.stack([np.sin(th) / k, (1 - np.cos(th)) / k, th], axis=1)
                prims.append((pts, step * pen, direction))
        if rot:
            for sgn in (1, -1):
                pts = np.stack([np.zeros(K), np.zeros(K), sgn * rot_step * s], axis=1)
                prims.append((pts, rot_cost * rot_step, 0))
        if strafe:
            for sgn in (1, -1):
                pts = np.stack([np.zeros(K), sgn * 0.5 * step * s, np.zeros(K)], axis=1)
                prims.append((pts, 0.5 * step * strafe_pen, 2))
        return prims

    def _key(self, x, y, th):
        return (int(math.floor(x / self.xy_res)), int(math.floor(y / self.xy_res)),
                int(math.floor(wrap_angle(th) / (2 * math.pi) * self.yaw_bins)) % self.yaw_bins)

    def _valid(self, poses: np.ndarray, costmap: Costmap2D, start: Pose2D, start_blocked: bool) -> np.ndarray:
        ok = costmap.footprint_free(poses)
        if start_blocked:
            near = np.hypot(poses[:, 0] - start.x, poses[:, 1] - start.y) < self.start_escape_radius
            ok |= near & (costmap.clearance_at(poses[:, :2]) >= 0.5 * costmap.footprint_radius)
        return ok

    def _shot(self, p: np.ndarray, goal: Pose2D, costmap: Costmap2D) -> Optional[np.ndarray]:
        """rotate -> straight -> rotate connection; None if it collides."""
        dx, dy = goal.x - p[0], goal.y - p[1]
        dist = math.hypot(dx, dy)
        segs: List[np.ndarray] = []
        if dist > 1e-3:
            head = math.atan2(dy, dx)
            segs.append(self._rot_samples(p[0], p[1], p[2], head))
            n = max(2, int(dist / (0.5 * costmap.resolution)))
            t = np.linspace(0, 1, n)[1:]
            segs.append(np.stack([p[0] + t * dx, p[1] + t * dy, np.full(len(t), head)], axis=1))
            segs.append(self._rot_samples(goal.x, goal.y, head, goal.yaw))
        else:
            segs.append(self._rot_samples(p[0], p[1], p[2], goal.yaw))
        poses = np.concatenate([s for s in segs if len(s)], axis=0) if any(len(s) for s in segs) else np.zeros((0, 3))
        if len(poses) and not costmap.footprint_free(poses).all():
            return None
        return poses

    @staticmethod
    def _rot_samples(x, y, a, b, step=math.radians(10)):
        d = wrap_angle(b - a)
        n = int(abs(d) / step) + 1
        t = np.linspace(0, 1, n + 1)[1:]
        return np.stack([np.full(n, x), np.full(n, y), wrap_angle(a + t * d)], axis=1)

    def _plan(self, start: Pose2D, goal: Pose2D, costmap: Costmap2D):
        res = costmap.resolution
        free2d = costmap.clearance >= costmap.footprint_radius
        g_ij = tuple(costmap.world_to_grid(goal.xy[None])[0])
        start_blocked = not costmap.footprint_free(start.as_array()[None])[0]
        if start_blocked:  # let the wavefront reach the start region too
            s_ij = costmap.world_to_grid(start.xy[None])[0]
            r = int(self.start_escape_radius / res) + 1
            sl = (slice(max(0, s_ij[0] - r), s_ij[0] + r + 1), slice(max(0, s_ij[1] - r), s_ij[1] + r + 1))
            free2d = free2d.copy()
            free2d[sl] |= costmap.clearance[sl] >= 0.5 * costmap.footprint_radius
        H2 = holonomic_cost_to_go(free2d, g_ij, res, cell_cost=1.0 + self.cost_weight * costmap.cost)
        hw = self.heuristic_weight

        def h(x, y):
            iy = int(math.floor((y - costmap.grid.origin[1]) / res))
            ix = int(math.floor((x - costmap.grid.origin[0]) / res))
            if not (0 <= iy < H2.shape[0] and 0 <= ix < H2.shape[1]):
                return math.inf
            return hw * max(H2[iy, ix], math.hypot(goal.x - x, goal.y - y))

        nodes_pose: List[np.ndarray] = [start.as_array()]
        nodes_parent: List[int] = [-1]
        nodes_seg: List[np.ndarray] = [start.as_array()[None]]
        nodes_dir: List[int] = [1]
        nodes_g: List[float] = [0.0]
        best_g: Dict[Tuple, float] = {self._key(*start.as_array()): 0.0}
        h0 = h(start.x, start.y)
        if not math.isfinite(h0):
            return None, {"message": "goal unreachable (holonomic heuristic)"}
        heap = [(h0, 0)]
        closed = set()
        n_exp = 0
        prims = self.primitives
        M, K = len(prims), self._K
        rel = self._rel
        while heap and n_exp < self.max_expansions:
            _, idx = heapq.heappop(heap)
            p = nodes_pose[idx]
            key = self._key(*p)
            if key in closed:
                continue
            closed.add(key)
            n_exp += 1
            if (n_exp & 255) == 0:
                self.check_time()
            d_goal = math.hypot(goal.x - p[0], goal.y - p[1])
            if d_goal <= self.goal_xy_tol and abs(wrap_angle(goal.yaw - p[2])) <= self.goal_yaw_tol:
                return self._extract(idx, nodes_parent, nodes_seg, None), {"expansions": n_exp}
            if d_goal <= self.analytic_max_dist and (n_exp % self.analytic_every == 0 or d_goal < 2 * self.step):
                shot = self._shot(p, goal, costmap)
                if shot is not None:
                    return self._extract(idx, nodes_parent, nodes_seg, shot), {"expansions": n_exp, "analytic": True}
            # expand all primitives at once
            c, s = math.cos(p[2]), math.sin(p[2])
            wx = p[0] + c * rel[:, 0] - s * rel[:, 1]
            wy = p[1] + s * rel[:, 0] + c * rel[:, 1]
            wt = wrap_angle(p[2] + rel[:, 2])
            world = np.stack([wx, wy, wt], axis=1)
            ok = self._valid(world, costmap, start, start_blocked).reshape(M, K).all(axis=1)
            ends = world.reshape(M, K, 3)[:, -1]
            soft = costmap.cost.ravel()
            end_ij = costmap.world_to_grid(ends[:, :2])
            inb = costmap.grid.in_bounds(end_ij)
            for m in np.nonzero(ok & inb)[0]:
                e = ends[m]
                k2 = self._key(*e)
                if k2 in closed:
                    continue
                pts, base_cost, direction = prims[m]
                cost = base_cost * (1.0 + self.cost_weight * soft[end_ij[m, 0] * costmap.shape[1] + end_ij[m, 1]])
                if direction in (1, -1) and nodes_dir[idx] in (1, -1) and direction != nodes_dir[idx]:
                    cost += self.direction_change_penalty
                ng = nodes_g[idx] + cost
                if ng >= best_g.get(k2, math.inf):
                    continue
                hh = h(e[0], e[1])
                if not math.isfinite(hh):
                    continue
                best_g[k2] = ng
                nodes_pose.append(e)
                nodes_parent.append(idx)
                nodes_seg.append(world.reshape(M, K, 3)[m])
                nodes_dir.append(direction if direction in (1, -1) else nodes_dir[idx])
                nodes_g.append(ng)
                heapq.heappush(heap, (ng + hh, len(nodes_pose) - 1))
        return None, {"message": "no path (expansion limit)" if n_exp >= self.max_expansions else "no path",
                      "expansions": n_exp}

    @staticmethod
    def _extract(idx, parents, segs, tail):
        chain = []
        while idx >= 0:
            chain.append(segs[idx])
            idx = parents[idx]
        chain.reverse()
        if tail is not None and len(tail):
            chain.append(tail)
        return np.concatenate(chain, axis=0)

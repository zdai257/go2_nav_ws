"""Occupancy grid + costmap used by every planner.

Conventions: grid index ``[iy, ix]`` (row = y), cell centres at
``origin + (i + 0.5) * resolution``. Occupancy values are probabilities in [0, 1]
with NaN = unknown.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np


def disk_offsets(radius_cells: float) -> np.ndarray:
    r = int(math.ceil(radius_cells))
    ys, xs = np.mgrid[-r : r + 1, -r : r + 1]
    m = xs**2 + ys**2 <= radius_cells**2 + 1e-9
    return np.stack([ys[m], xs[m]], axis=1)


def dilate(mask: np.ndarray, radius_cells: float) -> np.ndarray:
    """Binary dilation with a disk (pure numpy, no scipy)."""
    if radius_cells <= 0:
        return mask.copy()
    out = np.zeros_like(mask)
    h, w = mask.shape
    for dy, dx in disk_offsets(radius_cells):
        ys0, ys1 = max(0, dy), min(h, h + dy)
        xs0, xs1 = max(0, dx), min(w, w + dx)
        out[ys0:ys1, xs0:xs1] |= mask[ys0 - dy : ys1 - dy, xs0 - dx : xs1 - dx]
    return out


def distance_field(mask: np.ndarray, max_cells: int) -> np.ndarray:
    """Approximate Euclidean distance (in cells) to the nearest True cell, capped at ``max_cells``.

    Brushfire with alternating 4/8 connectivity (octagonal metric, <8% error).
    """
    dist = np.full(mask.shape, float(max_cells))
    dist[mask] = 0.0
    front = mask.copy()
    for k in range(1, max_cells + 1):
        grown = front.copy()
        grown[1:, :] |= front[:-1, :]
        grown[:-1, :] |= front[1:, :]
        grown[:, 1:] |= front[:, :-1]
        grown[:, :-1] |= front[:, 1:]
        if k % 2 == 0:
            grown[1:, 1:] |= front[:-1, :-1]
            grown[:-1, :-1] |= front[1:, 1:]
            grown[1:, :-1] |= front[:-1, 1:]
            grown[:-1, 1:] |= front[1:, :-1]
        new = grown & ~front
        dist[new] = k * 0.92  # scale octagonal steps toward Euclidean
        front = grown
        if front.all():
            break
    return dist


@dataclass
class OccupancyGrid:
    resolution: float
    origin: Tuple[float, float]
    data: np.ndarray  # (H, W) float occupancy probability, NaN = unknown

    @classmethod
    def empty(cls, bounds, resolution: float, unknown: bool = False) -> "OccupancyGrid":
        w = int(math.ceil((bounds[1] - bounds[0]) / resolution))
        h = int(math.ceil((bounds[3] - bounds[2]) / resolution))
        data = np.full((h, w), np.nan if unknown else 0.0)
        return cls(resolution, (bounds[0], bounds[2]), data)

    @classmethod
    def from_world(cls, world, resolution: float = 0.05, min_height: float = 0.0) -> "OccupancyGrid":
        """Ground-truth rasterization of an ``ObstacleWorld``."""
        g = cls.empty(world.bounds, resolution)
        xs, ys = g.cell_centers()
        pts = np.stack([xs.ravel(), ys.ravel()], axis=1)
        obs = [o for o in world.obstacles if o.height >= min_height]
        if obs:
            sdf = np.min(np.stack([o.sdf(pts) for o in obs]), axis=0)
            g.data = (sdf <= 0.5 * resolution).reshape(g.shape).astype(float)
        return g

    @property
    def shape(self) -> Tuple[int, int]:
        return self.data.shape

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        h, w = self.shape
        return (self.origin[0], self.origin[0] + w * self.resolution, self.origin[1], self.origin[1] + h * self.resolution)

    def cell_centers(self):
        h, w = self.shape
        xs = self.origin[0] + (np.arange(w) + 0.5) * self.resolution
        ys = self.origin[1] + (np.arange(h) + 0.5) * self.resolution
        return np.meshgrid(xs, ys)

    def world_to_grid(self, xy: np.ndarray) -> np.ndarray:
        """(N,2) world xy -> (N,2) integer [iy, ix]."""
        xy = np.atleast_2d(xy)
        ix = np.floor((xy[:, 0] - self.origin[0]) / self.resolution).astype(int)
        iy = np.floor((xy[:, 1] - self.origin[1]) / self.resolution).astype(int)
        return np.stack([iy, ix], axis=1)

    def grid_to_world(self, ij: np.ndarray) -> np.ndarray:
        ij = np.atleast_2d(ij)
        x = self.origin[0] + (ij[:, 1] + 0.5) * self.resolution
        y = self.origin[1] + (ij[:, 0] + 0.5) * self.resolution
        return np.stack([x, y], axis=1)

    def in_bounds(self, ij: np.ndarray) -> np.ndarray:
        ij = np.atleast_2d(ij)
        h, w = self.shape
        return (ij[:, 0] >= 0) & (ij[:, 0] < h) & (ij[:, 1] >= 0) & (ij[:, 1] < w)

    def occupied_mask(self, threshold: float = 0.65, unknown_as_occupied: bool = False) -> np.ndarray:
        d = self.data
        occ = np.nan_to_num(d, nan=1.0 if unknown_as_occupied else 0.0) >= threshold
        return occ

    def known_mask(self) -> np.ndarray:
        return ~np.isnan(self.data)


@dataclass
class Costmap2D:
    """Planner-facing costmap derived from an occupancy grid.

    * ``lethal``: cells whose centre is within ``inflation_radius`` of an obstacle
      (robot treated as a disc -> used by grid / sampling planners)
    * ``cost``: soft cost in [0, 1] decaying from 1 at ``inflation_radius`` to 0 at
      ``inflation_radius + cost_falloff`` (keeps paths away from obstacles)
    * ``footprint_free``: yaw-dependent check with a multi-circle Go2 footprint
      (used by SE(2) planners and path validation)
    """

    grid: OccupancyGrid
    inflation_radius: float = 0.30
    cost_falloff: float = 0.6
    footprint_circles: Sequence[Tuple[float, float]] = ((0.22, 0.0), (0.0, 0.0), (-0.22, 0.0))
    footprint_radius: float = 0.18
    occupied_threshold: float = 0.65
    unknown_as_occupied: bool = False
    obstacle: np.ndarray = field(init=False, repr=False)
    lethal: np.ndarray = field(init=False, repr=False)
    cost: np.ndarray = field(init=False, repr=False)
    clearance: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        res = self.grid.resolution
        self.obstacle = self.grid.occupied_mask(self.occupied_threshold, self.unknown_as_occupied)
        max_cells = int(math.ceil((self.inflation_radius + self.cost_falloff + max(self.footprint_radius, 0)) / res)) + 1
        self.clearance = distance_field(self.obstacle, max_cells) * res  # metres to nearest obstacle
        self.lethal = self.clearance < self.inflation_radius
        if self.cost_falloff > 0:
            self.cost = np.clip(1.0 - (self.clearance - self.inflation_radius) / self.cost_falloff, 0.0, 1.0)
        else:
            self.cost = np.zeros_like(self.clearance)
        self.cost[self.lethal] = 1.0
        self._fp = np.asarray(self.footprint_circles, dtype=float)

    # ------------------------------------------------------------- queries
    @property
    def resolution(self) -> float:
        return self.grid.resolution

    @property
    def shape(self):
        return self.grid.shape

    def world_to_grid(self, xy):
        return self.grid.world_to_grid(xy)

    def grid_to_world(self, ij):
        return self.grid.grid_to_world(ij)

    def is_free_xy(self, xy: np.ndarray) -> np.ndarray:
        """Disc-robot check (centre outside the inflated region)."""
        ij = self.world_to_grid(xy)
        ok = self.grid.in_bounds(ij)
        out = np.zeros(len(ij), dtype=bool)
        out[ok] = ~self.lethal[ij[ok, 0], ij[ok, 1]]
        return out

    def _clearance_xy(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        res, (ox, oy) = self.grid.resolution, self.grid.origin
        h, w = self.clearance.shape
        ix = np.floor((x - ox) / res).astype(np.int64)
        iy = np.floor((y - oy) / res).astype(np.int64)
        ok = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
        out = np.zeros(x.shape)
        out[ok] = self.clearance[iy[ok], ix[ok]]
        return out

    def clearance_at(self, xy: np.ndarray) -> np.ndarray:
        xy = np.atleast_2d(xy)
        return self._clearance_xy(xy[:, 0], xy[:, 1])

    def footprint_free(self, poses: np.ndarray) -> np.ndarray:
        """(N,3) poses -> bool[N]: every footprint circle has clearance >= footprint_radius."""
        poses = np.atleast_2d(poses)
        c, s = np.cos(poses[:, 2]), np.sin(poses[:, 2])
        fp = self._fp
        px = poses[:, 0, None] + c[:, None] * fp[None, :, 0] - s[:, None] * fp[None, :, 1]
        py = poses[:, 1, None] + s[:, None] * fp[None, :, 0] + c[:, None] * fp[None, :, 1]
        return (self._clearance_xy(px, py) >= self.footprint_radius).all(axis=1)

    def segment_free(self, a: np.ndarray, b: np.ndarray, step: Optional[float] = None) -> bool:
        """Disc-robot line-of-sight check between two xy points."""
        step = step or 0.5 * self.resolution
        n = max(2, int(math.ceil(np.linalg.norm(np.asarray(b) - np.asarray(a)) / step)) + 1)
        t = np.linspace(0.0, 1.0, n)[:, None]
        pts = (1 - t) * np.asarray(a)[None, :2] + t * np.asarray(b)[None, :2]
        return bool(self.is_free_xy(pts).all())

    def path_free(self, path: np.ndarray, use_footprint: bool = False, step: Optional[float] = None) -> Tuple[bool, int]:
        """Check a (N,2|3) path; returns (free, index of first blocked waypoint or -1)."""
        path = np.asarray(path)
        if len(path) == 0:
            return True, -1
        step = step or self.resolution
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            n = max(2, int(math.ceil(np.linalg.norm(b[:2] - a[:2]) / step)) + 1)
            t = np.linspace(0.0, 1.0, n)[:, None]
            pts = (1 - t) * a[None, :] + t * b[None, :]
            if use_footprint and path.shape[1] >= 3:
                pts[:, 2] = a[2] + t[:, 0] * ((b[2] - a[2] + np.pi) % (2 * np.pi) - np.pi)
                free = self.footprint_free(pts)
            else:
                free = self.is_free_xy(pts[:, :2])
            if not free.all():
                return False, i
        return True, -1

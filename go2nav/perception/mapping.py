"""Map providers: where the navigator gets its costmap from.

* ``GroundTruthMap``  - rasterizes the known ObstacleWorld (oracle map)
* ``LidarMapper``     - log-odds occupancy grid built online from lidar scans
                        (start with an unknown world; replan as obstacles appear)

Both expose ``version`` (bumped on change) and ``costmap()``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional

import numpy as np

from ..core.registry import Registry
from .occupancy import Costmap2D, OccupancyGrid

MAP_PROVIDERS = Registry("map_provider")


class MapProvider(ABC):
    def __init__(self, bounds, resolution: float = 0.05, costmap_params: Optional[Dict] = None):
        self.bounds = tuple(bounds)
        self.resolution = resolution
        self.costmap_params = dict(costmap_params or {})
        self.version = 0
        self._costmap: Optional[Costmap2D] = None
        self._costmap_version = -1

    @abstractmethod
    def grid(self) -> OccupancyGrid: ...

    def costmap(self) -> Costmap2D:
        if self._costmap is None or self._costmap_version != self.version:
            self._costmap = Costmap2D(self.grid(), **self.costmap_params)
            self._costmap_version = self.version
        return self._costmap

    def update_scan(self, origin_xy: np.ndarray, hits_xy: np.ndarray, free_endpoints_xy: Optional[np.ndarray] = None) -> None:
        """Integrate one lidar scan (world frame). No-op for oracle maps."""


@MAP_PROVIDERS.register("ground_truth")
class GroundTruthMap(MapProvider):
    def __init__(self, world, min_height: float = 0.0, **kw):
        super().__init__(world.bounds, **kw)
        self.world = world
        self.min_height = min_height
        self._grid = None
        self._world_version = -1

    def grid(self) -> OccupancyGrid:
        if self._grid is None or self._world_version != self.world.version:
            self._grid = OccupancyGrid.from_world(self.world, self.resolution, self.min_height)
            self._world_version = self.world.version
        return self._grid

    def costmap(self) -> Costmap2D:
        if self._world_version != self.world.version:
            self.version += 1
        return super().costmap()


@MAP_PROVIDERS.register("lidar")
class LidarMapper(MapProvider):
    """Log-odds occupancy mapping with free-space carving along each beam.

    ``update_scan`` takes beam endpoints that hit something (``hits_xy``) and
    endpoints of beams that returned nothing within range (``free_endpoints_xy``).
    The costmap is rebuilt at most every ``rebuild_every`` scans that changed
    the occupied set.
    """

    def __init__(self, bounds, resolution: float = 0.05, costmap_params=None, l_occ: float = 0.85,
                 l_free: float = -0.4, l_min: float = -2.0, l_max: float = 3.5, unknown_as_free: bool = True):
        super().__init__(bounds, resolution, costmap_params)
        empty = OccupancyGrid.empty(self.bounds, resolution)
        self.shape = empty.shape
        self.origin = empty.origin
        self.logodds = np.zeros(self.shape, dtype=np.float32)
        self.observed = np.zeros(self.shape, dtype=bool)
        self.l_occ, self.l_free, self.l_min, self.l_max = l_occ, l_free, l_min, l_max
        self.unknown_as_free = unknown_as_free
        self._occ_snapshot = np.zeros(self.shape, dtype=bool)

    def _to_ij(self, xy):
        ix = np.floor((xy[:, 0] - self.origin[0]) / self.resolution).astype(int)
        iy = np.floor((xy[:, 1] - self.origin[1]) / self.resolution).astype(int)
        ok = (iy >= 0) & (iy < self.shape[0]) & (ix >= 0) & (ix < self.shape[1])
        return iy[ok], ix[ok]

    def update_scan(self, origin_xy, hits_xy, free_endpoints_xy=None) -> None:
        origin_xy = np.asarray(origin_xy, dtype=float)[:2]
        if not np.all(np.isfinite(origin_xy)):
            return
        max_beam = float(np.hypot(*self.shape) * self.resolution)

        def sane(pts):
            pts = np.asarray(pts, dtype=float).reshape(-1, 2)
            ok = np.all(np.isfinite(pts), axis=1) & (np.linalg.norm(pts - origin_xy, axis=1) <= max_beam)
            return pts[ok]

        hits = sane(hits_xy)
        ends = [hits]
        if free_endpoints_xy is not None and len(free_endpoints_xy):
            ends.append(sane(free_endpoints_xy))
        ends = np.concatenate(ends, axis=0)
        if len(ends):
            # carve free space: sample each beam up to (but excluding) the last cell
            d = ends - origin_xy
            L = np.linalg.norm(d, axis=1)
            n = int(np.ceil(L.max() / (0.5 * self.resolution))) if len(L) else 0
            if n > 1:
                t = np.linspace(0.0, 1.0, n, endpoint=False)[None, :, None]
                shrink = np.clip((L - self.resolution) / np.maximum(L, 1e-6), 0.0, 1.0)[:, None, None]
                pts = (origin_xy[None, None, :] + t * shrink * d[:, None, :]).reshape(-1, 2)
                iy, ix = self._to_ij(pts)
                flat = np.unique(iy * self.shape[1] + ix)
                fy, fx = np.divmod(flat, self.shape[1])
                self.logodds[fy, fx] += self.l_free
                self.observed[fy, fx] = True
        if len(hits):
            iy, ix = self._to_ij(hits)
            flat = np.unique(iy * self.shape[1] + ix)
            hy, hx = np.divmod(flat, self.shape[1])
            self.logodds[hy, hx] += self.l_occ
            self.observed[hy, hx] = True
        np.clip(self.logodds, self.l_min, self.l_max, out=self.logodds)
        occ = self.logodds > 0.6
        if not np.array_equal(occ, self._occ_snapshot):
            self._occ_snapshot = occ
            self.version += 1

    def grid(self) -> OccupancyGrid:
        p = 1.0 / (1.0 + np.exp(-self.logodds.astype(float)))
        if not self.unknown_as_free:
            p[~self.observed] = np.nan
        return OccupancyGrid(self.resolution, self.origin, p)


def build_map_provider(cfg: Dict, world, bounds=None) -> MapProvider:
    """cfg = the ``map`` config section."""
    source = cfg.get("source", "ground_truth")
    common = dict(resolution=cfg.get("resolution", 0.05), costmap_params=cfg.get("costmap", {}))
    if source == "ground_truth":
        return GroundTruthMap(world, min_height=cfg.get("min_obstacle_height", 0.0), **common)
    if source == "lidar":
        return LidarMapper(bounds or world.bounds, unknown_as_free=cfg.get("unknown_as_free", True),
                           **common, **cfg.get("lidar_mapper", {}))
    return MAP_PROVIDERS.build(source, world=world, **common)

"""Obstacle primitives and a seeded random obstacle-field generator.

All geometry is 2.5D: a planar footprint (box or circle) extruded to ``height``.
The same ``ObstacleWorld`` drives the 2D simulator, ground-truth costmaps, and the
static colliders spawned in Isaac Sim, so a scene is fully reproducible from
(seed, generator params) or from a saved YAML file.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

# kind -> footprint shape. Add new kinds here (e.g. "barrel": "circle").
KIND_SHAPE = {"pillar": "box", "cube": "box", "box": "box", "wall": "box", "cylinder": "circle", "column": "circle"}

KIND_COLOR = {
    "pillar": (0.55, 0.55, 0.60),
    "cube": (0.80, 0.45, 0.20),
    "box": (0.70, 0.55, 0.35),
    "wall": (0.60, 0.60, 0.55),
    "cylinder": (0.20, 0.50, 0.80),
    "column": (0.35, 0.65, 0.45),
}


@dataclass
class Obstacle:
    """A static obstacle. ``size`` = (sx, sy) full extents for boxes, (radius,) for circles."""

    kind: str
    x: float
    y: float
    yaw: float = 0.0
    size: Tuple[float, ...] = (0.3, 0.3)
    height: float = 1.0
    name: str = ""

    @property
    def shape(self) -> str:
        return KIND_SHAPE.get(self.kind, "box")

    @property
    def radius(self) -> float:
        """Circumscribed radius of the footprint."""
        if self.shape == "circle":
            return float(self.size[0])
        return 0.5 * math.hypot(self.size[0], self.size[1])

    @property
    def footprint_area(self) -> float:
        if self.shape == "circle":
            return math.pi * self.size[0] ** 2
        return float(self.size[0] * self.size[1])

    @property
    def color(self) -> Tuple[float, float, float]:
        return KIND_COLOR.get(self.kind, (0.6, 0.6, 0.6))

    def sdf(self, pts: np.ndarray) -> np.ndarray:
        """Signed distance from (N,2) world points to the footprint (negative inside)."""
        pts = np.atleast_2d(pts)
        d = pts - np.array([self.x, self.y])
        if self.shape == "circle":
            return np.linalg.norm(d, axis=1) - self.size[0]
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        lx = np.abs(c * d[:, 0] + s * d[:, 1]) - 0.5 * self.size[0]
        ly = np.abs(-s * d[:, 0] + c * d[:, 1]) - 0.5 * self.size[1]
        outside = np.linalg.norm(np.stack([np.maximum(lx, 0), np.maximum(ly, 0)], axis=1), axis=1)
        inside = np.minimum(np.maximum(lx, ly), 0.0)
        return outside + inside

    def raycast(self, origin: np.ndarray, dirs: np.ndarray, max_range: float) -> np.ndarray:
        """Distance along unit ``dirs`` (N,2) from ``origin`` to the footprint boundary (inf if no hit)."""
        o = np.asarray(origin, dtype=float) - np.array([self.x, self.y])
        out = np.full(len(dirs), np.inf)
        if self.shape == "circle":
            b = dirs @ o
            c = o @ o - self.size[0] ** 2
            disc = b * b - c
            ok = disc >= 0
            t = -b[ok] - np.sqrt(disc[ok])
            t = np.where(t < 0, -b[ok] + np.sqrt(disc[ok]), t)
            tt = np.where(t >= 0, t, np.inf)
            out[ok] = tt
        else:
            c, s = math.cos(self.yaw), math.sin(self.yaw)
            ol = np.array([c * o[0] + s * o[1], -s * o[0] + c * o[1]])
            dl = np.stack([c * dirs[:, 0] + s * dirs[:, 1], -s * dirs[:, 0] + c * dirs[:, 1]], axis=1)
            half = 0.5 * np.array(self.size[:2])
            with np.errstate(divide="ignore", invalid="ignore"):
                t1 = (-half - ol) / dl
                t2 = (half - ol) / dl
            tmin = np.nanmax(np.minimum(t1, t2), axis=1)
            tmax = np.nanmin(np.maximum(t1, t2), axis=1)
            hit = (tmax >= np.maximum(tmin, 0.0))
            out[hit] = np.where(tmin[hit] >= 0, tmin[hit], tmax[hit])
        out[out > max_range] = np.inf
        return out

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["size"] = [float(v) for v in self.size]
        return d


@dataclass
class ObstacleWorld:
    """A rectangular arena with static obstacles. ``bounds`` = (xmin, xmax, ymin, ymax)."""

    bounds: Tuple[float, float, float, float] = (-10.0, 10.0, -10.0, 10.0)
    obstacles: List[Obstacle] = field(default_factory=list)
    version: int = 0  # bumped on every change so consumers know when to re-rasterize

    def add(self, ob: Obstacle) -> Obstacle:
        if not ob.name:
            ob.name = f"{ob.kind}_{len(self.obstacles):03d}"
        self.obstacles.append(ob)
        self.version += 1
        return ob

    def extend(self, obs: Sequence[Obstacle]) -> None:
        for ob in obs:
            self.add(ob)

    def clear(self) -> None:
        self.obstacles.clear()
        self.version += 1

    def sdf(self, pts: np.ndarray) -> np.ndarray:
        pts = np.atleast_2d(pts)
        if not self.obstacles:
            return np.full(len(pts), np.inf)
        return np.min(np.stack([ob.sdf(pts) for ob in self.obstacles]), axis=0)

    def collides(self, x: float, y: float, radius: float) -> bool:
        return bool(self.sdf(np.array([[x, y]]))[0] < radius)

    def raycast(self, origin: np.ndarray, angles: np.ndarray, max_range: float, min_height: float = 0.0) -> np.ndarray:
        dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        best = np.full(len(angles), np.inf)
        for ob in self.obstacles:
            if ob.height < min_height:
                continue
            if np.linalg.norm(np.asarray(origin) - [ob.x, ob.y]) - ob.radius > max_range:
                continue
            best = np.minimum(best, ob.raycast(origin, dirs, max_range))
        return best

    @property
    def area(self) -> float:
        b = self.bounds
        return (b[1] - b[0]) * (b[3] - b[2])

    def coverage(self) -> float:
        return sum(ob.footprint_area for ob in self.obstacles) / self.area

    # ---------------------------------------------------------------- io
    def to_dict(self) -> Dict:
        return {"bounds": [float(v) for v in self.bounds], "obstacles": [ob.to_dict() for ob in self.obstacles]}

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    @classmethod
    def from_dict(cls, d: Dict) -> "ObstacleWorld":
        w = cls(bounds=tuple(d.get("bounds", (-10, 10, -10, 10))))
        for o in d.get("obstacles", []):
            w.add(Obstacle(kind=o["kind"], x=o["x"], y=o["y"], yaw=o.get("yaw", 0.0),
                           size=tuple(o["size"]), height=o.get("height", 1.0), name=o.get("name", "")))
        return w

    @classmethod
    def load(cls, path: str) -> "ObstacleWorld":
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f))


# --------------------------------------------------------------------------- generator
DEFAULT_KIND_PARAMS: Dict[str, Dict] = {
    # tall slender columns (square section)
    "pillar": {"side": [0.2, 0.4], "height": [2.0, 3.0]},
    # cubes resting on the ground (edge length == height)
    "cube": {"edge": [0.3, 0.9]},
    # round obstacles of varying radius / height
    "cylinder": {"radius": [0.15, 0.5], "height": [0.5, 1.5]},
    # rectangular boxes / walls
    "box": {"sx": [0.4, 1.2], "sy": [0.3, 0.8], "height": [0.4, 1.2]},
    "wall": {"length": [1.5, 4.0], "thickness": [0.15, 0.25], "height": [1.0, 2.0]},
}


@dataclass
class ObstacleGenerator:
    """Seeded random obstacle placement.

    Args:
        bounds: arena (xmin, xmax, ymin, ymax) [m]
        mix: relative sampling weight per kind, e.g. {"pillar": 1, "cube": 1, "cylinder": 1}
        kind_params: per-kind size ranges (defaults in ``DEFAULT_KIND_PARAMS``)
        min_gap: minimum free gap between obstacle bounding circles [m]
        border_margin: keep obstacles this far inside the arena [m]
        seed: RNG seed (None -> random)
    """

    bounds: Tuple[float, float, float, float] = (-10.0, 10.0, -10.0, 10.0)
    mix: Dict[str, float] = field(default_factory=lambda: {"pillar": 1.0, "cube": 1.0, "cylinder": 1.0})
    kind_params: Dict[str, Dict] = field(default_factory=dict)
    min_gap: float = 0.6
    border_margin: float = 0.5
    seed: Optional[int] = 0

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        params = {k: dict(v) for k, v in DEFAULT_KIND_PARAMS.items()}
        for k, v in (self.kind_params or {}).items():
            params.setdefault(k, {}).update(v)
        self.params = params

    def _u(self, rng_pair) -> float:
        lo, hi = rng_pair
        return float(self.rng.uniform(lo, hi))

    def sample_obstacle(self, kind: Optional[str] = None, xy: Optional[Tuple[float, float]] = None) -> Obstacle:
        if kind is None:
            kinds = [k for k, w in self.mix.items() if w > 0]
            w = np.array([self.mix[k] for k in kinds], dtype=float)
            kind = str(self.rng.choice(kinds, p=w / w.sum()))
        p = self.params[kind]
        b, m = self.bounds, self.border_margin
        x, y = xy if xy is not None else (self._u((b[0] + m, b[1] - m)), self._u((b[2] + m, b[3] - m)))
        yaw = self._u((-math.pi, math.pi))
        if kind == "pillar":
            s = self._u(p["side"])
            return Obstacle(kind, x, y, yaw, (s, s), self._u(p["height"]))
        if kind == "cube":
            e = self._u(p["edge"])
            return Obstacle(kind, x, y, yaw, (e, e), e)
        if kind in ("cylinder", "column"):
            return Obstacle(kind, x, y, 0.0, (self._u(p["radius"]),), self._u(p["height"]))
        if kind == "wall":
            return Obstacle(kind, x, y, yaw, (self._u(p["length"]), self._u(p["thickness"])), self._u(p["height"]))
        return Obstacle(kind, x, y, yaw, (self._u(p["sx"]), self._u(p["sy"])), self._u(p["height"]))

    def _fits(self, ob: Obstacle, placed: List[Obstacle], keepouts: Sequence[Tuple[float, float, float]]) -> bool:
        for kx, ky, kr in keepouts:
            if math.hypot(ob.x - kx, ob.y - ky) < kr + ob.radius:
                return False
        for o in placed:
            if math.hypot(ob.x - o.x, ob.y - o.y) < ob.radius + o.radius + self.min_gap:
                return False
        return True

    def generate(
        self,
        count: Optional[int] = None,
        density: Optional[float] = None,
        keepouts: Sequence[Tuple[float, float, float]] = (),
        existing: Sequence[Obstacle] = (),
        max_attempts: int = 50000,
    ) -> List[Obstacle]:
        """Place obstacles until ``count`` is reached or footprint ``density`` (0-1 of arena area) is covered.

        ``keepouts`` = [(x, y, radius)] discs kept free (start / goal / robot).
        """
        if count is None and density is None:
            count = 20
        b = self.bounds
        area = (b[1] - b[0]) * (b[3] - b[2])
        placed = list(existing)
        new: List[Obstacle] = []
        covered = 0.0
        for _ in range(max_attempts):
            if count is not None and len(new) >= count:
                break
            if density is not None and covered / area >= density:
                break
            ob = self.sample_obstacle()
            if self._fits(ob, placed, keepouts):
                placed.append(ob)
                new.append(ob)
                covered += ob.footprint_area
        return new


def build_world(cfg: Dict, start=None, goal=None) -> ObstacleWorld:
    """Construct an ObstacleWorld from the ``world`` config section.

    world:
      bounds: [-10, 10, -10, 10]
      file: null                 # load a saved layout instead of sampling
      generator: {count: 30, density: null, seed: 0, mix: {...}, min_gap: 0.6, kind_params: {...}}
      keepout_radius: 1.0        # free disc around start and goal
      extra: []                  # hand-placed obstacles (dicts like in a saved file)
    """
    from ..core.config import resolve_path

    if cfg.get("file"):
        world = ObstacleWorld.load(resolve_path(cfg["file"]))
    else:
        world = ObstacleWorld(bounds=tuple(cfg.get("bounds", (-10, 10, -10, 10))))
        g = dict(cfg.get("generator", {}))
        count, density = g.pop("count", None), g.pop("density", None)
        if density is not None:
            count = None
        gen = ObstacleGenerator(bounds=world.bounds, **g)
        r = float(cfg.get("keepout_radius", 1.0))
        keep = [(p.x, p.y, r) for p in (start, goal) if p is not None]
        world.extend(gen.generate(count=count, density=density, keepouts=keep))
        if density is not None and world.coverage() < 0.9 * density:
            print(f"[go2nav] requested density {density:.2f} but only {world.coverage():.3f} fits with "
                  f"min_gap={gen.min_gap}; lower world.generator.min_gap or use smaller kind sizes")
    for o in cfg.get("extra", []) or []:
        world.add(Obstacle(kind=o["kind"], x=o["x"], y=o["y"], yaw=o.get("yaw", 0.0),
                           size=tuple(o["size"]), height=o.get("height", 1.0)))
    return world

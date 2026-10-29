"""Build a full navigation scenario (world, start/goal, map, planner, follower, navigator) from config.

Shared by the 2D simulator, the benchmark and Isaac Sim so a given config + seed
gives the identical obstacle layout everywhere.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .control import PathFollower, build_follower
from .core.types import Pose2D
from .navigation import Navigator
from .perception import MapProvider, build_map_provider
from .planning import GlobalPlanner, build_planner
from .world import Obstacle, ObstacleGenerator, ObstacleWorld, build_world


@dataclass
class Scenario:
    cfg: Dict
    world: ObstacleWorld
    start: Pose2D
    goal: Pose2D
    map: MapProvider
    planner: GlobalPlanner
    follower: PathFollower
    navigator: Navigator
    spawner: Optional["RuntimeSpawner"] = None


def wall_obstacles(bounds, thickness: float = 0.2, height: float = 1.0) -> List[Obstacle]:
    x0, x1, y0, y1 = bounds
    cx, cy, lx, ly = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
    return [
        Obstacle("wall", cx, y0, 0.0, (lx, thickness), height, "wall_s"),
        Obstacle("wall", cx, y1, 0.0, (lx, thickness), height, "wall_n"),
        Obstacle("wall", x0, cy, math.pi / 2, (ly, thickness), height, "wall_w"),
        Obstacle("wall", x1, cy, math.pi / 2, (ly, thickness), height, "wall_e"),
    ]


def _feasible(cfg: Dict, world: ObstacleWorld, start: Pose2D, goal: Pose2D) -> bool:
    from .perception import GroundTruthMap
    from .planning import AStarPlanner

    cm = GroundTruthMap(world, resolution=cfg["map"]["resolution"], costmap_params=cfg["map"]["costmap"]).costmap()
    return AStarPlanner(cost_weight=0.0, goal_search_radius=0.0).plan(start, goal, cm).success


def _clamp_into(p: Pose2D, bounds, margin: float) -> Pose2D:
    return Pose2D(float(np.clip(p.x, bounds[0] + margin, bounds[1] - margin)),
                  float(np.clip(p.y, bounds[2] + margin, bounds[3] - margin)), p.yaw)


def sample_start_goal(bounds, rng: np.random.Generator, min_dist: float, margin: float = 1.0):
    for _ in range(1000):
        a = rng.uniform([bounds[0] + margin, bounds[2] + margin], [bounds[1] - margin, bounds[3] - margin], size=(2, 2))
        if np.linalg.norm(a[0] - a[1]) >= min_dist:
            yaws = rng.uniform(-math.pi, math.pi, 2)
            return Pose2D(a[0, 0], a[0, 1], yaws[0]), Pose2D(a[1, 0], a[1, 1], yaws[1])
    raise RuntimeError("could not sample start/goal; reduce task.min_start_goal_dist")


def build_scenario(cfg: Dict) -> Scenario:
    wcfg = cfg["world"]
    bounds = tuple(wcfg.get("bounds", (-10, 10, -10, 10)))
    seed = wcfg.get("generator", {}).get("seed", 0)
    task = cfg.get("task", {})
    if task.get("random_start_goal"):
        rng = np.random.default_rng(None if seed is None else seed + 1000)
        start, goal = sample_start_goal(bounds, rng, task.get("min_start_goal_dist", 8.0))
    else:
        start, goal = Pose2D.from_seq(task["start"]), Pose2D.from_seq(task["goal"])
        start, goal = _clamp_into(start, bounds, 1.0), _clamp_into(goal, bounds, 1.0)

    world = None
    max_regen = int(wcfg.get("max_regenerate", 10))
    for attempt in range(max_regen + 1):
        world = build_world(wcfg, start, goal)
        if wcfg.get("walls", True) and not wcfg.get("file"):
            world.extend(wall_obstacles(world.bounds))
        if not wcfg.get("ensure_feasible", True) or wcfg.get("file") or seed is None or _feasible(cfg, world, start, goal):
            break
        if attempt == max_regen:
            print(f"[go2nav] WARNING: no feasible layout after {max_regen + 1} seeds - obstacles too dense for the "
                  f"robot (gaps < 2 x inflation_radius). Lower density or obstacle sizes, or raise min_gap.")
            break
        print(f"[go2nav] seed {wcfg['generator']['seed']} gives no feasible start->goal path; trying seed "
              f"{wcfg['generator']['seed'] + 1}")
        wcfg = copy.deepcopy(wcfg)
        wcfg["generator"]["seed"] += 1
    cfg["world"]["seed_used"] = wcfg.get("generator", {}).get("seed")

    mp = build_map_provider(cfg["map"], world)
    planner = build_planner(cfg["planner"])
    follower = build_follower(cfg["follower"])
    nav = Navigator(planner, follower, mp, cfg.get("navigator", {}))
    spawner = None
    rs = wcfg.get("runtime_spawn", {}) or {}
    if rs.get("count", 0) > 0:
        g = dict(wcfg.get("generator", {}))
        g.pop("count", None), g.pop("density", None)
        g["seed"] = None if seed is None else seed + 7
        spawner = RuntimeSpawner(ObstacleGenerator(bounds=world.bounds, **g), goal=goal, **rs)
    return Scenario(cfg, world, start, goal, mp, planner, follower, nav, spawner)


class RuntimeSpawner:
    """Drops new obstacles during a run - by default onto the current path ahead of the robot."""

    def __init__(self, generator: ObstacleGenerator, goal: Pose2D, count: int = 3, start_after_s: float = 5.0,
                 interval_s: float = 4.0, on_path: bool = True, ahead_distance=(1.5, 3.0), robot_clearance: float = 0.9,
                 goal_clearance: float = 1.2):
        self.gen = generator
        self.goal = goal
        self.remaining = count
        self.next_t = start_after_s
        self.interval = interval_s
        self.on_path = on_path
        self.ahead = tuple(ahead_distance)
        self.robot_clearance = robot_clearance
        self.goal_clearance = goal_clearance

    def maybe_spawn(self, t: float, pose: Pose2D, path: Optional[np.ndarray], world: ObstacleWorld) -> Optional[Obstacle]:
        if self.remaining <= 0 or t < self.next_t:
            return None
        for _ in range(50):
            xy = None
            if self.on_path and path is not None and len(path) > 1:
                d = np.linalg.norm(path[:, :2] - pose.xy, axis=1)
                i0 = int(np.argmin(d))
                seg = np.cumsum(np.r_[0, np.linalg.norm(np.diff(path[i0:, :2], axis=0), axis=1)])
                target = self.gen.rng.uniform(*self.ahead)
                j = i0 + int(np.searchsorted(seg, target))
                if j < len(path):
                    xy = (float(path[j, 0]), float(path[j, 1]))
            ob = self.gen.sample_obstacle(xy=xy)
            if math.hypot(ob.x - pose.x, ob.y - pose.y) < self.robot_clearance + ob.radius:
                continue
            if math.hypot(ob.x - self.goal.x, ob.y - self.goal.y) < self.goal_clearance + ob.radius:
                continue
            if world.sdf(np.array([[ob.x, ob.y]]))[0] < ob.radius + 0.2:
                continue
            world.add(ob)
            self.remaining -= 1
            self.next_t = t + self.interval
            return ob
        self.next_t = t + 0.5
        return None

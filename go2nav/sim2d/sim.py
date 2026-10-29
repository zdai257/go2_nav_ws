"""Lightweight kinematic simulator: same Navigator / planners / mapping as Isaac, no physics.

The Go2 is modelled at the "sport mode" level: body-frame velocity commands are
tracked with a first-order lag and multiplicative noise. A planar 2.5D lidar is
ray-cast against the ObstacleWorld.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..core.types import Pose2D, VelocityCommand, wrap_angle
from ..scenario import Scenario, build_scenario


class KinematicGo2:
    def __init__(self, pose: Pose2D, tau: float = 0.15, noise: float = 0.0, seed: Optional[int] = 0):
        self.pose = Pose2D(pose.x, pose.y, pose.yaw)
        self.vel = np.zeros(3)
        self.tau = tau
        self.noise = noise
        self.rng = np.random.default_rng(seed)

    def step(self, cmd: VelocityCommand, dt: float) -> None:
        target = cmd.as_array()
        if self.noise > 0:
            target = target * (1.0 + self.noise * self.rng.standard_normal(3))
        a = 1.0 - math.exp(-dt / self.tau) if self.tau > 0 else 1.0
        self.vel += a * (target - self.vel)
        vx, vy, wz = self.vel
        c, s = math.cos(self.pose.yaw), math.sin(self.pose.yaw)
        self.pose = Pose2D(self.pose.x + (c * vx - s * vy) * dt, self.pose.y + (s * vx + c * vy) * dt,
                           wrap_angle(self.pose.yaw + wz * dt))


class Lidar2D:
    def __init__(self, num_beams: int = 360, max_range: float = 8.0, min_range: float = 0.2,
                 mount_xy=(0.18, 0.0), min_obstacle_height: float = 0.0, noise_std: float = 0.01, seed=0):
        self.angles = np.linspace(-math.pi, math.pi, num_beams, endpoint=False)
        self.max_range, self.min_range = max_range, min_range
        self.mount = np.array(mount_xy)
        self.min_h = min_obstacle_height
        self.noise_std = noise_std
        self.rng = np.random.default_rng(seed)

    def scan(self, pose: Pose2D, world):
        c, s = math.cos(pose.yaw), math.sin(pose.yaw)
        origin = pose.xy + np.array([c * self.mount[0] - s * self.mount[1], s * self.mount[0] + c * self.mount[1]])
        ang = self.angles + pose.yaw
        r = world.raycast(origin, ang, self.max_range, self.min_h)
        hit = np.isfinite(r) & (r >= self.min_range)
        r = np.where(hit, r + self.noise_std * self.rng.standard_normal(len(r)), self.max_range)
        dirs = np.stack([np.cos(ang), np.sin(ang)], axis=1)
        pts = origin + dirs * r[:, None]
        return origin, pts[hit], pts[~hit]


@dataclass
class EpisodeResult:
    stats: Dict
    scenario: Scenario
    collisions: int = 0
    min_clearance: float = math.inf
    wall_time_s: float = 0.0
    spawned: List = field(default_factory=list)


def footprint_clearance(world, pose: Pose2D, circles, radius) -> float:
    c, s = math.cos(pose.yaw), math.sin(pose.yaw)
    pts = np.array([[pose.x + c * cx - s * cy, pose.y + s * cx + c * cy] for cx, cy in circles])
    return float(world.sdf(pts).min() - radius)


def run_episode(cfg: Dict, scenario: Optional[Scenario] = None, max_time: Optional[float] = None,
                callback=None) -> EpisodeResult:
    sc = scenario or build_scenario(cfg)
    scfg = cfg.get("sim2d", {})
    pcfg = cfg.get("perception", {}).get("lidar", {})
    dt = scfg.get("dt", 0.05)
    robot = KinematicGo2(sc.start, tau=scfg.get("tau", 0.15), noise=scfg.get("velocity_noise", 0.0),
                         seed=cfg["world"]["generator"].get("seed"))
    lidar = Lidar2D(num_beams=int(pcfg.get("horizontal_fov", 360) / pcfg.get("horizontal_resolution", 1.0)),
                    max_range=pcfg.get("max_range", 8.0), min_range=pcfg.get("min_range", 0.2))
    lidar_period = 1.0 / pcfg.get("rate_hz", 10.0)
    cm = cfg["map"]["costmap"]
    circles, rad = cm["footprint_circles"], 0.12  # physical body half-width for collision accounting
    nav = sc.navigator
    nav.set_goal(sc.goal)
    use_lidar = cfg["map"]["source"] == "lidar"
    t, next_scan = 0.0, 0.0
    collisions, min_clear, in_collision = 0, math.inf, False
    spawned = []
    max_time = max_time or cfg.get("navigator", {}).get("timeout", 180.0) + 1.0
    wall0 = time.perf_counter()
    while not nav.finished and t < max_time:
        if sc.spawner is not None:
            ob = sc.spawner.maybe_spawn(t, robot.pose, nav.path, sc.world)
            if ob is not None:
                spawned.append((t, ob))
        if use_lidar and t >= next_scan:
            origin, hits, misses = lidar.scan(robot.pose, sc.world)
            sc.map.update_scan(origin, hits, misses)
            next_scan = t + lidar_period
        cmd = nav.step(robot.pose, t, dt)
        robot.step(cmd, dt)
        clr = footprint_clearance(sc.world, robot.pose, circles, rad)
        min_clear = min(min_clear, clr)
        if clr < 0 and not in_collision:
            collisions += 1
        in_collision = clr < 0
        if callback is not None:
            callback(t, robot, nav, sc)
        t += dt
    stats = nav.stats()
    stats.update(collisions=collisions, min_clearance_m=round(min_clear, 3))
    return EpisodeResult(stats, sc, collisions, min_clear, time.perf_counter() - wall0, spawned)

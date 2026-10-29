"""Isaac Sim scene: ground, lights, obstacle colliders (synced from an ObstacleWorld), debug overlays."""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence

import numpy as np
from isaacsim.core.api.materials import PhysicsMaterial
from isaacsim.core.api.objects import FixedCuboid, FixedCylinder, GroundPlane
from isaacsim.core.utils.stage import get_current_stage
from pxr import Gf, Sdf, UsdLux

from ..world import Obstacle, ObstacleWorld
from .go2_robot import yaw_to_quat


class IsaacScene:
    def __init__(self, world, obstacles: ObstacleWorld, root: str = "/World/Obstacles"):
        self.world = world
        self.obstacles = obstacles
        self.root = root
        self._spawned: Dict[str, object] = {}

    def build(self, ground_friction: float = 1.0) -> None:
        b = self.obstacles.bounds
        size = 2.0 * max(abs(v) for v in b) + 10.0
        mat = PhysicsMaterial("/World/Physics/GroundMaterial", static_friction=ground_friction,
                              dynamic_friction=ground_friction, restitution=0.0)
        GroundPlane(prim_path="/World/Ground", size=size, color=np.array([0.42, 0.44, 0.46]), physics_material=mat)
        stage = get_current_stage()
        dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/Lights/Dome"))
        dome.CreateIntensityAttr(900.0)
        sun = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/Lights/Sun"))
        sun.CreateIntensityAttr(2500.0)
        sun.AddRotateXYZOp().Set(Gf.Vec3f(-40.0, 20.0, 0.0))
        self.sync()

    def sync(self) -> None:
        """Spawn colliders for obstacles that are not in the stage yet (call after world.add(...))."""
        for ob in self.obstacles.obstacles:
            if ob.name not in self._spawned:
                self._spawned[ob.name] = self.spawn(ob)

    def spawn(self, ob: Obstacle):
        path = f"{self.root}/{ob.name}"
        color = np.array(ob.color)
        if ob.shape == "circle":
            return FixedCylinder(prim_path=path, name=ob.name, position=np.array([ob.x, ob.y, ob.height / 2]),
                                 radius=float(ob.size[0]), height=float(ob.height), color=color)
        return FixedCuboid(prim_path=path, name=ob.name, position=np.array([ob.x, ob.y, ob.height / 2]),
                           orientation=yaw_to_quat(ob.yaw), size=1.0,
                           scale=np.array([ob.size[0], ob.size[1], ob.height]), color=color)


class DebugOverlay:
    """Path / trajectory / goal / lidar overlays drawn with isaacsim.util.debug_draw."""

    def __init__(self):
        from isaacsim.util.debug_draw import _debug_draw

        self.draw = _debug_draw.acquire_debug_draw_interface()

    def clear(self):
        self.draw.clear_lines()
        self.draw.clear_points()

    def polyline(self, xy: np.ndarray, z: float, color, width: float = 3.0):
        if xy is None or len(xy) < 2:
            return
        a = [(float(p[0]), float(p[1]), z) for p in xy[:-1]]
        b = [(float(p[0]), float(p[1]), z) for p in xy[1:]]
        self.draw.draw_lines(a, b, [tuple(color)] * len(a), [width] * len(a))

    def goal_marker(self, x: float, y: float, yaw: float, color=(1.0, 0.2, 0.2, 1.0)):
        r = 0.35
        n = 24
        ang = np.linspace(0, 2 * math.pi, n + 1)
        ring = np.stack([x + r * np.cos(ang), y + r * np.sin(ang)], axis=1)
        self.polyline(ring, 0.03, color, 4.0)
        self.draw.draw_lines([(x, y, 0.03)], [(x + 0.6 * math.cos(yaw), y + 0.6 * math.sin(yaw), 0.03)], [color], [5.0])

    def points(self, pts: np.ndarray, color, size: float = 4.0, max_points: int = 4000):
        if pts is None or len(pts) == 0:
            return
        if len(pts) > max_points:
            pts = pts[np.linspace(0, len(pts) - 1, max_points).astype(int)]
        p = [tuple(map(float, q)) for q in pts]
        self.draw.draw_points(p, [tuple(color)] * len(p), [size] * len(p))

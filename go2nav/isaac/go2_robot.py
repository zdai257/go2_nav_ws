"""Go2 articulation wrapper: spawning, base state, planar pose."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
from pxr import UsdPhysics

from ..core.types import Pose2D


def yaw_to_quat(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


def quat_to_yaw(q: np.ndarray) -> float:
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


class Go2Robot:
    """Spawns the Go2 USD under ``prim_path`` and exposes base state in the world frame."""

    def __init__(self, world, usd_path: str, start: Pose2D, prim_path: str = "/World/Go2", z0: float = 0.4):
        from .go2_asset import force_joint_drives

        self.prim_path = prim_path
        add_reference_to_stage(usd_path, prim_path)
        stage = get_current_stage()
        force_joint_drives(stage, prim_path)
        root = None
        for prim in stage.Traverse():
            p = str(prim.GetPath())
            if p.startswith(prim_path) and prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                root = p
                break
        self.root_path = root or prim_path
        self.base_path = self._find_base(stage) or self.root_path
        self.art = SingleArticulation(prim_path=self.root_path, name="go2",
                                      position=np.array([start.x, start.y, z0]), orientation=yaw_to_quat(start.yaw))
        world.scene.add(self.art)

    def _find_base(self, stage) -> Optional[str]:
        for prim in stage.Traverse():
            p = str(prim.GetPath())
            if p.startswith(self.prim_path) and p.endswith("/base") and prim.HasAPI(UsdPhysics.RigidBodyAPI):
                return p
        return None

    # ------------------------------------------------------------------ state
    @property
    def dof_names(self):
        return list(self.art.dof_names)

    def world_pose(self):
        return self.art.get_world_pose()  # position (3,), quaternion wxyz (4,)

    def pose2d(self) -> Pose2D:
        p, q = self.world_pose()
        return Pose2D(float(p[0]), float(p[1]), quat_to_yaw(q))

    def base_velocity_body(self):
        _, q = self.world_pose()
        R_bw = quat_to_rot(q).T
        return R_bw @ self.art.get_linear_velocity(), R_bw @ self.art.get_angular_velocity()

    def projected_gravity(self) -> np.ndarray:
        _, q = self.world_pose()
        return quat_to_rot(q).T @ np.array([0.0, 0.0, -1.0])

    def has_fallen(self, min_height: float = 0.1) -> bool:
        p, _ = self.world_pose()
        return bool(p[2] < min_height or self.projected_gravity()[2] > -0.5)

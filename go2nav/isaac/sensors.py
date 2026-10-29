"""Simulated perception on the Go2: PhysX lidar (2.5D scan -> occupancy) and optional front RGB-D camera."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import omni.kit.commands
import omni.usd
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.sensors.physx import _range_sensor

from ..core.config import resolve_path
from .go2_asset import rpy_to_quat_wxyz, urdf_frames


class Go2Lidar:
    """PhysX ray-cast lidar mounted on a URDF frame (``nav_lidar`` or ``radar``) of the Go2 base.

    ``rotation_rate = 0`` -> a full 360 deg sweep is produced every physics step.
    ``scan()`` returns world-frame (origin_xy, obstacle_hits_xy, free_endpoints_xy):
    returns between ``z_min`` and ``z_max`` are obstacles; ground hits, overhead hits and
    no-returns only carve free space.
    """

    def __init__(self, base_prim_path: str, cfg: Dict):
        lc = cfg["perception"]["lidar"]
        self.cfg = lc
        frames = urdf_frames(resolve_path(cfg["isaac"]["urdf"]))
        mount = lc.get("mount", "nav_lidar")
        xyz, rpy, parent = frames.get(mount, ((0.18, 0.0, 0.095), (0.0, 0.0, 0.0), "base"))
        self.path = f"{base_prim_path}/{mount}_sensor"
        omni.kit.commands.execute(
            "RangeSensorCreateLidar", path=f"/{mount}_sensor", parent=base_prim_path,
            min_range=float(lc["min_range"]), max_range=float(lc["max_range"]),
            draw_points=False, draw_lines=False,
            horizontal_fov=float(lc["horizontal_fov"]), vertical_fov=float(lc["vertical_fov"]),
            horizontal_resolution=float(lc["horizontal_resolution"]), vertical_resolution=float(lc["vertical_resolution"]),
            rotation_rate=0.0, high_lod=True, yaw_offset=0.0, enable_semantics=False,
        )
        SingleXFormPrim(self.path).set_local_pose(translation=np.array(xyz), orientation=np.array(rpy_to_quat_wxyz(rpy)))
        self.iface = _range_sensor.acquire_lidar_sensor_interface()
        self.max_range = float(lc["max_range"])
        self.min_range = float(lc["min_range"])
        self.z_min, self.z_max = float(lc["z_min"]), float(lc["z_max"])
        self._prim = omni.usd.get_context().get_stage().GetPrimAtPath(self.path)
        self.last_points_world: Optional[np.ndarray] = None

    def _world_tf(self) -> np.ndarray:
        return np.array(omni.usd.get_world_transform_matrix(self._prim), dtype=float)  # row-vector convention

    def scan(self, base_pose=None, self_box=(0.5, 0.32)) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """``base_pose`` (Pose2D) enables removal of returns from the robot's own legs/body.

        The PhysX lidar is computed during the Kit app update, so the simulation must be stepped
        with ``render=True`` (also headless); otherwise its buffers hold uninitialised memory.
        """
        local = np.asarray(self.iface.get_point_cloud_data(self.path), dtype=float).reshape(-1, 3)
        if local.size == 0:
            return None
        r = np.linalg.norm(local, axis=1)
        if not np.all(np.isfinite(r)) or r.max() > 2.0 * self.max_range or r.max() == 0.0:
            return None
        # no-return beams come back at max range
        hit = (r >= self.min_range) & (r < 0.999 * self.max_range)
        M = self._world_tf()
        pts = local @ M[:3, :3] + M[3, :3]
        origin = M[3, :3]
        keep = np.ones(len(pts), dtype=bool)
        if base_pose is not None:
            loc = base_pose.to_local(pts[:, :2])
            keep = ~((np.abs(loc[:, 0]) < self_box[0]) & (np.abs(loc[:, 1]) < self_box[1]))
        self.last_points_world = pts[hit & keep]
        obstacle = hit & keep & (pts[:, 2] >= self.z_min) & (pts[:, 2] <= self.z_max)
        free = keep & ~obstacle
        return origin[:2], pts[obstacle, :2], pts[free, :2]


class Go2FrontCamera:
    """Front RGB-D camera at the URDF ``front_camera`` frame (optional; costs GPU time)."""

    def __init__(self, base_prim_path: str, cfg: Dict):
        from isaacsim.sensors.camera import Camera

        cc = cfg["perception"]["camera"]
        xyz, rpy, _ = urdf_frames(resolve_path(cfg["isaac"]["urdf"])).get(
            "front_camera", ((0.32715, 0.0, 0.04297), (0, 0, 0), "base"))
        self.cam = Camera(prim_path=f"{base_prim_path}/front_camera_sensor", frequency=cc.get("rate_hz", 10),
                          resolution=tuple(cc.get("resolution", (640, 480))), translation=np.array(xyz),
                          orientation=np.array(rpy_to_quat_wxyz(rpy)))

    def initialize(self):
        self.cam.initialize()
        self.cam.add_distance_to_image_plane_to_frame()

    def rgb(self):
        return self.cam.get_rgba()

    def depth(self):
        return self.cam.get_current_frame().get("distance_to_image_plane")

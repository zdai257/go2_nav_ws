"""Path followers: (pose, path) -> body-frame VelocityCommand (vx, vy, wz).

This is the boundary to the locomotion layer: everything below it (RL policy or
kinematic base) only sees (vx, vy, wz) - the "sport mode" style interface of the
real Go2 (``SportClient.Move(vx, vy, vyaw)``).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..core.registry import Registry
from ..core.types import Pose2D, VelocityCommand, wrap_angle

FOLLOWERS = Registry("follower")


@dataclass
class VelocityLimits:
    vx_max: float = 0.8
    vx_min: float = -0.4
    vy_max: float = 0.4
    wz_max: float = 1.0
    ax_max: float = 1.5      # m/s^2
    ay_max: float = 1.0
    alpha_max: float = 3.0   # rad/s^2
    min_turn_rate: float = 0.0  # floor for pure in-place rotation (gaits with a yaw deadband) [rad/s]


class PathFollower(ABC):
    """Base follower with progress tracking, goal handling and velocity/acceleration limiting.

    Subclasses implement ``_track(pose, path, idx) -> VelocityCommand``.

    Args:
        lookahead: lookahead distance [m]
        goal_xy_tol / goal_yaw_tol_deg: success tolerances
        align_final_yaw: rotate in place to the goal yaw after reaching goal xy
        slow_down_dist: start decelerating this far from the goal [m]
    """

    registry_name = "?"

    def __init__(self, lookahead: float = 0.6, goal_xy_tol: float = 0.2, goal_yaw_tol_deg: float = 10.0,
                 align_final_yaw: bool = True, slow_down_dist: float = 0.8, limits: Optional[dict] = None):
        self.lookahead = lookahead
        self.goal_xy_tol = goal_xy_tol
        self.goal_yaw_tol = math.radians(goal_yaw_tol_deg)
        self.align_final_yaw = align_final_yaw
        self.slow_down_dist = slow_down_dist
        self.limits = VelocityLimits(**(limits or {}))
        self.path: Optional[np.ndarray] = None
        self.idx = 0
        self._last = VelocityCommand.zero()
        self.done = False

    def set_path(self, path: np.ndarray) -> None:
        self.path = np.asarray(path, dtype=float)
        self.idx = 0
        self.done = False

    def reset(self) -> None:
        self.path, self.idx, self.done = None, 0, False
        self._last = VelocityCommand.zero()

    # ------------------------------------------------------------------ main
    def compute(self, pose: Pose2D, dt: float) -> VelocityCommand:
        if self.path is None or len(self.path) == 0 or self.done:
            return self._limit(VelocityCommand.zero(), dt)
        goal = self.path[-1]
        self._advance(pose)
        dist_goal = math.hypot(goal[0] - pose.x, goal[1] - pose.y)
        if dist_goal <= self.goal_xy_tol:
            yaw_err = wrap_angle(goal[2] - pose.yaw)
            if not self.align_final_yaw or abs(yaw_err) <= self.goal_yaw_tol:
                self.done = True
                return self._limit(VelocityCommand.zero(), dt)
            return self._limit(VelocityCommand(0.0, 0.0, float(np.clip(1.5 * yaw_err, -self.limits.wz_max, self.limits.wz_max))), dt)
        cmd = self._track(pose, self.path, self.idx)
        scale = min(1.0, max(0.25, dist_goal / max(self.slow_down_dist, 1e-6)))
        cmd.vx *= scale
        cmd.vy *= scale
        return self._limit(cmd, dt)

    @abstractmethod
    def _track(self, pose: Pose2D, path: np.ndarray, idx: int) -> VelocityCommand: ...

    # --------------------------------------------------------------- helpers
    def _advance(self, pose: Pose2D, window: int = 40) -> None:
        """Move the progress index to the closest path point ahead (monotone)."""
        seg = self.path[self.idx : self.idx + window, :2]
        if len(seg):
            self.idx += int(np.argmin(np.linalg.norm(seg - pose.xy, axis=1)))

    def lookahead_index(self, pose: Pose2D, L: Optional[float] = None) -> int:
        L = self.lookahead if L is None else L
        d = np.linalg.norm(self.path[self.idx :, :2] - pose.xy, axis=1)
        beyond = np.nonzero(d >= L)[0]
        return self.idx + int(beyond[0]) if len(beyond) else len(self.path) - 1

    def cross_track_error(self, pose: Pose2D) -> float:
        if self.path is None or len(self.path) == 0:
            return 0.0
        return float(np.min(np.linalg.norm(self.path[:, :2] - pose.xy, axis=1)))

    def _limit(self, cmd: VelocityCommand, dt: float) -> VelocityCommand:
        L = self.limits
        vx = float(np.clip(cmd.vx, L.vx_min, L.vx_max))
        vy = float(np.clip(cmd.vy, -L.vy_max, L.vy_max))
        wz = float(np.clip(cmd.wz, -L.wz_max, L.wz_max))
        if L.min_turn_rate > 0 and abs(vx) < 0.05 and abs(vy) < 0.05 and abs(wz) > 0.02:
            wz = math.copysign(max(abs(wz), min(L.min_turn_rate, L.wz_max)), wz)
        if dt > 0:
            p = self._last
            vx = p.vx + float(np.clip(vx - p.vx, -L.ax_max * dt, L.ax_max * dt))
            vy = p.vy + float(np.clip(vy - p.vy, -L.ay_max * dt, L.ay_max * dt))
            wz = p.wz + float(np.clip(wz - p.wz, -L.alpha_max * dt, L.alpha_max * dt))
        self._last = VelocityCommand(vx, vy, wz)
        return self._last


@FOLLOWERS.register("pure_pursuit")
class PurePursuitFollower(PathFollower):
    """Regulated pure pursuit (unicycle style: vx + wz, vy = 0).

    Rotates in place when the heading error to the lookahead point exceeds
    ``rotate_in_place_deg``; slows down on high curvature.
    """

    def __init__(self, rotate_in_place_deg: float = 50.0, k_heading: float = 1.8,
                 min_turn_radius_for_full_speed: float = 1.0, **kw):
        super().__init__(**kw)
        self.rot_thresh = math.radians(rotate_in_place_deg)
        self.k_heading = k_heading
        self.r_full = min_turn_radius_for_full_speed

    def _track(self, pose, path, idx):
        j = self.lookahead_index(pose)
        tgt = path[j]
        local = pose.to_local(tgt[None, :2])[0]
        alpha = math.atan2(local[1], local[0])
        Ld = max(np.hypot(*local), 1e-3)
        if abs(alpha) > self.rot_thresh:
            return VelocityCommand(0.0, 0.0, self.k_heading * alpha)
        curvature = 2.0 * math.sin(alpha) / Ld
        v = self.limits.vx_max
        if abs(curvature) > 1e-6:
            v = min(v, v * (1.0 / abs(curvature)) / self.r_full)
        v = max(v, 0.15)
        return VelocityCommand(v, 0.0, v * curvature)


@FOLLOWERS.register("holonomic")
class HolonomicFollower(PathFollower):
    """Omnidirectional tracking (uses vy): translate toward the lookahead point in the
    body frame while turning toward the path's yaw (tangent or planner-provided yaw).
    """

    def __init__(self, k_pos: float = 1.2, k_yaw: float = 1.5, heading_source: str = "path", **kw):
        super().__init__(**kw)
        self.k_pos, self.k_yaw = k_pos, k_yaw
        self.heading_source = heading_source  # "path" | "lookahead"

    def _track(self, pose, path, idx):
        j = self.lookahead_index(pose)
        tgt = path[j]
        local = pose.to_local(tgt[None, :2])[0]
        dist = np.hypot(*local)
        speed = min(self.limits.vx_max, self.k_pos * dist)
        v = local / max(dist, 1e-6) * speed
        if self.heading_source == "lookahead":
            yaw_ref = math.atan2(tgt[1] - pose.y, tgt[0] - pose.x)
        else:
            yaw_ref = path[min(idx + 2, len(path) - 1), 2]
        yaw_err = wrap_angle(yaw_ref - pose.yaw)
        # walk mostly forward: shrink lateral speed relative to the Go2's weaker strafing
        vy = float(np.clip(v[1], -self.limits.vy_max, self.limits.vy_max))
        return VelocityCommand(float(v[0]), vy, self.k_yaw * yaw_err)


def build_follower(cfg: dict) -> PathFollower:
    name = cfg.get("name", "pure_pursuit")
    kwargs = dict(cfg.get("common", {}) or {})
    kwargs.update((cfg.get("params", {}) or {}).get(name, {}) or {})
    return FOLLOWERS.build(name, **kwargs)

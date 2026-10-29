"""A-to-B navigation state machine shared by the 2D simulator and Isaac Sim.

    nav = Navigator(planner, follower, map_provider)
    nav.set_goal(Pose2D(8, 3, 0))
    while not nav.finished:
        map_provider.update_scan(...)          # perception (optional)
        cmd = nav.step(robot_pose, t, dt)      # -> VelocityCommand for the locomotion layer
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..control.base import PathFollower
from ..core.types import Pose2D, VelocityCommand, path_length
from ..perception.mapping import MapProvider
from ..planning.base import GlobalPlanner, PlanResult


class NavStatus(str, Enum):
    IDLE = "idle"
    FOLLOWING = "following"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass
class NavEvent:
    t: float
    kind: str
    message: str = ""


@dataclass
class NavigatorConfig:
    replan_period: float = 0.0          # periodic replanning [s] (0 = only on events)
    path_check_period: float = 0.25     # how often to validate the remaining path against the map [s]
    path_check_horizon: float = 5.0     # only validate this much of the remaining path [m]
    use_footprint_check: bool = False   # validate with the yaw-dependent footprint instead of the disc
    max_cross_track: float = 0.8        # replan if further than this from the path [m]
    max_plan_failures: int = 10         # consecutive failures before giving up
    retry_period: float = 0.5           # wait between failed planning attempts [s]
    stuck_window: float = 8.0           # replan if the robot moved < stuck_distance in this window [s]
    stuck_distance: float = 0.15
    max_stuck_replans: int = 4
    timeout: float = 180.0              # give up after this much sim time [s]


class Navigator:
    def __init__(self, planner: GlobalPlanner, follower: PathFollower, map_provider: MapProvider,
                 config: Optional[Dict] = None):
        self.planner = planner
        self.follower = follower
        self.map = map_provider
        self.cfg = NavigatorConfig(**(config or {}))
        self.reset()

    # ----------------------------------------------------------------- state
    def reset(self) -> None:
        self.status = NavStatus.IDLE
        self.goal: Optional[Pose2D] = None
        self.path: Optional[np.ndarray] = None
        self.plans: List[Tuple[float, PlanResult]] = []
        self.events: List[NavEvent] = []
        self.trajectory: List[Tuple[float, float, float, float]] = []
        self.commands: List[Tuple[float, float, float, float]] = []
        self._t0 = None
        self._last_plan_t = -math.inf
        self._last_check_t = -math.inf
        self._checked_version = -1
        self._fail_count = 0
        self._stuck_count = 0
        self._last_fail_t = -math.inf
        self.follower.reset()

    @property
    def finished(self) -> bool:
        return self.status in (NavStatus.SUCCEEDED, NavStatus.FAILED)

    def set_goal(self, goal: Pose2D) -> None:
        self.reset()
        self.goal = goal
        self.status = NavStatus.FOLLOWING

    # ------------------------------------------------------------------ loop
    def step(self, pose: Pose2D, t: float, dt: float) -> VelocityCommand:
        if self._t0 is None:
            self._t0 = t
        self.trajectory.append((t, pose.x, pose.y, pose.yaw))
        if self.status != NavStatus.FOLLOWING:
            return VelocityCommand.zero()
        if t - self._t0 > self.cfg.timeout:
            return self._fail(t, "timeout")

        reason = self._replan_reason(pose, t)
        if reason:
            if t - self._last_fail_t < self.cfg.retry_period and self.path is None:
                return VelocityCommand.zero()
            if not self._plan(pose, t, reason):
                if self._fail_count >= self.cfg.max_plan_failures:
                    return self._fail(t, "planning failed repeatedly")
                if self.path is None or reason == "path blocked":
                    self.path = None
                    self.follower.reset()
                    return VelocityCommand.zero()

        cmd = self.follower.compute(pose, dt)
        if self.follower.done:
            self.status = NavStatus.SUCCEEDED
            self.events.append(NavEvent(t, "succeeded"))
            cmd = VelocityCommand.zero()
        self.commands.append((t, cmd.vx, cmd.vy, cmd.wz))
        return cmd

    # --------------------------------------------------------------- helpers
    def _replan_reason(self, pose: Pose2D, t: float) -> Optional[str]:
        c = self.cfg
        if self.path is None:
            return "no path"
        if c.replan_period > 0 and t - self._last_plan_t >= c.replan_period:
            return "periodic"
        if self.follower.cross_track_error(pose) > c.max_cross_track:
            return "off path"
        costmap = self.map.costmap()
        if t - self._last_check_t >= c.path_check_period and self.map.version != self._checked_version:
            self._last_check_t = t
            self._checked_version = self.map.version
            rem = self.path[self.follower.idx :]
            if len(rem) > 1:
                L = np.cumsum(np.r_[0, np.linalg.norm(np.diff(rem[:, :2], axis=0), axis=1)])
                rem = rem[L <= c.path_check_horizon]
                # skip the first ~robot length: the robot itself may sit in the inflated zone
                rem = rem[np.linalg.norm(rem[:, :2] - pose.xy, axis=1) > 0.35] if len(rem) > 2 else rem
                free, _ = costmap.path_free(rem, use_footprint=c.use_footprint_check)
                if not free:
                    return "path blocked"
        if len(self.trajectory) > 2:
            win = [p for p in self.trajectory if p[0] >= t - c.stuck_window]
            if win and t - self._last_plan_t > c.stuck_window and win[0][0] <= t - 0.95 * c.stuck_window:
                moved = math.hypot(pose.x - win[0][1], pose.y - win[0][2])
                turned = abs(pose.yaw - win[0][3])
                if moved < c.stuck_distance and turned < 0.2:
                    self._stuck_count += 1
                    if self._stuck_count > c.max_stuck_replans:
                        self._fail(t, "stuck")
                        return None
                    return "stuck"
        return None

    def _plan(self, pose: Pose2D, t: float, reason: str) -> bool:
        costmap = self.map.costmap()
        res = self.planner.plan(pose, self.goal, costmap)
        self._last_plan_t = t
        self.plans.append((t, res))
        self.events.append(NavEvent(t, "plan" if res.success else "plan_failed",
                                    f"{reason}: {res.message} ({res.planning_time * 1e3:.0f} ms, L={res.length:.2f} m)"))
        if res.success:
            self.path = res.path
            self.follower.set_path(res.path)
            self._fail_count = 0
            self._checked_version = self.map.version
            return True
        self._fail_count += 1
        self._last_fail_t = t
        return False

    def _fail(self, t: float, msg: str) -> VelocityCommand:
        self.status = NavStatus.FAILED
        self.events.append(NavEvent(t, "failed", msg))
        return VelocityCommand.zero()

    # ----------------------------------------------------------------- stats
    def stats(self) -> Dict:
        traj = np.array([p[1:] for p in self.trajectory]) if self.trajectory else np.zeros((0, 3))
        ok_plans = [r for _, r in self.plans if r.success]
        first = ok_plans[0] if ok_plans else None
        dur = self.trajectory[-1][0] - self.trajectory[0][0] if len(self.trajectory) > 1 else 0.0
        return {
            "planner": self.planner.name,
            "follower": self.follower.registry_name,
            "status": self.status.value,
            "success": self.status == NavStatus.SUCCEEDED,
            "time_s": round(dur, 3),
            "travelled_m": round(path_length(traj), 3) if len(traj) else 0.0,
            "first_plan_length_m": round(first.length, 3) if first else None,
            "first_plan_time_ms": round(first.planning_time * 1e3, 2) if first else None,
            "mean_plan_time_ms": round(float(np.mean([r.planning_time for r in ok_plans])) * 1e3, 2) if ok_plans else None,
            "num_plans": len(self.plans),
            "num_replans": max(0, len(self.plans) - 1),
            "final_goal_error_m": round(math.hypot(traj[-1, 0] - self.goal.x, traj[-1, 1] - self.goal.y), 3)
            if len(traj) and self.goal else None,
        }

"""Locomotion backends: turn a body-frame (vx, vy, wz) command into robot motion.

* ``RLPolicyLocomotion`` - Isaac Lab pretrained Go2 velocity-tracking policy (physics, real gait)
* ``KinematicLocomotion`` - ideal base tracking (teleports the base, legs frozen): isolates planning
  from locomotion errors and runs faster

Both are called once per physics step via ``step(cmd, dt)``.
"""

from __future__ import annotations

import math
import os
from abc import ABC, abstractmethod
from typing import Dict

import numpy as np
import yaml
from isaacsim.core.utils.types import ArticulationAction

from ..core.config import resolve_path
from ..core.registry import Registry
from ..core.types import VelocityCommand
from .go2_robot import Go2Robot, quat_to_yaw, yaw_to_quat

LOCOMOTION = Registry("locomotion")


class Locomotion(ABC):
    registry_name = "?"

    def __init__(self, robot: Go2Robot, cfg: Dict):
        self.robot = robot
        self.cfg = cfg
        pcfg_path = os.path.join(resolve_path(cfg["locomotion"]["policy_dir"]), "policy.yaml")
        with open(pcfg_path) as f:
            self.pcfg = yaml.safe_load(f)
        self.policy_joint_names = self.pcfg["joint_names"]
        self.default_pos_policy = np.array(self.pcfg["default_joint_pos"], dtype=np.float32)

    def initialize(self) -> None:
        """Call after ``world.reset()``: map joints, set gains, put legs in the standing pose."""
        dof = self.robot.dof_names
        missing = [n for n in self.policy_joint_names if n not in dof]
        if missing:
            raise RuntimeError(f"Go2 USD is missing joints {missing}; has {dof}")
        self.p2d = np.array([dof.index(n) for n in self.policy_joint_names])  # policy idx -> dof idx
        self.default_pos_dof = np.zeros(len(dof), dtype=np.float32)
        self.default_pos_dof[self.p2d] = self.default_pos_policy
        art = self.robot.art
        n = len(dof)
        av = art._articulation_view
        av.set_gains(kps=np.full((1, n), self.pcfg["stiffness"]), kds=np.full((1, n), self.pcfg["damping"]))
        av.set_max_efforts(np.full((1, n), self.pcfg["effort_limit"]))
        av.set_max_joint_velocities(np.full((1, n), self.pcfg["velocity_limit"]))
        art.set_solver_position_iteration_count(4)
        art.set_solver_velocity_iteration_count(0)
        art.set_enabled_self_collisions(False)
        art.set_joint_positions(self.default_pos_dof)
        art.set_joint_velocities(np.zeros(n))
        art.apply_action(ArticulationAction(joint_positions=self.default_pos_dof))

    @abstractmethod
    def step(self, cmd: VelocityCommand, dt: float) -> None: ...


@LOCOMOTION.register("policy")
class RLPolicyLocomotion(Locomotion):
    """Numpy re-implementation of the rsl_rl actor (no torch needed)."""

    def __init__(self, robot, cfg):
        super().__init__(robot, cfg)
        w = np.load(os.path.join(resolve_path(cfg["locomotion"]["policy_dir"]), self.pcfg["weights"]))
        n_layers = len([k for k in w.files if k.startswith("W")])
        self.layers = [(w[f"W{i}"].astype(np.float32), w[f"b{i}"].astype(np.float32)) for i in range(n_layers)]
        self.decimation = int(self.pcfg["decimation"])
        self.action_scale = float(self.pcfg["action_scale"])
        lim = self.pcfg["command_limits"]
        self.cmd_lo = np.array([lim["vx"][0], lim["vy"][0], lim["wz"][0]])
        self.cmd_hi = np.array([lim["vx"][1], lim["vy"][1], lim["wz"][1]])
        self.last_action = np.zeros(12, dtype=np.float32)
        self.counter = 0
        self.height_scan_mode = self.pcfg.get("height_scan_mode", "relative")
        self.height_scan = np.full(int(self.pcfg.get("height_scan_dim", 0)),
                                   float(self.pcfg.get("height_scan_value", 1.0)), dtype=np.float32)

    def _mlp(self, x: np.ndarray) -> np.ndarray:
        for i, (W, b) in enumerate(self.layers):
            x = W @ x + b
            if i < len(self.layers) - 1:
                x = np.where(x > 0, x, np.expm1(x))  # ELU
        return x

    def observation(self, cmd: VelocityCommand) -> np.ndarray:
        lin_b, ang_b = self.robot.base_velocity_body()
        q = self.robot.art.get_joint_positions()[self.p2d]
        qd = self.robot.art.get_joint_velocities()[self.p2d]
        c = np.clip(cmd.as_array(), self.cmd_lo, self.cmd_hi)
        if self.height_scan.size and self.height_scan_mode == "relative":
            # flat ground at z=0: sensor height - hit height - 0.5
            self.height_scan[:] = np.clip(self.robot.world_pose()[0][2] - 0.5, -1.0, 1.0)
        return np.concatenate([lin_b, ang_b, self.robot.projected_gravity(), c,
                               q - self.default_pos_policy, qd, self.last_action, self.height_scan]).astype(np.float32)

    def initialize(self) -> None:
        super().initialize()
        self.actuator = self.cfg["locomotion"].get("actuator", "implicit")
        self.target = self.default_pos_dof.copy()
        if self.actuator == "dc_motor":
            n = len(self.robot.dof_names)
            self.robot.art._articulation_view.set_gains(kps=np.zeros((1, n)), kds=np.zeros((1, n)))
            self.robot.art.get_articulation_controller().switch_control_mode("effort")

    def _dc_motor_torque(self) -> np.ndarray:
        """Isaac Lab ``DCMotor``: explicit PD + four-quadrant torque-speed saturation (as in training)."""
        p = self.pcfg
        q = self.robot.art.get_joint_positions()
        qd = self.robot.art.get_joint_velocities()
        tau = p["stiffness"] * (self.target - q) - p["damping"] * qd
        sat, vlim, elim = p.get("saturation_effort", p["effort_limit"]), p["velocity_limit"], p["effort_limit"]
        qd_c = np.clip(qd, -vlim, vlim)
        hi = np.minimum(sat * (1.0 - qd_c / vlim), elim)
        lo = np.maximum(sat * (-1.0 - qd_c / vlim), -elim)
        return np.clip(tau, lo, hi)

    def step(self, cmd: VelocityCommand, dt: float) -> None:
        if self.counter % self.decimation == 0:
            action = self._mlp(self.observation(cmd))
            self.last_action = action
            self.target = self.default_pos_dof.copy()
            self.target[self.p2d] = self.default_pos_policy + self.action_scale * action
        if self.actuator == "dc_motor":
            self.robot.art.set_joint_efforts(self._dc_motor_torque())
        else:
            self.robot.art.apply_action(ArticulationAction(joint_positions=self.target))
        self.counter += 1


@LOCOMOTION.register("kinematic")
class KinematicLocomotion(Locomotion):
    """Ideal 'sport mode': base follows the command (first-order lag), legs held in stance."""

    STAND_HEIGHT = 0.345

    def __init__(self, robot, cfg):
        super().__init__(robot, cfg)
        self.tau = float(cfg["locomotion"].get("kinematic", {}).get("tau", 0.15))
        self.vel = np.zeros(3)

    def initialize(self) -> None:
        super().initialize()
        p, q = self.robot.world_pose()
        self.x, self.y, self.yaw = float(p[0]), float(p[1]), quat_to_yaw(q)

    def step(self, cmd: VelocityCommand, dt: float) -> None:
        a = 1.0 - math.exp(-dt / self.tau) if self.tau > 0 else 1.0
        self.vel += a * (cmd.as_array() - self.vel)
        vx, vy, wz = self.vel
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        vwx, vwy = c * vx - s * vy, s * vx + c * vy
        self.x += vwx * dt
        self.y += vwy * dt
        self.yaw = (self.yaw + wz * dt + math.pi) % (2 * math.pi) - math.pi
        art = self.robot.art
        art.set_world_pose(np.array([self.x, self.y, self.STAND_HEIGHT]), yaw_to_quat(self.yaw))
        art.set_linear_velocity(np.array([vwx, vwy, 0.0]))
        art.set_angular_velocity(np.array([0.0, 0.0, wz]))
        art.apply_action(ArticulationAction(joint_positions=self.default_pos_dof))


def build_locomotion(robot: Go2Robot, cfg: Dict) -> Locomotion:
    return LOCOMOTION.build(cfg["locomotion"]["backend"], robot=robot, cfg=cfg)

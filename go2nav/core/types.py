from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


def wrap_angle(a):
    """Wrap angle(s) to [-pi, pi)."""
    return (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi if isinstance(a, np.ndarray) else (a + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class Pose2D:
    """Planar pose of the robot base in the world frame: (x [m], y [m], yaw [rad])."""

    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0

    @classmethod
    def from_seq(cls, s: Sequence[float]) -> "Pose2D":
        return cls(float(s[0]), float(s[1]), float(s[2]) if len(s) > 2 else 0.0)

    @classmethod
    def parse(cls, text: str) -> "Pose2D":
        """'x,y[,yaw]' with yaw in radians (or degrees with a trailing 'd', e.g. '1,2,90d')."""
        parts = [p.strip() for p in text.split(",")]
        yaw = 0.0
        if len(parts) > 2:
            yaw = math.radians(float(parts[2][:-1])) if parts[2].endswith("d") else float(parts[2])
        return cls(float(parts[0]), float(parts[1]), yaw)

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.yaw], dtype=float)

    @property
    def xy(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=float)

    def distance_to(self, other: "Pose2D") -> float:
        return math.hypot(other.x - self.x, other.y - self.y)

    def yaw_error_to(self, other: "Pose2D") -> float:
        return wrap_angle(other.yaw - self.yaw)

    def to_local(self, pts: np.ndarray) -> np.ndarray:
        """World-frame xy points (N,2) -> this pose's body frame."""
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        d = np.atleast_2d(pts) - self.xy
        return np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], axis=1)

    def __repr__(self) -> str:
        return f"Pose2D(x={self.x:.3f}, y={self.y:.3f}, yaw={math.degrees(self.yaw):.1f}deg)"


@dataclass
class VelocityCommand:
    """Body-frame twist sent to the locomotion layer: forward vx, lateral vy [m/s], yaw rate wz [rad/s]."""

    vx: float = 0.0
    vy: float = 0.0
    wz: float = 0.0

    def as_array(self) -> np.ndarray:
        return np.array([self.vx, self.vy, self.wz], dtype=float)

    @classmethod
    def zero(cls) -> "VelocityCommand":
        return cls(0.0, 0.0, 0.0)


def path_length(path: np.ndarray | Iterable) -> float:
    p = np.asarray(path, dtype=float)
    if len(p) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1)))

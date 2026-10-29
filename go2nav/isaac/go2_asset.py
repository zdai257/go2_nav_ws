"""Go2 digital-twin asset handling: URDF -> USD conversion (cached) and sensor-frame lookup."""

from __future__ import annotations

import math
import os
import xml.etree.ElementTree as ET
from typing import Dict, Tuple

from ..core.config import resolve_path

NVIDIA_GO2_USD = ("https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/"
                  "Isaac/IsaacLab/Robots/Unitree/Go2/go2.usd")


def urdf_frames(urdf_path: str) -> Dict[str, Tuple[Tuple[float, float, float], Tuple[float, float, float], str]]:
    """{child_link: (xyz, rpy, parent)} for every fixed joint - used to mount sensors on the base."""
    out = {}
    for j in ET.parse(urdf_path).getroot().iter("joint"):
        o = j.find("origin")
        xyz = tuple(float(v) for v in (o.get("xyz", "0 0 0").split() if o is not None else (0, 0, 0)))
        rpy = tuple(float(v) for v in (o.get("rpy", "0 0 0").split() if o is not None else (0, 0, 0)))
        out[j.find("child").get("link")] = (xyz, rpy, j.find("parent").get("link"))
    return out


def ensure_go2_usd(cfg: Dict, force: bool = False) -> str:
    """Return a USD path for the Go2. ``robot_source: urdf`` converts the workspace URDF once and caches it."""
    icfg = cfg["isaac"]
    if icfg.get("robot_source", "urdf") == "nvidia":
        return NVIDIA_GO2_USD
    urdf = resolve_path(icfg["urdf"])
    usd = resolve_path(icfg["usd_cache"])
    if not force and os.path.exists(usd) and os.path.getmtime(usd) >= os.path.getmtime(urdf):
        return usd
    convert_urdf_to_usd(urdf, usd)
    return usd


def convert_urdf_to_usd(urdf_path: str, usd_path: str) -> str:
    import omni.kit.commands

    os.makedirs(os.path.dirname(usd_path), exist_ok=True)
    if os.path.exists(usd_path):
        os.remove(usd_path)
    _, cfg = omni.kit.commands.execute("URDFCreateImportConfig")
    cfg.merge_fixed_joints = True          # rotor / foot / head links merged; sensor frames kept (dont_collapse)
    cfg.fix_base = False
    cfg.import_inertia_tensor = True
    cfg.self_collision = False
    cfg.convex_decomp = False
    cfg.make_default_prim = True
    cfg.create_physics_scene = False
    cfg.distance_scale = 1.0
    if hasattr(cfg, "default_drive_type"):
        from isaacsim.asset.importer.urdf import _urdf

        cfg.default_drive_type = _urdf.UrdfJointTargetType.JOINT_DRIVE_POSITION
    if hasattr(cfg, "default_drive_strength"):
        cfg.default_drive_strength = 25.0
    if hasattr(cfg, "default_position_drive_damping"):
        cfg.default_position_drive_damping = 0.5
    ok, prim_path = omni.kit.commands.execute("URDFParseAndImportFile", urdf_path=urdf_path,
                                              import_config=cfg, dest_path=usd_path)
    from pxr import Usd

    stage = Usd.Stage.Open(usd_path)
    n = force_joint_drives(stage)
    stage.GetRootLayer().Save()
    print(f"[go2nav] URDF -> USD: {usd_path} (root {prim_path}, ok={ok}, {n} force drives)")
    return usd_path


def force_joint_drives(stage, root: str = "/") -> int:
    """The URDF importer authors 'acceleration' drives; the locomotion policy was trained with
    'force' (torque) PD drives. Gains themselves are set at runtime via the articulation view."""
    from pxr import UsdPhysics

    n = 0
    for prim in stage.Traverse():
        if str(prim.GetPath()).startswith(root) and prim.IsA(UsdPhysics.RevoluteJoint):
            drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
            drive.CreateTypeAttr().Set("force")
            n += 1
    return n


def rpy_to_quat_wxyz(rpy) -> Tuple[float, float, float, float]:
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = (math.cos(r / 2), math.sin(r / 2), math.cos(p / 2), math.sin(p / 2),
                              math.cos(y / 2), math.sin(y / 2))
    return (cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy)

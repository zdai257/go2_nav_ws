"""Go2 A-to-B navigation in Isaac Sim (digital twin + simulated lidar + global planner + RL locomotion).

    ./run_isaac.sh scripts/isaac_nav.py                                  # GUI, defaults (A*, lidar map, RL policy)
    ./run_isaac.sh scripts/isaac_nav.py --planner hybrid_astar --follower holonomic
    ./run_isaac.sh scripts/isaac_nav.py --config scenarios/dense.yaml --seed 3
    ./run_isaac.sh scripts/isaac_nav.py --spawn-during-run 5 --locomotion kinematic
    ./run_isaac.sh scripts/isaac_nav.py --headless --close-on-finish     # batch / CI

All go2nav CLI options (see --help) are shared with scripts/sim2d_nav.py, so the same
seed + config reproduces the same obstacle field in both simulators.
"""

import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from go2nav.core.cli import add_common_args, config_from_args, print_registries
from go2nav.core.config import resolve_path, set_by_path

parser = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
parser.add_argument("--headless", action="store_true")
parser.add_argument("--locomotion", choices=["policy", "kinematic"], default=None)
parser.add_argument("--robot-source", choices=["urdf", "nvidia"], default=None, help="Go2 model: workspace URDF or NVIDIA USD")
parser.add_argument("--no-lidar", action="store_true", help="disable lidar (forces ground-truth map)")
parser.add_argument("--camera", action="store_true", help="enable the front RGB-D camera and save frames")
parser.add_argument("--max-steps", type=int, default=None, help="stop after N render steps")
parser.add_argument("--close-on-finish", action="store_true")
args, _unknown = parser.parse_known_args()
cfg = config_from_args(args)
if args.list:
    print_registries()
    sys.exit(0)
for flag, key in ((args.locomotion, "locomotion.backend"), (args.robot_source, "isaac.robot_source"),
                  (args.max_steps, "isaac.max_steps")):
    if flag is not None:
        set_by_path(cfg, key, flag)
if args.headless:
    set_by_path(cfg, "isaac.headless", True)
if args.close_on_finish:
    set_by_path(cfg, "isaac.close_on_finish", True)
if args.no_lidar:
    set_by_path(cfg, "perception.lidar.enabled", False)
    set_by_path(cfg, "map.source", "ground_truth")
if args.camera:
    set_by_path(cfg, "perception.camera.enabled", True)

from go2nav.isaac.app import launch

app = launch(headless=cfg["isaac"]["headless"])

import numpy as np
from isaacsim.core.api import World
from isaacsim.core.utils.viewports import set_camera_view

from go2nav.core.types import Pose2D, VelocityCommand
from go2nav.isaac.go2_asset import ensure_go2_usd
from go2nav.isaac.go2_robot import Go2Robot
from go2nav.isaac.locomotion import build_locomotion
from go2nav.isaac.scene import DebugOverlay, IsaacScene
from go2nav.isaac.sensors import Go2FrontCamera, Go2Lidar
from go2nav.navigation import NavStatus
from go2nav.scenario import build_scenario
from go2nav.sim2d.sim import footprint_clearance

icfg = cfg["isaac"]
sc = build_scenario(cfg)
nav = sc.navigator
print(f"[go2nav] {len(sc.world.obstacles)} obstacles (coverage {sc.world.coverage():.1%}), "
      f"start {sc.start}, goal {sc.goal}, planner={sc.planner.name}, follower={sc.follower.registry_name}, "
      f"map={cfg['map']['source']}, locomotion={cfg['locomotion']['backend']}")

world = World(stage_units_in_meters=1.0, physics_dt=icfg["physics_dt"], rendering_dt=icfg["rendering_dt"])
scene = IsaacScene(world, sc.world)
scene.build()
robot = Go2Robot(world, ensure_go2_usd(cfg), sc.start)
loco = build_locomotion(robot, cfg)
lidar = Go2Lidar(robot.base_path, cfg) if cfg["perception"]["lidar"]["enabled"] else None
camera = Go2FrontCamera(robot.base_path, cfg) if cfg["perception"]["camera"]["enabled"] else None
overlay = DebugOverlay() if icfg.get("draw_debug", True) else None

world.reset()
loco.initialize()
if camera:
    camera.initialize()

physics_dt = icfg["physics_dt"]
nav_every = max(1, round(icfg["control_dt"] / physics_dt))
lidar_every = max(1, round(1.0 / (cfg["perception"]["lidar"]["rate_hz"] * physics_dt)))
settle_steps = int(1.0 / physics_dt) if cfg["locomotion"]["backend"] == "policy" else 1
cm = cfg["map"]["costmap"]
state = {"step": 0, "cmd": VelocityCommand.zero(), "t_nav0": None, "collisions": 0, "in_coll": False,
         "min_clear": math.inf, "fell": False, "spawned": []}
use_lidar_map = cfg["map"]["source"] == "lidar" and lidar is not None


def on_physics_step(step_size: float) -> None:
    k = state["step"]
    state["step"] = k + 1
    if k < settle_steps:  # let the robot stand up before navigating
        loco.step(VelocityCommand.zero(), step_size)
        return
    t = (k - settle_steps) * physics_dt
    if state["t_nav0"] is None:
        state["t_nav0"] = t
        nav.set_goal(sc.goal)
    if lidar is not None and k % lidar_every == 0:
        scan = lidar.scan(robot.pose2d())
        if scan is not None and use_lidar_map:
            sc.map.update_scan(*scan)
    if k % nav_every == 0 and not nav.finished:
        pose = robot.pose2d()
        if sc.spawner is not None:
            ob = sc.spawner.maybe_spawn(t, pose, nav.path, sc.world)
            if ob is not None:
                scene.sync()
                state["spawned"].append((t, ob))
                print(f"[go2nav] t={t:.1f}s spawned {ob.kind} at ({ob.x:.2f}, {ob.y:.2f})")
        state["cmd"] = nav.step(pose, t, nav_every * physics_dt)
        clr = footprint_clearance(sc.world, pose, cm["footprint_circles"], 0.12)
        state["min_clear"] = min(state["min_clear"], clr)
        if clr < 0 and not state["in_coll"]:
            state["collisions"] += 1
        state["in_coll"] = clr < 0
        if cfg["locomotion"]["backend"] == "policy" and robot.has_fallen():
            state["fell"] = True
            nav._fail(t, "robot fell")
    loco.step(state["cmd"] if not nav.finished else VelocityCommand.zero(), step_size)


world.add_physics_callback("go2nav_step", callback_fn=on_physics_step)


def draw_overlay():
    overlay.clear()
    overlay.goal_marker(sc.goal.x, sc.goal.y, sc.goal.yaw)
    if nav.path is not None:
        overlay.polyline(nav.path[:, :2], 0.05, (0.1, 0.6, 1.0, 1.0), 4.0)
    if len(nav.trajectory) > 1:
        tr = np.array([p[1:3] for p in nav.trajectory[:: max(1, len(nav.trajectory) // 400)]])
        overlay.polyline(tr, 0.04, (0.2, 1.0, 0.3, 1.0), 2.5)
    if lidar is not None and lidar.last_points_world is not None:
        overlay.points(lidar.last_points_world, (1.0, 0.9, 0.1, 1.0), 3.0, 2500)


out_dir = resolve_path(cfg["output"]["dir"])
os.makedirs(out_dir, exist_ok=True)
# always render: the PhysX lidar and camera only update during the app update (headless too)
render = True
render_step, finished_at = 0, None
cam_eye = None
wall0 = time.time()
while app.is_running():
    world.step(render=render)
    render_step += 1
    if not icfg["headless"] and overlay is not None and render_step % 5 == 0:
        draw_overlay()
    if not icfg["headless"] and icfg.get("follow_camera", True):
        p, _ = robot.world_pose()
        pose = robot.pose2d()
        target = np.array([p[0], p[1], 0.3])
        eye = target + np.array([-4.0 * math.cos(pose.yaw), -4.0 * math.sin(pose.yaw), 3.0])
        cam_eye = eye if cam_eye is None else 0.95 * cam_eye + 0.05 * eye
        set_camera_view(eye=cam_eye, target=target, camera_prim_path="/OmniverseKit_Persp")
    if camera is not None and render_step % 50 == 0:
        rgb = camera.rgb()
        if rgb is not None and rgb.size:
            from PIL import Image

            os.makedirs(os.path.join(out_dir, "camera"), exist_ok=True)
            Image.fromarray(rgb[..., :3]).save(os.path.join(out_dir, "camera", f"rgb_{render_step:06d}.png"))
    if nav.finished and finished_at is None:
        finished_at = render_step
        stats = nav.stats()
        stats.update(collisions=state["collisions"], min_clearance_m=round(state["min_clear"], 3),
                     fell=state["fell"], locomotion=cfg["locomotion"]["backend"], simulator="isaac",
                     wall_time_s=round(time.time() - wall0, 1))
        for e in nav.events:
            print(f"  [{e.t:6.2f}s] {e.kind:12s} {e.message}")
        print(json.dumps(stats, indent=2))
        tag = f"isaac_{sc.planner.name}_{cfg['locomotion']['backend']}"
        with open(os.path.join(out_dir, f"{tag}.json"), "w") as f:
            json.dump(stats, f, indent=2)
        try:
            import matplotlib

            matplotlib.use("Agg")
            from go2nav.sim2d.sim import EpisodeResult
            from go2nav.sim2d.viz import plot_episode

            res = EpisodeResult(stats, sc, state["collisions"], state["min_clear"], 0.0, state["spawned"])
            print(f"[go2nav] figure: {plot_episode(res, os.path.join(out_dir, tag + '.png'))}")
        except Exception as exc:  # matplotlib is optional inside Isaac
            print(f"[go2nav] (plot skipped: {exc})")
        if icfg.get("close_on_finish") or icfg["headless"]:
            break
    if icfg.get("max_steps") and render_step >= icfg["max_steps"]:
        print(f"[go2nav] max_steps reached; status={nav.status.value}")
        break
app.close()

"""Health check of the Isaac side: Go2 stands, tracks velocity commands, lidar returns data.

    ./run_isaac.sh scripts/isaac_smoke_test.py                 # headless
    ./run_isaac.sh scripts/isaac_smoke_test.py --gui --locomotion kinematic
"""

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

parser = argparse.ArgumentParser()
parser.add_argument("--gui", action="store_true")
parser.add_argument("--locomotion", default="policy", choices=["policy", "kinematic"])
parser.add_argument("--robot-source", default="urdf", choices=["urdf", "nvidia"])
args, _ = parser.parse_known_args()

from go2nav.isaac.app import launch

app = launch(headless=not args.gui)

import numpy as np
from isaacsim.core.api import World

from go2nav.core.config import load_config
from go2nav.core.types import Pose2D, VelocityCommand
from go2nav.isaac.go2_asset import ensure_go2_usd
from go2nav.isaac.go2_robot import Go2Robot
from go2nav.isaac.locomotion import build_locomotion
from go2nav.isaac.scene import IsaacScene
from go2nav.isaac.sensors import Go2Lidar
from go2nav.world import Obstacle, ObstacleWorld

cfg = load_config()
cfg["locomotion"]["backend"] = args.locomotion
cfg["isaac"]["robot_source"] = args.robot_source
ow = ObstacleWorld(bounds=(-6, 6, -6, 6))
ow.add(Obstacle("cube", 2.5, 0.0, 0.0, (0.6, 0.6), 0.6))
ow.add(Obstacle("cylinder", 0.0, 2.0, 0.0, (0.3,), 1.2))
ow.add(Obstacle("pillar", -2.0, -1.5, 0.3, (0.3, 0.3), 2.5))

world = World(stage_units_in_meters=1.0, physics_dt=0.005, rendering_dt=0.02)
IsaacScene(world, ow).build()
robot = Go2Robot(world, ensure_go2_usd(cfg), Pose2D(0, 0, 0))
loco = build_locomotion(robot, cfg)
lidar = Go2Lidar(robot.base_path, cfg)
world.reset()
loco.initialize()
print("[smoke] dof order:", robot.dof_names)

cmd = {"v": VelocityCommand.zero()}
world.add_physics_callback("loco", lambda dt: loco.step(cmd["v"], dt))


def run(seconds, v: VelocityCommand, label):
    cmd["v"] = v
    p0 = robot.pose2d()
    vs = []
    # render=True advances rendering_dt (4 physics steps) and updates the PhysX lidar
    for _ in range(int(seconds / 0.02)):
        world.step(render=True)
        lin_b, ang_b = robot.base_velocity_body()
        vs.append([lin_b[0], lin_b[1], ang_b[2]])
    p1 = robot.pose2d()
    z = robot.world_pose()[0][2]
    m = np.mean(vs[len(vs) // 2:], axis=0)
    print(f"[smoke] {label:22s} cmd=({v.vx:+.2f},{v.vy:+.2f},{v.wz:+.2f}) measured=({m[0]:+.2f},{m[1]:+.2f},{m[2]:+.2f}) "
          f"base_z={z:.3f} fallen={robot.has_fallen()} moved={math.hypot(p1.x - p0.x, p1.y - p0.y):.2f}m "
          f"dyaw={math.degrees(p1.yaw - p0.yaw):+.0f}deg")


run(2.0, VelocityCommand(0, 0, 0), "stand")
run(3.0, VelocityCommand(0.6, 0, 0), "forward 0.6")
run(3.0, VelocityCommand(0, 0, 0.8), "turn 0.8")
run(3.0, VelocityCommand(0, 0.3, 0), "strafe 0.3")
run(2.0, VelocityCommand(0, 0, 0), "stop")

scan = lidar.scan(robot.pose2d())
if scan is None:
    print("[smoke] lidar: NO DATA")
else:
    o, hits, free = scan
    print(f"[smoke] lidar origin {np.round(o, 2)}, obstacle hits {len(hits)}, free endpoints {len(free)}")
    for ob in ow.obstacles:
        d = np.hypot(hits[:, 0] - ob.x, hits[:, 1] - ob.y) - ob.radius
        print(f"[smoke]   {ob.kind:8s} at ({ob.x:+.1f},{ob.y:+.1f}): {(d < 0.15).sum()} hits on its surface")
    stray = np.ones(len(hits), dtype=bool)
    for ob in ow.obstacles:
        stray &= np.hypot(hits[:, 0] - ob.x, hits[:, 1] - ob.y) - ob.radius >= 0.15
    print(f"[smoke]   stray hits (not on any obstacle): {stray.sum()}")
app.close()

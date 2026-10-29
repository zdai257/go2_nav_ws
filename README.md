# go2_nav_ws — Unitree Go2 global path planning among obstacles (Isaac Sim 4.5)

A research workspace for **A-to-B navigation of the Unitree Go2 in an (x, y, yaw) action space**:
a Go2 digital twin (URDF → USD) with a simulated lidar, a randomisable obstacle world
(pillars / cubes / cylinders), swappable object-oriented global planners, and a body-velocity
interface `(vx, vy, wz)` — the same abstraction as Go2's sport-mode `Move()` — so joint-level
control stays hidden behind a pretrained RL gait.

The same planners, configs and seeds run in two simulators:

| | `scripts/sim2d_nav.py` (2D) | `scripts/isaac_nav.py` (Isaac Sim) |
|---|---|---|
| robot | kinematic unicycle/holonomic base | Go2 articulation (URDF twin) walking with an RL policy, or kinematic |
| perception | 2D ray-cast lidar | PhysX 3D lidar (360°×30°, 16 rows) on the `nav_lidar` frame |
| speed | ~1 s per episode | ~3× slower than real time headless (RTX 3080) |
| use it for | developing / benchmarking planners | validating on the walking robot |

---

## 1. Layout

```
go2_nav_ws/                       (also reachable as ~/isaacsim/go2_nav_ws)
├── run_isaac.sh                  launcher: Isaac python + this workspace on PYTHONPATH
├── configs/default.yaml          every knob (task, world, map, planner, follower, locomotion, isaac)
├── configs/scenarios/*.yaml      presets: sparse, dense, pillar_forest, cluttered_narrow, dynamic_spawn
├── assets/go2/urdf/go2.urdf      digital twin (Unitree go2_description + nav_lidar frame)
├── assets/go2/usd/               converted USD (auto-generated, force-drive joints)
├── assets/policies/go2_rough/    Isaac Lab Go2 velocity policy (numpy, default)
├── assets/policies/go2_flat/     Isaac Lab flat policy (weaker; kept for comparison)
├── go2nav/
│   ├── core/        Pose2D, VelocityCommand, Registry, config + shared CLI
│   ├── world/       Obstacle, ObstacleWorld, ObstacleGenerator (count / density / mix)
│   ├── perception/  OccupancyGrid, Costmap2D (inflation, clearance, footprint), GroundTruthMap, LidarMapper
│   ├── planning/    GlobalPlanner base + A*, Dijkstra, Greedy, Theta*, Hybrid A*, RRT, RRT*
│   ├── control/     PathFollower base + pure_pursuit, holonomic   → (vx, vy, wz)
│   ├── navigation/  Navigator state machine (plan, follow, replan on block/stuck/off-path)
│   ├── sim2d/       kinematic simulator + plotting
│   └── isaac/       app launch, Go2 asset/robot, locomotion backends, lidar/camera, scene
├── plugins/my_planner_template.py   how to add your own planner
├── scripts/                     entry points (below)
└── tests/                       pytest, simulator-free
```

Data flow every control tick (50 Hz in Isaac, 20 Hz in 2D):

```
lidar scan ─► LidarMapper (log-odds grid) ─► Costmap2D ─► GlobalPlanner.plan(start, goal) ─► path (x, y, yaw)
                                                              ▲ replan: blocked / stuck / off-path
pose (x, y, yaw) ─► PathFollower.compute ─► (vx, vy, wz) ─► Locomotion (RL policy | kinematic) ─► Go2
```

## 2. Setup (done once; already done on this machine)

```bash
cd ~/go2_nav_ws
conda activate go2nav                           # dedicated env for the 2D side (conda base also works)
pip install -r requirements.txt                 # only needed in a fresh env
./run_isaac.sh scripts/convert_urdf_to_usd.py   # URDF -> USD (auto on first Isaac run as well)
python scripts/export_go2_policy.py --task rough   # needs torch; policy npz already exported
python -m pytest tests -q                       # 12 tests, ~3 s
```

**Note on this Isaac install:** numpy, torch, PIL, typing_extensions and packaging had been removed
from Isaac Sim 4.5's bundled packages (`omni.kit.pip_archive` was modified on 2026-04-02, most likely by a
pip run from the Isaac Lab env). So `isaacsim.core.api` could not even import. Rather than touching
`~/isaacsim` or `env_isaaclab`, `run_isaac.sh` installs `isaac_requirements.txt` into the
workspace-private `.isaac_pydeps/` and puts it on `PYTHONPATH`. If you repair/reinstall Isaac Sim you can
delete that folder. (`env_isaaclab` is also missing numpy/yaml/scipy/matplotlib, so Isaac Lab training
will need `pip install` there first.)

## 3. Quick start

```bash
# health check: Go2 stands, tracks vx / wz / vy commands, lidar sees the 3 test obstacles
./run_isaac.sh scripts/isaac_smoke_test.py            # add --gui to watch, --locomotion kinematic

# A-to-B in Isaac Sim (GUI, follow camera, path + trajectory + lidar points drawn)
./run_isaac.sh scripts/isaac_nav.py
./run_isaac.sh scripts/isaac_nav.py --headless        # batch: writes outputs/isaac_<planner>_<backend>.json/.png

# the same in 2D, in about a second
python scripts/sim2d_nav.py --show
```

Verified results (seed 0, start (−8, −8) → goal (8, 8), lidar mapping, RL gait):

| scenario | planner / follower | sim time | replans | collisions | min clearance |
|---|---|---|---|---|---|
| sparse (19 obstacles) | A* / pure_pursuit | 32.7 s | 1 | 0 | 0.24 m |
| default (39) | A* / pure_pursuit | 32.9 s | 4 | 0 | 0.21 m |
| dense (145, 12 % coverage) | Hybrid A* / holonomic | 55.5 s | 23 | 0 | 0.14 m |
| dynamic_spawn (6 dropped on the path) | A* / pure_pursuit | 39.8 s | 8 | 0 | 0.22 m |

## 4. Obstacles: more, denser, different

All world options are shared by `sim2d_nav.py`, `isaac_nav.py`, `compare_planners.py` and `benchmark_planners.py`:

```bash
--num-obstacles 60                     # exact count
--density 0.08                         # target footprint coverage (overrides count)
--mix pillar=3,cylinder=1,cube=1       # relative frequency of each kind
--seed 7                               # different random layout (same seed = same layout in 2D and Isaac)
--arena 30                             # 30 x 30 m arena
--spawn-during-run 5                   # drop obstacles onto the path while walking (tests replanning)
--random-start-goal                    # random A and B in free space
--start -8,-8,0 --goal 8,8,90d         # explicit poses (yaw in rad or with a d suffix)
--config scenarios/pillar_forest.yaml  # preset
--set world.generator.min_gap=0.5 world.generator.kind_params.cube.size=[0.4,1.0]
--save-world my.yaml / --world-file my.yaml   # freeze / reload a layout
```

Density saturates at ~8.5 % coverage with the default `min_gap: 0.7` m (~15 % at 0.35, ~21 % at 0.2).
Gaps narrower than 2 × the costmap inflation radius (~0.64 m) are impassable for the Go2, so
`world.ensure_feasible: true` re-rolls the seed until a path exists (and warns when it gives up).

## 5. Planners and followers (object-oriented, swappable)

```bash
python scripts/sim2d_nav.py --list                         # registered names
--planner astar | dijkstra | greedy_bfs | theta_star | hybrid_astar | rrt | rrt_star
--follower pure_pursuit | holonomic
--set planner.params.hybrid_astar.allow_strafe=true planner.common.smooth=false
python scripts/compare_planners.py --planners astar theta_star hybrid_astar rrt_star --show
python scripts/benchmark_planners.py --planners astar theta_star hybrid_astar rrt_star \
       --densities 0.02 0.04 0.06 0.08 --seeds 0 1 2 3 4 --map-source lidar
```

* Grid planners (A*, Dijkstra, Greedy, Lazy Theta*) work on the inflated costmap; yaw is assigned along the path.
* **Hybrid A\*** searches (x, y, yaw) directly with Go2 motion primitives (arcs, reverse, rotate-in-place,
  optional strafe), the 3-circle footprint, and an analytic rotate–straight–rotate goal shot. It is the
  natural starting point for an SE(2) / yaw-aware research planner.
* RRT / RRT* sample in continuous space with footprint collision checks.

**Your own planner**: copy `plugins/my_planner_template.py`, implement `_plan(start, goal, costmap)`, register
a name, and run it anywhere with `--plugin path/to/file.py --planner your_name`. Its constructor kwargs come
from `planner.params.your_name` in YAML. Followers (`go2nav.control.FOLLOWERS`), map providers and
locomotion backends use the same registry pattern.

## 6. Robot, perception and locomotion

* `assets/go2/urdf/go2.urdf` is generated by `scripts/prepare_go2_urdf.py` from Unitree's `go2_description`.
  It keeps the sensor frames (`imu`, `radar` = Go2's L1 lidar, `front_camera`) and adds a `nav_lidar` frame.
  The URDF importer merges fixed joints, so sensors are mounted on `base` with the URDF offsets.
* Lidar: `perception.lidar` (FOV, resolution, range, rate, `z_min`/`z_max` obstacle band, `mount`).
  Returns from the robot's own body are removed, and returns in the height band become obstacles while every beam
  carves free space (`map.source: lidar`). `map.source: ground_truth` gives a perfect map.
  The PhysX lidar only updates during an app update, so Isaac is always stepped with `render=True`.
* Front RGB camera: `--camera` (saves frames to `outputs/camera/`).
* Locomotion: `--locomotion policy` (default) or `kinematic` (ideal tracking, isolates planning from gait error).
  * The policy is Isaac Lab's `Isaac-Velocity-Rough-Unitree-Go2-v0` checkpoint (1500 iterations), re-implemented in
    numpy. It stands at ~0.33 m and tracks vx ≈ 1.0, vy ≈ 0.9, wz ≈ 0.65 of the command; its small-yaw deadband
    is why `follower.common.limits.min_turn_rate: 0.6` exists.
  * The flat checkpoint (`policy_dir: assets/policies/go2_flat`) was trained for only 300 iterations; it walks
    crouched and stalls on in-place turns.
  * `locomotion.actuator: dc_motor` switches from PhysX implicit PD to Isaac Lab's explicit DC-motor torques.
* `--robot-source nvidia` uses NVIDIA's Go2 USD instead of the URDF twin (they behave the same).

## 7. Roadmap

**Phase 0 — orient (day 1)**
1. `./run_isaac.sh scripts/isaac_smoke_test.py --gui` and `./run_isaac.sh scripts/isaac_nav.py` — watch it work.
2. Read `configs/default.yaml` top to bottom; it is the map of every knob.
3. `python scripts/compare_planners.py --show` on a few seeds/densities to build intuition.

**Phase 1 — planning baselines in 2D (fast loop)**
1. `benchmark_planners.py` across density × seed with `--map-source lidar`; record success rate, path length,
   plan time and minimum clearance. This becomes your baseline table.
2. Stress cases: `cluttered_narrow`, `pillar_forest`, `--spawn-during-run`, `--random-start-goal`.
3. Tune the costmap (`inflation_radius`, `cost_falloff`, footprint circles). It is the biggest lever on
   safety versus path length.

**Phase 2 — your planner**
1. Start from `plugins/my_planner_template.py`. If yaw matters to your method, subclass or adapt `HybridAStarPlanner`
   (primitives, heuristic, cost terms) — it already plans in (x, y, yaw).
2. Add a test to `tests/` and a benchmark run. Compare against A* / Theta* / Hybrid A* / RRT* with the same seeds.
3. Ideas: kinodynamic lattice with Go2 velocity/acceleration limits, clearance-aware or risk-aware costs,
   time-optimal (x, y, yaw) trajectories (using lateral vy), anytime/incremental replanning (D* Lite, ARA*),
   learned heuristics.

**Phase 3 — validate on the walking robot (Isaac)**
1. Run the same seeds with `isaac_nav.py --headless` for the RL gait and the `kinematic` backend. The gap between
   the two is the cost of real locomotion (tracking error, slower turns, sway) that your planner must tolerate.
2. Add local reactivity if needed: a DWA / MPPI / MPC follower registered in `go2nav.control` (same
   `(vx, vy, wz)` output), or a velocity-obstacle layer for moving obstacles.
3. Better locomotion: train your own Go2 policy in Isaac Lab (`env_isaaclab`), e.g. rough terrain with more
   yaw-tracking reward, export it with `scripts/export_go2_policy.py --checkpoint ... --out-dir assets/policies/<name>`,
   and set `locomotion.policy_dir`.

**Phase 4 — realism and sim-to-real**
1. Sensors: switch to an RTX lidar matching the Go2 L1 (or Mid-360) and add noise/dropouts; build the map from
   the depth camera; use odometry drift instead of ground-truth pose (add a localisation layer).
2. Environments: rough terrain / stairs (rough policy + real height scan), moving obstacles, indoor USD scenes.
3. ROS 2 bridge (`isaacsim.ros2.bridge`): publish `/scan`, `/odom`, `/tf` and subscribe `/cmd_vel` so the same
   planner can run against Nav2 baselines and later on the real robot.
4. Real Go2: the follower output `(vx, vy, wz)` maps 1:1 to `unitree_sdk2` `SportClient.Move(vx, vy, vyaw)`.
   Respect the same limits in `follower.common.limits`.

## 8. Troubleshooting

* `fell=True` / the robot sinks: make sure the USD has force drives (`./run_isaac.sh scripts/convert_urdf_to_usd.py --force`)
  and `policy_dir` points at `go2_rough`.
* Lidar shows nothing / garbage points: the simulation must be stepped with `render=True` (as the scripts do).
* "no feasible layout": density too high for the gaps — lower `--density`, raise `world.generator.min_gap`.
* Always launch Isaac via `./run_isaac.sh` (it clears conda variables and sets `PYTHONPATH`).

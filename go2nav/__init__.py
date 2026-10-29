"""go2nav: global path planning / navigation research stack for the Unitree Go2.

Layering (only ``go2nav.isaac`` depends on Isaac Sim; everything else is numpy-only):

    core        types (Pose2D, VelocityCommand), registry, config
    world       obstacle primitives + random/dense obstacle field generator
    perception  occupancy grids, costmaps, lidar mapping
    planning    global planners (A*, Dijkstra, Theta*, Hybrid A*, RRT*) + post-processing
    control     path followers that turn a path into (vx, vy, wz) commands
    navigation  Navigator: plan -> follow -> replan state machine
    sim2d       lightweight kinematic simulator for fast planner iteration / benchmarks
    isaac       Isaac Sim backend: Go2 digital twin, locomotion, lidar, scene
"""

__version__ = "0.1.0"

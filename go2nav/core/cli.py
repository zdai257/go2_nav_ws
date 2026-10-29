"""Shared command-line interface for every entry point (2D sim, benchmark, Isaac)."""

from __future__ import annotations

import argparse
import importlib.util
import os
from typing import Dict, List, Optional

from .config import apply_overrides, load_config, set_by_path


def add_common_args(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    g = p.add_argument_group("go2nav")
    g.add_argument("--config", default=None, help="scenario YAML merged on top of configs/default.yaml")
    g.add_argument("--set", nargs="*", default=[], metavar="KEY=VAL", help="dotted overrides, e.g. planner.name=rrt_star")
    g.add_argument("--plugin", action="append", default=[], help="python file(s) registering custom planners/followers")
    g.add_argument("--planner", default=None, help="planner name (see --list)")
    g.add_argument("--follower", default=None, help="path follower name")
    g.add_argument("--map-source", default=None, choices=["ground_truth", "lidar"], help="costmap source")
    g.add_argument("--start", default=None, help="x,y[,yaw]  (yaw in rad, or e.g. 90d)")
    g.add_argument("--goal", default=None, help="x,y[,yaw]")
    g.add_argument("--random-start-goal", action="store_true", help="sample start/goal in free space")
    g.add_argument("--seed", type=int, default=None, help="obstacle generator seed")
    g.add_argument("--num-obstacles", type=int, default=None, help="obstacle count")
    g.add_argument("--density", type=float, default=None, help="obstacle footprint coverage (0-1), overrides count")
    g.add_argument("--mix", default=None, help="kind weights, e.g. pillar=2,cube=1,cylinder=1")
    g.add_argument("--arena", type=float, default=None, help="square arena side length [m]")
    g.add_argument("--world-file", default=None, help="load a saved obstacle layout YAML")
    g.add_argument("--spawn-during-run", type=int, default=None, help="obstacles dropped on the path while walking")
    g.add_argument("--list", action="store_true", help="list registered planners / followers and exit")
    return p


def load_plugins(paths: List[str]) -> None:
    for path in paths:
        path = os.path.abspath(path)
        spec = importlib.util.spec_from_file_location(os.path.splitext(os.path.basename(path))[0], path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)


def config_from_args(args: argparse.Namespace) -> Dict:
    load_plugins(args.plugin)
    cfg = load_config(args.config)
    shortcuts = {
        "planner": "planner.name",
        "follower": "follower.name",
        "map_source": "map.source",
        "seed": "world.generator.seed",
        "num_obstacles": "world.generator.count",
        "density": "world.generator.density",
        "world_file": "world.file",
        "spawn_during_run": "world.runtime_spawn.count",
    }
    for attr, key in shortcuts.items():
        v = getattr(args, attr, None)
        if v is not None:
            set_by_path(cfg, key, v)
    if args.mix:
        set_by_path(cfg, "world.generator.mix", {k: float(v) for k, v in (kv.split("=") for kv in args.mix.split(","))})
    if args.arena:
        h = args.arena / 2.0
        set_by_path(cfg, "world.bounds", [-h, h, -h, h])
    if args.random_start_goal:
        set_by_path(cfg, "task.random_start_goal", True)
    from .types import Pose2D

    for k in ("start", "goal"):
        v = getattr(args, k, None)
        if v:
            set_by_path(cfg, f"task.{k}", Pose2D.parse(v).as_array().tolist())
    apply_overrides(cfg, args.set)
    return cfg


def print_registries() -> None:
    from ..control import FOLLOWERS
    from ..planning import PLANNERS

    print("planners :", ", ".join(PLANNERS.names()))
    print("followers:", ", ".join(FOLLOWERS.names()))

"""Fast regression tests for the simulator-independent stack (no Isaac Sim needed).

    python -m pytest tests -q
"""

import copy
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from go2nav.core.config import load_config  # noqa: E402
from go2nav.core.types import Pose2D  # noqa: E402
from go2nav.perception import build_map_provider  # noqa: E402
from go2nav.planning import PLANNERS, build_planner  # noqa: E402
from go2nav.scenario import build_scenario  # noqa: E402
from go2nav.sim2d.sim import run_episode  # noqa: E402
from go2nav.world import Obstacle, ObstacleWorld, build_world  # noqa: E402


@pytest.fixture(scope="module")
def cfg():
    c = load_config()
    c["map"]["source"] = "ground_truth"
    return c


@pytest.fixture(scope="module")
def scenario(cfg):
    return build_scenario(copy.deepcopy(cfg))


def test_obstacle_sdf_and_raycast():
    w = ObstacleWorld(bounds=(-5, 5, -5, 5))
    w.add(Obstacle("cylinder", 2.0, 0.0, 0.0, (0.5,), 1.0))
    w.add(Obstacle("cube", -2.0, 0.0, 0.0, (1.0, 1.0), 1.0))
    sdf = w.sdf(np.array([[0.0, 0.0], [2.0, 0.0], [-2.0, 0.0]]))
    assert sdf[0] == pytest.approx(1.5, abs=1e-6)
    assert sdf[1] < 0 and sdf[2] < 0
    r = w.raycast(np.array([0.0, 0.0]), np.array([0.0, math.pi]), 8.0)
    assert r[0] == pytest.approx(1.5, abs=1e-3) and r[1] == pytest.approx(1.5, abs=1e-3)


@pytest.mark.parametrize("count", [10, 40])
def test_generator_count_and_keepout(cfg, count):
    wcfg = copy.deepcopy(cfg["world"])
    wcfg["generator"].update(count=count, density=None, seed=3)
    start, goal = Pose2D(-8, -8, 0), Pose2D(8, 8, 0)
    w = build_world(wcfg, start, goal)
    assert len(w.obstacles) == count
    keep = wcfg.get("keepout_radius", 1.2)
    assert w.sdf(np.array([start.xy, goal.xy])).min() >= keep - 1e-6


def test_density_increases_coverage(cfg):
    cov = []
    for density in (0.02, 0.06):
        wcfg = copy.deepcopy(cfg["world"])
        wcfg["generator"].update(density=density, seed=0)
        cov.append(build_world(wcfg, Pose2D(-8, -8, 0), Pose2D(8, 8, 0)).coverage())
    assert cov[0] < cov[1]
    assert cov[1] == pytest.approx(0.06, abs=0.015)


@pytest.mark.parametrize("name", sorted(PLANNERS.names()))
def test_every_planner_finds_collision_free_path(cfg, scenario, name):
    pcfg = copy.deepcopy(cfg["planner"])
    pcfg["name"] = name
    planner = build_planner(pcfg)
    costmap = build_map_provider(cfg["map"], scenario.world).costmap()
    res = planner.plan(scenario.start, scenario.goal, costmap)
    assert res.success, f"{name}: {res.message}"
    assert np.linalg.norm(res.path[0, :2] - scenario.start.xy) < 0.3
    assert np.linalg.norm(res.path[-1, :2] - scenario.goal.xy) < 0.3
    free, _ = costmap.path_free(res.path, use_footprint=False)
    assert free
    straight = scenario.start.distance_to(scenario.goal)
    assert straight <= res.length < 2.0 * straight


def test_closed_loop_episode_with_lidar_map():
    c = load_config()
    res = run_episode(c)
    assert res.stats["success"], res.stats
    assert res.collisions == 0

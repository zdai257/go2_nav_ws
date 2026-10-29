"""Plan once with several planners on the same (ground-truth) map and overlay the paths.

    python scripts/compare_planners.py --density 0.15 --seed 3
    python scripts/compare_planners.py --planners astar theta_star hybrid_astar rrt_star
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import matplotlib

matplotlib.use("Agg")

from go2nav.core.cli import add_common_args, config_from_args
from go2nav.core.config import resolve_path


def main():
    p = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    p.add_argument("--planners", nargs="*", default=None, help="default: all registered")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    cfg = config_from_args(args)

    from go2nav.perception import GroundTruthMap
    from go2nav.planning import PLANNERS, build_planner
    from go2nav.scenario import build_scenario
    from go2nav.sim2d.viz import plot_plans

    sc = build_scenario(cfg)
    gt = GroundTruthMap(sc.world, resolution=cfg["map"]["resolution"], costmap_params=cfg["map"]["costmap"])
    cm = gt.costmap()
    names = args.planners or PLANNERS.names()
    results = {}
    for n in names:
        pc = dict(cfg["planner"])
        pc["name"] = n
        r = build_planner(pc).plan(sc.start, sc.goal, cm)
        results[n] = r
        print(f"{n:18s} {'ok ' if r.success else 'FAIL'} L={r.length:6.2f} m  t={r.planning_time * 1e3:8.1f} ms  "
              f"exp={r.expansions:6d}  {r.message}")
    out_dir = resolve_path(cfg["output"]["dir"])
    os.makedirs(out_dir, exist_ok=True)
    out = args.out or os.path.join(out_dir, "compare_planners.png")
    g = cfg["world"]["generator"]
    plot_plans(sc.world, cm, results, sc.start, sc.goal, out,
               title=f"{len(sc.world.obstacles)} obstacles, coverage {sc.world.coverage():.1%}, seed {g.get('seed')}")
    print(f"figure: {out}")


if __name__ == "__main__":
    main()

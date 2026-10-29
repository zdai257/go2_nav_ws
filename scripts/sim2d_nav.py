"""Run one A-to-B navigation episode in the lightweight 2D simulator (no Isaac Sim needed).

Examples:
    python scripts/sim2d_nav.py                                   # defaults: A*, lidar mapping, 35 obstacles
    python scripts/sim2d_nav.py --planner hybrid_astar --follower holonomic
    python scripts/sim2d_nav.py --density 0.18 --mix pillar=3,cylinder=1 --seed 4
    python scripts/sim2d_nav.py --spawn-during-run 4 --random-start-goal --seed 11
    python scripts/sim2d_nav.py --plugin plugins/my_planner_template.py --planner potential_descent
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import matplotlib

from go2nav.core.cli import add_common_args, config_from_args, print_registries
from go2nav.core.config import resolve_path


def main():
    p = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    p.add_argument("--show", action="store_true", help="open a matplotlib window")
    p.add_argument("--out", default=None, help="output PNG (default outputs/sim2d_<planner>.png)")
    p.add_argument("--save-world", default=None, help="save the generated obstacle layout to YAML")
    args = p.parse_args()
    if not args.show:
        matplotlib.use("Agg")
    cfg = config_from_args(args)
    if args.list:
        print_registries()
        return

    from go2nav.sim2d import run_episode
    from go2nav.sim2d.viz import plot_episode

    res = run_episode(cfg)
    out_dir = resolve_path(cfg["output"]["dir"])
    os.makedirs(out_dir, exist_ok=True)
    out = args.out or os.path.join(out_dir, f"sim2d_{cfg['planner']['name']}.png")
    plot_episode(res, out, show=args.show)
    if args.save_world:
        res.scenario.world.save(args.save_world)
    for e in res.scenario.navigator.events:
        print(f"  [{e.t:6.2f}s] {e.kind:12s} {e.message}")
    print(json.dumps(res.stats, indent=2))
    print(f"figure: {out}   (wall time {res.wall_time_s:.2f}s)")


if __name__ == "__main__":
    main()

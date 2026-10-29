"""Closed-loop benchmark: planners x obstacle densities x seeds in the 2D simulator.

    python scripts/benchmark_planners.py --planners astar theta_star hybrid_astar rrt_star \
        --densities 0.02 0.04 0.06 0.08 --seeds 0 1 2 3 4 --map-source lidar

Writes outputs/benchmark.csv and outputs/benchmark.png (success rate, path length, plan time).
"""

import argparse
import copy
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import matplotlib

matplotlib.use("Agg")
import numpy as np

from go2nav.core.cli import add_common_args, config_from_args
from go2nav.core.config import resolve_path, set_by_path


def main():
    p = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    p.add_argument("--planners", nargs="*", default=["astar", "dijkstra", "theta_star", "hybrid_astar", "rrt_star"])
    p.add_argument("--densities", nargs="*", type=float, default=[0.02, 0.04, 0.06, 0.08],
                   help="footprint coverage; >~0.085 needs a smaller world.generator.min_gap")
    p.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    p.add_argument("--out", default=None, help="CSV path")
    args = p.parse_args()
    base = config_from_args(args)

    from go2nav.sim2d import run_episode

    rows = []
    for dens in args.densities:
        for seed in args.seeds:
            for name in args.planners:
                cfg = copy.deepcopy(base)
                set_by_path(cfg, "world.generator.density", dens)
                set_by_path(cfg, "world.generator.seed", seed)
                set_by_path(cfg, "planner.name", name)
                res = run_episode(cfg)
                s = dict(res.stats, density=dens, seed=seed, wall_s=round(res.wall_time_s, 2),
                         obstacles=len(res.scenario.world.obstacles))
                rows.append(s)
                print(f"d={dens:.2f} seed={seed} {name:14s} {s['status']:10s} t={s['time_s']:6.1f}s "
                      f"L={s['travelled_m']:6.2f}m plan={s['mean_plan_time_ms']}ms replans={s['num_replans']} "
                      f"coll={s['collisions']}")
    out_dir = resolve_path(base["output"]["dir"])
    os.makedirs(out_dir, exist_ok=True)
    out = args.out or os.path.join(out_dir, "benchmark.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    _plot(rows, args.planners, args.densities, out.replace(".csv", ".png"))
    print(f"csv: {out}")


def _plot(rows, planners, densities, path):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for name in planners:
        sr, L, T = [], [], []
        for d in densities:
            sel = [r for r in rows if r["planner"] == name and r["density"] == d]
            ok = [r for r in sel if r["success"]]
            sr.append(len(ok) / max(1, len(sel)))
            L.append(np.mean([r["travelled_m"] for r in ok]) if ok else np.nan)
            T.append(np.mean([r["mean_plan_time_ms"] for r in ok if r["mean_plan_time_ms"]]) if ok else np.nan)
        axes[0].plot(densities, sr, "o-", label=name)
        axes[1].plot(densities, L, "o-", label=name)
        axes[2].plot(densities, T, "o-", label=name)
    for ax, t in zip(axes, ["success rate", "travelled length [m] (successful)", "mean planning time [ms]"]):
        ax.set_xlabel("obstacle density (footprint coverage)")
        ax.set_title(t)
        ax.grid(alpha=0.3)
    axes[2].set_yscale("log")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"figure: {path}")


if __name__ == "__main__":
    main()

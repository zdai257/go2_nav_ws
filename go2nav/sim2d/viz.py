"""Matplotlib rendering of worlds, costmaps, plans and trajectories."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np


def _mpl():
    import matplotlib.pyplot as plt
    from matplotlib import patches

    return plt, patches


def draw_world(ax, world, alpha: float = 0.9):
    _, patches = _mpl()
    for ob in world.obstacles:
        if ob.shape == "circle":
            p = patches.Circle((ob.x, ob.y), ob.size[0], color=ob.color, alpha=alpha)
        else:
            sx, sy = ob.size[:2]
            c, s = math.cos(ob.yaw), math.sin(ob.yaw)
            corner = (ob.x - (c * sx - s * sy) / 2, ob.y - (s * sx + c * sy) / 2)
            p = patches.Rectangle(corner, sx, sy, angle=math.degrees(ob.yaw), color=ob.color, alpha=alpha)
        ax.add_patch(p)
    b = world.bounds
    ax.set_xlim(b[0] - 0.5, b[1] + 0.5)
    ax.set_ylim(b[2] - 0.5, b[3] + 0.5)
    ax.set_aspect("equal")


def draw_costmap(ax, costmap, show_lethal: bool = True, show_cost: bool = True):
    g = costmap.grid
    b = g.bounds
    if show_cost:
        ax.imshow(costmap.cost, origin="lower", extent=b, cmap="Greys", alpha=0.25, vmin=0, vmax=1)
    if show_lethal:
        m = np.ma.masked_where(~costmap.lethal, costmap.lethal.astype(float))
        ax.imshow(m, origin="lower", extent=b, cmap="Reds", alpha=0.25, vmin=0, vmax=1)


def draw_occupancy(ax, grid):
    d = grid.data
    img = np.where(np.isnan(d), 0.5, d)
    ax.imshow(img, origin="lower", extent=grid.bounds, cmap="Greys", alpha=0.45, vmin=0, vmax=1)


def draw_pose(ax, pose, color="k", size=0.5, label=None):
    x, y, yaw = (pose.x, pose.y, pose.yaw) if hasattr(pose, "x") else pose
    ax.arrow(x, y, size * math.cos(yaw), size * math.sin(yaw), head_width=0.18, color=color, length_includes_head=True)
    ax.plot([x], [y], "o", color=color, ms=6, label=label)


def draw_footprint(ax, pose, length=0.70, width=0.31, color="tab:green", alpha=0.35):
    _, patches = _mpl()
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    corner = (x - (c * length - s * width) / 2, y - (s * length + c * width) / 2)
    ax.add_patch(patches.Rectangle(corner, length, width, angle=math.degrees(yaw), fc=color, ec="k", lw=0.5, alpha=alpha))


def plot_episode(result, path_out: Optional[str] = None, title: Optional[str] = None, show: bool = False):
    plt, _ = _mpl()
    sc = result.scenario
    nav = sc.navigator
    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    if sc.cfg["map"]["source"] == "lidar":
        draw_occupancy(ax, sc.map.grid())
    else:
        draw_costmap(ax, sc.map.costmap())
    draw_world(ax, sc.world, alpha=0.55)
    for k, (t, ob) in enumerate(result.spawned):
        ax.annotate(f"+{t:.0f}s", (ob.x, ob.y), color="m", fontsize=8, ha="center")
    plans = [(t, r) for t, r in nav.plans if r.success]
    for i, (t, r) in enumerate(plans):
        last = i == len(plans) - 1
        ax.plot(r.path[:, 0], r.path[:, 1], "-", color="tab:blue" if last else "tab:cyan",
                lw=2.0 if last or i == 0 else 0.8, alpha=1.0 if last or i == 0 else 0.5,
                label="first plan" if i == 0 else ("last plan" if last else None))
    traj = np.array([p[1:] for p in nav.trajectory])
    if len(traj):
        ax.plot(traj[:, 0], traj[:, 1], "-", color="tab:green", lw=1.5, label="executed")
        for p in traj[:: max(1, len(traj) // 25)]:
            draw_footprint(ax, p)
    draw_pose(ax, sc.start, "tab:green", label="start")
    draw_pose(ax, sc.goal, "tab:red", label="goal")
    s = result.stats
    ax.set_title(title or f"{s['planner']} + {s['follower']} | {s['status']} | t={s['time_s']:.1f}s "
                 f"L={s['travelled_m']:.1f}m replans={s['num_replans']} coll={s['collisions']}", fontsize=10)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    if path_out:
        fig.savefig(path_out, dpi=130)
    if show:
        plt.show()
    plt.close(fig)
    return path_out


def plot_plans(world, costmap, results, start, goal, path_out: str, title: str = ""):
    """Overlay several planners' paths on one map (static comparison)."""
    plt, _ = _mpl()
    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    draw_costmap(ax, costmap)
    draw_world(ax, world, alpha=0.6)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, (name, r) in enumerate(results.items()):
        if r.success:
            ax.plot(r.path[:, 0], r.path[:, 1], "-", lw=2, color=colors[i % len(colors)],
                    label=f"{name}: {r.length:.2f} m, {r.planning_time * 1e3:.0f} ms")
        else:
            ax.plot([], [], "x", color=colors[i % len(colors)], label=f"{name}: FAILED ({r.message})")
    draw_pose(ax, start, "tab:green")
    draw_pose(ax, goal, "tab:red")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path_out, dpi=130)
    plt.close(fig)
    return path_out

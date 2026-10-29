"""Template for a bespoke global planner, loaded at runtime with ``--plugin``.

    python scripts/sim2d_nav.py  --plugin plugins/my_planner_template.py --planner potential_descent
    ./run_isaac.sh scripts/isaac_nav.py --plugin plugins/my_planner_template.py --planner potential_descent

Contract:
  * subclass ``GlobalPlanner`` and register a unique name
  * implement ``_plan(start, goal, costmap)`` -> ((N,2) xy or (N,3) xy-yaw array, info dict)
    or (None, {"message": ...}) on failure
  * constructor kwargs come from ``planner.params.<name>`` in the YAML config;
    forward ``**kw`` to ``super().__init__`` to keep the shared options
    (smooth, densify_step, yaw_mode, time_limit, goal_search_radius)
  * set ``outputs_yaw = True`` if your planner already produces feasible headings

Useful costmap API (go2nav.perception.Costmap2D):
  lethal[iy, ix] (bool), cost[iy, ix] (0..1), clearance[iy, ix] (m),
  world_to_grid(xy) / grid_to_world(ij), is_free_xy(xy), segment_free(a, b),
  footprint_free(poses_xyyaw), path_free(path)
"""

import numpy as np

from go2nav.planning import PLANNERS, GlobalPlanner


@PLANNERS.register("potential_descent")
class PotentialDescentPlanner(GlobalPlanner):
    """Toy example: attractive (goal) + repulsive (clearance gradient) potential field, with
    a "slide along the obstacle" fallback when the next step would enter the inflated zone.

    Still gets trapped in concave pockets - a deliberately weak baseline to beat.
    """

    def __init__(self, step: float = 0.1, repulse_gain: float = 1.5, influence: float = 0.9,
                 max_steps: int = 3000, **kw):
        super().__init__(**kw)
        self.step, self.k_rep, self.influence, self.max_steps = step, repulse_gain, influence, max_steps

    def _plan(self, start, goal, costmap):
        p = start.xy.copy()
        path = [p.copy()]
        eps = costmap.resolution
        side = 1.0
        for i in range(self.max_steps):
            if np.linalg.norm(goal.xy - p) < self.step:
                path.append(goal.xy)
                return np.array(path), {"expansions": i}
            att = (goal.xy - p) / max(np.linalg.norm(goal.xy - p), 1e-6)
            cx = costmap.clearance_at(np.array([p + [eps, 0], p - [eps, 0], p + [0, eps], p - [0, eps]]))
            grad = np.array([cx[0] - cx[1], cx[2] - cx[3]]) / (2 * eps)
            d = costmap.clearance_at(p[None])[0]
            v = att + self.k_rep * grad * max(0.0, self.influence - d) / self.influence
            heading = np.arctan2(v[1], v[0])
            for k in range(13):  # try rotating the step up to +-90 deg, keep sliding to the same side
                h = heading + side * np.deg2rad(15 * ((k + 1) // 2)) * (1 if k % 2 else -1) if k else heading
                q = p + self.step * np.array([np.cos(h), np.sin(h)])
                if costmap.is_free_xy(q[None])[0]:
                    if k:
                        side = 1.0 if k % 2 else -1.0
                    break
            else:
                return None, {"message": "trapped", "expansions": i}
            p = q
            path.append(p.copy())
        return None, {"message": "local minimum / step limit", "expansions": self.max_steps}

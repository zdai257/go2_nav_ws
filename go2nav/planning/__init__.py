"""Global planners. Importing this package registers all built-in planners in ``PLANNERS``."""

from .base import PLANNERS, GlobalPlanner, PlanResult
from . import grid_search, hybrid_astar, rrt_star  # noqa: F401  (registration side effects)
from .grid_search import AStarPlanner, DijkstraPlanner, GreedyBestFirstPlanner, GridSearchPlanner, LazyThetaStarPlanner
from .hybrid_astar import HybridAStarPlanner
from .rrt_star import RRTPlanner, RRTStarPlanner


def build_planner(cfg: dict) -> GlobalPlanner:
    """cfg = ``planner`` section: {name: astar, common: {...}, params: {astar: {...}, ...}}"""
    name = cfg.get("name", "astar")
    kwargs = dict(cfg.get("common", {}) or {})
    kwargs.update((cfg.get("params", {}) or {}).get(name, {}) or {})
    return PLANNERS.build(name, **kwargs)


__all__ = ["PLANNERS", "GlobalPlanner", "PlanResult", "build_planner", "GridSearchPlanner", "AStarPlanner",
           "DijkstraPlanner", "GreedyBestFirstPlanner", "LazyThetaStarPlanner", "HybridAStarPlanner",
           "RRTStarPlanner", "RRTPlanner"]

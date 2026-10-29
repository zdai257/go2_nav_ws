"""YAML config loading with dotted CLI overrides (``--set planner.name=rrt_star``)."""

from __future__ import annotations

import copy
import os
from typing import Any, Dict, Iterable

import yaml

WS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_CONFIG = os.path.join(WS_ROOT, "configs", "default.yaml")


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | None = None) -> Dict[str, Any]:
    """Load ``default.yaml`` and merge ``path`` on top. A file may also declare ``base: other.yaml``."""
    with open(DEFAULT_CONFIG) as f:
        cfg = yaml.safe_load(f)
    if path:
        cfg = deep_merge(cfg, _load_with_bases(path))
    return cfg


def _load_with_bases(path: str) -> Dict[str, Any]:
    if not os.path.isabs(path) and not os.path.exists(path):
        path = os.path.join(WS_ROOT, "configs", path)
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    base = data.pop("base", None)
    if base:
        base_path = base if os.path.isabs(base) else os.path.join(os.path.dirname(path), base)
        data = deep_merge(_load_with_bases(base_path), data)
    return data


def set_by_path(cfg: Dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def get_by_path(cfg: Dict[str, Any], dotted: str, default: Any = None) -> Any:
    node = cfg
    for k in dotted.split("."):
        if not isinstance(node, dict) or k not in node:
            return default
        node = node[k]
    return node


def apply_overrides(cfg: Dict[str, Any], overrides: Iterable[str]) -> Dict[str, Any]:
    """Apply ``key.sub=value`` strings; values are parsed as YAML (numbers, lists, bools...)."""
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must be key=value, got '{item}'")
        key, raw = item.split("=", 1)
        set_by_path(cfg, key.strip(), yaml.safe_load(raw))
    return cfg


def resolve_path(p: str) -> str:
    """Paths in configs are relative to the workspace root."""
    return p if os.path.isabs(p) else os.path.join(WS_ROOT, p)

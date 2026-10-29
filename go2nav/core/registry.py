from __future__ import annotations

from typing import Callable, Dict, Type


class Registry:
    """Name -> class registry so components can be swapped from config/CLI.

    >>> PLANNERS = Registry("planner")
    >>> @PLANNERS.register("my_planner")
    ... class MyPlanner(GlobalPlanner): ...
    >>> PLANNERS.build("my_planner", resolution=0.1)
    """

    def __init__(self, kind: str):
        self.kind = kind
        self._items: Dict[str, Type] = {}

    def register(self, name: str) -> Callable[[Type], Type]:
        def deco(cls: Type) -> Type:
            if name in self._items and self._items[name] is not cls:
                raise KeyError(f"{self.kind} '{name}' already registered by {self._items[name]}")
            self._items[name] = cls
            cls.registry_name = name
            return cls

        return deco

    def get(self, name: str) -> Type:
        if name not in self._items:
            raise KeyError(f"unknown {self.kind} '{name}'. available: {sorted(self._items)}")
        return self._items[name]

    def build(self, name: str, **kwargs):
        return self.get(name)(**kwargs)

    def names(self):
        return sorted(self._items)

    def __contains__(self, name: str) -> bool:
        return name in self._items

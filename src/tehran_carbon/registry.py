"""Name -> class/function registries, so a config string selects an implementation.

    SOURCES = Registry("data source")
    @SOURCES.register("traffic")
    def load_traffic(ctx, year): ...
    SOURCES.get(cfg["..."])
"""
from __future__ import annotations


class Registry:
    def __init__(self, kind: str):
        self.kind = kind
        self._items: dict = {}

    def register(self, name: str):
        def deco(obj):
            if name in self._items:
                raise KeyError(f"{self.kind} '{name}' is registered twice")
            self._items[name] = obj
            return obj
        return deco

    def get(self, name: str):
        if name not in self._items:
            raise KeyError(f"unknown {self.kind} '{name}'; available: {', '.join(sorted(self._items))}")
        return self._items[name]

    def names(self) -> list[str]:
        return sorted(self._items)

    def __contains__(self, name: str) -> bool:
        return name in self._items

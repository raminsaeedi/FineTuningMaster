"""A tiny registry for pluggable components.

Anything that has several interchangeable variants (methods, retrievers,
metrics, trainers) is registered here by a string key. New variants are added
by decorating a class with ``@REGISTRY.register("key")`` — no central dispatch
code needs to change.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterator


class Registry:
    """Maps string keys to classes (or factories), with helpful errors."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._items: Dict[str, type] = {}

    def register(self, key: str) -> Callable[[type], type]:
        """Decorator that records ``cls`` under ``key``."""

        def decorator(cls: type) -> type:
            if key in self._items:
                raise ValueError(f"'{key}' already registered in {self.name}")
            self._items[key] = cls
            return cls

        return decorator

    def get(self, key: str) -> type:
        if key not in self._items:
            raise KeyError(
                f"'{key}' not found in {self.name}. "
                f"Available: {sorted(self._items)}"
            )
        return self._items[key]

    def keys(self) -> list[str]:
        return sorted(self._items)

    def __contains__(self, key: str) -> bool:
        return key in self._items

    def __iter__(self) -> Iterator[str]:
        return iter(self._items)

    def __repr__(self) -> str:
        return f"Registry({self.name!r}, items={self.keys()})"


# Global registries used across the project.
METHODS = Registry("methods")
RETRIEVERS = Registry("retrievers")
METRICS = Registry("metrics")
TRAINERS = Registry("trainers")

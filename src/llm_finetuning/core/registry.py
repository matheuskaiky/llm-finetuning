"""Generic registry and factory.

Components register under a string key; configuration selects an implementation
by that key.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .interfaces import DatasetLoader, Evaluator, Metric, ModelProvider, Trainer


class Registry[T]:
    """Maps string keys to classes and builds instances from them."""

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._entries: dict[str, type[T]] = {}

    def register(self, key: str | None = None) -> Callable[[type[T]], type[T]]:
        """Decorator registering a class under ``key`` (defaults to its name)."""

        def decorator(cls: type[T]) -> type[T]:
            name = key or cls.__name__
            if name in self._entries:
                raise KeyError(
                    f"{self._kind!r} already has an entry named {name!r}"
                )
            self._entries[name] = cls
            return cls

        return decorator

    def get(self, key: str) -> type[T]:
        """Return the class registered under ``key`` or raise a helpful error."""
        try:
            return self._entries[key]
        except KeyError:
            raise KeyError(
                f"unknown {self._kind} {key!r}; available: {self.available()}"
            ) from None

    def build(self, key: str, **params: Any) -> T:
        """Instantiate the class registered under ``key`` with ``params``."""
        return self.get(key)(**params)

    def available(self) -> list[str]:
        """Sorted list of registered keys."""
        return sorted(self._entries)

    def __contains__(self, key: str) -> bool:
        return key in self._entries


def instantiate[T](registry: Registry[T], spec: Any) -> T:
    """Build a component from a :class:`ComponentSpec` (or plain dict).

    ``spec`` is expected to expose a ``type`` (the registry key) and ``params``
    (keyword arguments). A bare mapping without ``params`` is treated as the
    params themselves.
    """
    if hasattr(spec, "type"):
        key, params = spec.type, dict(getattr(spec, "params", {}) or {})
    else:
        spec = dict(spec)
        key = spec.pop("type")
        params = dict(spec.get("params", spec))
    return registry.build(key, **params)


# One registry per core abstraction.
MODEL_PROVIDERS: Registry[ModelProvider] = Registry("model_provider")
DATASET_LOADERS: Registry[DatasetLoader] = Registry("dataset_loader")
TRAINERS: Registry[Trainer] = Registry("trainer")
METRICS: Registry[Metric] = Registry("metric")
EVALUATORS: Registry[Evaluator] = Registry("evaluator")

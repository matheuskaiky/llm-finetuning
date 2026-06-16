"""Model providers. Importing this module registers the built-in providers."""

from __future__ import annotations

from .providers import CloudModelProvider, LocalModelProvider

__all__ = ["CloudModelProvider", "LocalModelProvider"]

"""Training strategies. Importing this module registers the built-in trainers."""

from __future__ import annotations

from .pretrain import ContinualPretrainTrainer

__all__ = ["ContinualPretrainTrainer"]

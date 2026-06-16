"""Training strategies. Importing this module registers the built-in trainers."""

from __future__ import annotations

from .distill import LogitKDTrainer
from .pretrain import ContinualPretrainTrainer
from .sft import SupervisedFineTuneTrainer

__all__ = ["ContinualPretrainTrainer", "SupervisedFineTuneTrainer", "LogitKDTrainer"]

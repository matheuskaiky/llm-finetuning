"""Core abstractions: interfaces, registry, configuration and reproducibility.

Import-time dependencies are limited to pydantic, pyyaml and numpy.
"""

from __future__ import annotations

from .config import (
    ComponentSpec,
    EvaluationConfig,
    ExperimentConfig,
    load_config,
)
from .interfaces import (
    DatasetLoader,
    Evaluator,
    Metric,
    ModelProvider,
    Trainer,
    TrainResult,
)
from .registry import (
    DATASET_LOADERS,
    EVALUATORS,
    METRICS,
    MODEL_PROVIDERS,
    TRAINERS,
    Registry,
    instantiate,
)
from .seed import set_global_seed

__all__ = [
    "ComponentSpec",
    "EvaluationConfig",
    "ExperimentConfig",
    "load_config",
    "DatasetLoader",
    "Evaluator",
    "Metric",
    "ModelProvider",
    "Trainer",
    "TrainResult",
    "DATASET_LOADERS",
    "EVALUATORS",
    "METRICS",
    "MODEL_PROVIDERS",
    "TRAINERS",
    "Registry",
    "instantiate",
    "set_global_seed",
]

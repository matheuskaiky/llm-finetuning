"""Contract tests: every registered component honors its core interface.

These tests are intentionally generic. They assert the shared contracts of the
core abstractions for whatever is currently registered, without pinning down
question-specific behavior, so new implementations are checked automatically as
they are added.
"""

from __future__ import annotations

import math

import pytest

from llm_finetuning.core.interfaces import (
    DatasetLoader,
    Evaluator,
    Metric,
    ModelProvider,
    Trainer,
)
from llm_finetuning.core.registry import (
    DATASET_LOADERS,
    EVALUATORS,
    METRICS,
    MODEL_PROVIDERS,
    TRAINERS,
)


@pytest.mark.parametrize("name", MODEL_PROVIDERS.available())
def test_model_provider_contract(name: str) -> None:
    cls = MODEL_PROVIDERS.get(name)
    assert issubclass(cls, ModelProvider)
    assert callable(getattr(cls, "load", None))


@pytest.mark.parametrize("name", DATASET_LOADERS.available())
def test_dataset_loader_contract(name: str) -> None:
    cls = DATASET_LOADERS.get(name)
    assert issubclass(cls, DatasetLoader)
    assert callable(getattr(cls, "load", None))


@pytest.mark.parametrize("name", TRAINERS.available())
def test_trainer_contract(name: str) -> None:
    cls = TRAINERS.get(name)
    assert issubclass(cls, Trainer)
    assert callable(getattr(cls, "train", None))


@pytest.mark.parametrize("name", EVALUATORS.available())
def test_evaluator_contract(name: str) -> None:
    cls = EVALUATORS.get(name)
    assert issubclass(cls, Evaluator)
    assert callable(getattr(cls, "evaluate", None))


@pytest.mark.parametrize("name", METRICS.available())
def test_metric_contract(name: str) -> None:
    cls = METRICS.get(name)
    assert issubclass(cls, Metric)
    metric = METRICS.build(name)
    assert metric.name
    # A reset metric with no data yields a defined (NaN) value, not a crash.
    metric.reset()
    assert math.isnan(metric.compute())


def test_registries_expose_sorted_names() -> None:
    for registry in (MODEL_PROVIDERS, DATASET_LOADERS, TRAINERS, METRICS, EVALUATORS):
        names = registry.available()
        assert names == sorted(names)

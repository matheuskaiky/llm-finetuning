"""Smoke tests: the package imports and registers its built-in components."""

from __future__ import annotations

import llm_finetuning
from llm_finetuning.core.registry import (
    DATASET_LOADERS,
    EVALUATORS,
    METRICS,
    MODEL_PROVIDERS,
    TRAINERS,
)


def test_version() -> None:
    assert isinstance(llm_finetuning.__version__, str)


def test_builtin_components_registered() -> None:
    assert {"perplexity", "cross_entropy", "token_accuracy"} <= set(METRICS.available())
    assert {"local", "cloud"} <= set(MODEL_PROVIDERS.available())
    assert {"pdf_to_text", "text_corpus"} <= set(DATASET_LOADERS.available())
    assert "language_model" in EVALUATORS
    assert "continual_pretrain" in TRAINERS

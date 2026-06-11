"""Typed configuration loaded and validated from YAML (``configs/*.yaml``)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class ComponentSpec(BaseModel):
    """Selects a registered component by ``type`` and configures it.

    Example (YAML)::

        model:
          type: local
          params:
            model_name: gpt2
    """

    type: str
    params: dict[str, Any] = Field(default_factory=dict)


class EvaluationConfig(BaseModel):
    """How to run an intrinsic language-model evaluation."""

    metrics: list[str] = Field(
        default_factory=lambda: ["perplexity", "cross_entropy", "token_accuracy"]
    )
    #: Path to a benchmark file (``.txt`` corpus or ``.jsonl`` with a ``text`` field).
    benchmark: str
    max_length: int = 1024
    #: Sliding-window stride for long documents (0 = no overlap).
    stride: int = 0
    output_path: str | None = None


class ExperimentConfig(BaseModel):
    """Top-level experiment description."""

    name: str = "experiment"
    seed: int = 42
    model: ComponentSpec | None = None
    dataset: ComponentSpec | None = None
    trainer: ComponentSpec | None = None
    evaluation: EvaluationConfig | None = None


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment config from a YAML file."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return ExperimentConfig.model_validate(raw or {})

"""Tests for YAML config loading and validation."""

from __future__ import annotations

from pathlib import Path

from llm_finetuning.core import load_config

_YAML = """
name: demo
seed: 7
model:
  type: local
  params:
    model_name: gpt2
evaluation:
  metrics: [perplexity]
  benchmark: benchmarks/sample_corpus.jsonl
  max_length: 256
"""


def test_load_config(tmp_path: Path) -> None:
    cfg_path = tmp_path / "exp.yaml"
    cfg_path.write_text(_YAML, encoding="utf-8")

    config = load_config(cfg_path)

    assert config.name == "demo"
    assert config.seed == 7
    assert config.model is not None
    assert config.model.type == "local"
    assert config.model.params["model_name"] == "gpt2"
    assert config.evaluation is not None
    assert config.evaluation.metrics == ["perplexity"]
    assert config.evaluation.max_length == 256


def test_defaults_for_empty_config(tmp_path: Path) -> None:
    cfg_path = tmp_path / "empty.yaml"
    cfg_path.write_text("", encoding="utf-8")

    config = load_config(cfg_path)

    assert config.seed == 42
    assert config.model is None

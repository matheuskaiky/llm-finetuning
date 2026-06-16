"""Validation tests for the versioned benchmark question sets.

Fast and dependency-free: they only parse the JSONL and check structure, so they
run without the ML stack.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
Q1_BENCHMARK = REPO_ROOT / "benchmarks" / "pre_treino" / "diarios_qa.jsonl"


def _load_jsonl(path: Path) -> list[dict]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def test_q1_benchmark_exists() -> None:
    assert Q1_BENCHMARK.is_file()


def test_q1_benchmark_has_at_least_25_questions() -> None:
    # Tarefa Q1 requires at least 25 questions with reference answers.
    assert len(_load_jsonl(Q1_BENCHMARK)) >= 25


def test_q1_benchmark_records_well_formed() -> None:
    for record in _load_jsonl(Q1_BENCHMARK):
        assert record.get("instruction", "").strip(), record
        assert record.get("output", "").strip(), record


def test_q1_benchmark_questions_unique() -> None:
    instructions = [r["instruction"] for r in _load_jsonl(Q1_BENCHMARK)]
    assert len(instructions) == len(set(instructions))

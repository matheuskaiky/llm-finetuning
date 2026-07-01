"""Tests for the per-item, multi-run results helpers (no ML stack needed)."""

from __future__ import annotations

import csv

from llm_finetuning.evaluation.results import (
    DEFAULT_EVAL_RUNS,
    run_seeds,
    summarize_runs,
    write_item_results,
)


def test_default_runs_is_five():
    assert DEFAULT_EVAL_RUNS == 5


def test_run_seeds_are_distinct_and_offset():
    assert run_seeds(42, 5) == [42, 43, 44, 45, 46]
    assert len(set(run_seeds(7))) == DEFAULT_EVAL_RUNS


def test_write_item_results_single_csv_all_runs(tmp_path):
    rows = [
        {"run": r, "id": i, "model": "m", "score": r + i}
        for r in (1, 2)
        for i in (1, 2)
    ]
    out = write_item_results(tmp_path / "res.csv", rows)
    got = list(csv.DictReader(out.open(encoding="utf-8")))
    assert len(got) == 4  # 2 runs x 2 items, all in one file
    assert {r["run"] for r in got} == {"1", "2"}
    assert got[0]["model"] == "m"


def test_write_item_results_field_union_order(tmp_path):
    rows = [{"id": 1, "a": 1}, {"id": 2, "b": 2}]
    out = write_item_results(tmp_path / "u.csv", rows)
    header = out.read_text(encoding="utf-8").splitlines()[0]
    assert header == "id,a,b"


def test_summarize_runs_mean_std_over_runs():
    rows = [
        {"run": 1, "model": "m", "score": 1.0},
        {"run": 2, "model": "m", "score": 3.0},
    ]
    summ = summarize_runs(rows, ["model"], ["score"])
    assert len(summ) == 1
    row = summ[0]
    assert row["model"] == "m"
    assert row["runs"] == 2
    assert row["score_mean"] == 2.0
    assert round(row["score_std"], 4) == round((2.0**0.5), 4)  # sample std of {1,3}


def test_summarize_runs_skips_non_numeric():
    rows = [
        {"run": 1, "model": "m", "score": "n/a"},
        {"run": 2, "model": "m", "score": 4.0},
    ]
    summ = summarize_runs(rows, ["model"], ["score"])
    assert summ[0]["score_mean"] == 4.0

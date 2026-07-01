"""Shared helpers for per-item, multi-run benchmark results.

Every benchmark evaluation repeats N times (default 5) and records one row per
``(run, item)`` into a single CSV, so all repetitions live in the same file and each
benchmark id keeps its own model result. A ``run`` column (1..N) distinguishes the
repetitions; aggregate mean/std across runs is computed by :func:`summarize_runs`.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

# Mandatory number of evaluation repetitions per benchmark.
DEFAULT_EVAL_RUNS = 5


def run_seeds(base_seed: int, runs: int = DEFAULT_EVAL_RUNS) -> list[int]:
    """Return one seed per run (``base_seed`` for run 1, then +1 each run)."""
    return [base_seed + i for i in range(runs)]


def _union_fields(rows: Sequence[dict[str, Any]]) -> list[str]:
    """Field names across ``rows`` in first-seen order."""
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    return fields


def write_item_results(
    path: str | Path,
    rows: Iterable[dict[str, Any]],
    fieldnames: Sequence[str] | None = None,
) -> Path:
    """Write per-item result rows (all runs) to a single CSV.

    ``rows`` should already carry a ``run`` and an ``id`` field. ``fieldnames`` sets
    the column order; when omitted it is the union of keys in first-seen order.
    """
    path = Path(path)
    rows = list(rows)
    fields = list(fieldnames) if fieldnames is not None else _union_fields(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _mean_std(values: Sequence[float]) -> tuple[float, float]:
    n = len(values)
    if n == 0:
        return (float("nan"), float("nan"))
    mean = sum(values) / n
    if n == 1:
        return (mean, 0.0)
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return (mean, math.sqrt(var))


def summarize_runs(
    rows: Sequence[dict[str, Any]],
    group_fields: Sequence[str],
    metric_fields: Sequence[str],
) -> list[dict[str, Any]]:
    """Aggregate per-item rows across runs into mean/std per group.

    Groups by ``group_fields`` (e.g. ``["model"]`` or ``["model", "type"]``),
    averaging each metric in ``metric_fields`` over all rows of the group (across the
    5 runs and all items). Non-numeric or missing metric cells are skipped.
    """
    buckets: dict[tuple, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    runs_seen: dict[tuple, set] = defaultdict(set)
    for row in rows:
        key = tuple(row.get(g) for g in group_fields)
        runs_seen[key].add(row.get("run"))
        for m in metric_fields:
            val = row.get(m)
            try:
                buckets[key][m].append(float(val))
            except (TypeError, ValueError):
                continue

    summary: list[dict[str, Any]] = []
    for key in sorted(buckets, key=lambda k: tuple(str(x) for x in k)):
        out: dict[str, Any] = dict(zip(group_fields, key, strict=False))
        out["runs"] = len(runs_seen[key])
        out["n"] = max(
            (len(buckets[key][m]) for m in metric_fields if buckets[key][m]),
            default=0,
        )
        for m in metric_fields:
            mean, std = _mean_std(buckets[key][m])
            out[f"{m}_mean"] = round(mean, 6)
            out[f"{m}_std"] = round(std, 6)
        summary.append(out)
    return summary

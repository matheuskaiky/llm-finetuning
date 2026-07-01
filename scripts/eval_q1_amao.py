#!/usr/bin/env python3
"""Q1 hand-made benchmark evaluation (P&R and Cloze) on base and instruct models.

Two shapes, auto-detected per item:
  - cloze (``context``/``target``): the ``____`` blank marks where the answer goes.
    Reports the teacher-forced perplexity of the target given the prefix and an
    exact-match of a greedy continuation. Perplexity is the fair metric for base
    models (no instruction following required); exact-match complements it.
  - Q&A (``instruction``/``output``): raw completion + exact-match + answer
    perplexity.

Runs the benchmark N times (default 5) and writes one row per (run, id, model) to a
single CSV, plus a mean/std summary. Matching is OCR-tolerant (accents stripped,
``O``/``l`` read as ``0``/``1`` inside numbers).

Usage:
    python scripts/eval_q1_amao.py \
        --benchmark benchmarks/pre_treino/diarios_cloze.jsonl \
        --models base=models/Qwen3-0.6B-Base instruct=models/Qwen3-0.6B \
        --out results/q1_amao_cloze.csv
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

from llm_finetuning.core import set_global_seed
from llm_finetuning.evaluation.evaluator import CLOZE_BLANK, load_benchmark_items
from llm_finetuning.evaluation.results import (
    DEFAULT_EVAL_RUNS,
    run_seeds,
    summarize_runs,
    write_item_results,
)
from llm_finetuning.rag.llm_client import LocalChatLLM


def _norm(s: str) -> str:
    """Lowercase, strip accents, keep alphanumerics only."""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]", "", s)


def _digits(s: str) -> str:
    """Digits only, reading OCR ``O``/``l`` as ``0``/``1``."""
    return re.sub(r"\D", "", s.lower().translate(str.maketrans("ol", "01")))


def _exact_match(generated: str, target: str) -> int:
    """1 if the target is recovered at the start of the generated continuation.

    Two OCR-tolerant checks: alphanumeric substring, and (for numeric targets) a
    digits-only match with ``O``/``l`` read as ``0``/``1``.
    """
    nt = _norm(target)
    if nt and nt in _norm(generated)[: len(nt) + 40]:
        return 1
    dt = _digits(target)
    if len(dt) >= 3 and dt in _digits(generated)[: len(dt) + 40]:
        return 1
    return 0


def _prefix_and_target(item: dict) -> tuple[str, str, str]:
    """Return (kind, prompt-prefix, target) for a cloze or Q&A item."""
    if "context" in item and "target" in item:
        ctx = item["context"]
        prefix = ctx.split(CLOZE_BLANK)[0].rstrip() if CLOZE_BLANK in ctx else ctx.rstrip()
        return "cloze", prefix, item["target"]
    prefix = f"{item.get('instruction', '')}\nResposta:"
    return "pr", prefix, item.get("output", "")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path,
                        default=Path("benchmarks/pre_treino/diarios_cloze.jsonl"))
    parser.add_argument("--models", nargs="+", required=True,
                        help="label=path entries for the models under test")
    parser.add_argument("--out", type=Path, default=Path("results/q1_amao_cloze.csv"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--limit", type=int, default=0, help="limit items (0=all)")
    parser.add_argument("--runs", type=int, default=DEFAULT_EVAL_RUNS,
                        help="mandatory number of evaluation repetitions (default 5)")
    parser.add_argument("--seed", type=int, default=42, help="base seed (run r uses seed+r-1)")
    args = parser.parse_args()

    items = load_benchmark_items(args.benchmark)
    if args.limit:
        items = items[: args.limit]
    print(f"items: {len(items)} | runs: {args.runs}", flush=True)

    specs = [m.split("=", 1) for m in args.models]
    seeds = run_seeds(args.seed, args.runs)

    per_item: list[dict] = []
    for label, path in specs:
        print(f"=== {label} ({path}) ===", flush=True)
        llm = LocalChatLLM(model_name=path, device=args.device, temperature=0.0,
                           max_new_tokens=args.max_new_tokens)
        for run, seed in enumerate(seeds, start=1):
            set_global_seed(seed)
            for idx, it in enumerate(items):
                kind, prefix, target = _prefix_and_target(it)
                gen = llm.complete(prefix, args.max_new_tokens)
                ppl = llm.response_perplexity(prefix, target)
                per_item.append({
                    "run": run, "id": it.get("id", idx + 1), "model": label, "kind": kind,
                    "tipo_documento": it.get("tipo_documento", ""),
                    "arquivo": it.get("arquivo_origem", it.get("arquivo", "")),
                    "target": target, "answer": gen,
                    "exact_match": _exact_match(gen, target),
                    "target_ppl": round(ppl, 4),
                })
        llm.unload()

    write_item_results(
        args.out, per_item,
        fieldnames=["run", "id", "model", "kind", "tipo_documento", "arquivo",
                    "target", "answer", "exact_match", "target_ppl"],
    )
    summary = summarize_runs(per_item, ["model"], ["exact_match", "target_ppl"])
    summary_path = args.out.with_name(args.out.stem + "_summary.csv")
    write_item_results(summary_path, summary)
    for row in summary:
        print(f"  {row['model']}: exact={row['exact_match_mean']:.3f}  "
              f"target_ppl={row['target_ppl_mean']:.3f}", flush=True)
    print(f"wrote {args.out} (per-item, {len(per_item)} linhas) e {summary_path}")


if __name__ == "__main__":
    main()

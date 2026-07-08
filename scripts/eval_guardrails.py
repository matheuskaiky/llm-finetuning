#!/usr/bin/env python3
"""Q6: measure the protection added by the guardrail layer on a 30-question set.

Compares WITH the GuardrailLayer vs WITHOUT (pass-through). Adversarial inputs
(jailbreak/unsafe) should be blocked; PII in outputs should be masked; benign inputs
should pass (false positive = a blocked benign). Pure layer evaluation (no LLM):
isolates the guardrail's protection.

Usage:
    python scripts/eval_guardrails.py
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from llm_finetuning.evaluation.results import (
    DEFAULT_EVAL_RUNS,
    summarize_runs,
    write_item_results,
)
from llm_finetuning.guardrails import GUARDRAILS, GuardrailLayer
from llm_finetuning.guardrails.pii import has_pii


def _layers() -> dict[str, GuardrailLayer]:
    return {
        "regex_only": GuardrailLayer([
            GUARDRAILS.build("jailbreak_block"),
            GUARDRAILS.build("unsafe_block"),
            GUARDRAILS.build("pii_mask"),
        ]),
        "regex_plus_semantic": GuardrailLayer([
            GUARDRAILS.build("jailbreak_block"),
            GUARDRAILS.build("unsafe_block"),
            GUARDRAILS.build("semantic_block"),
            GUARDRAILS.build("pii_mask"),
        ]),
    }


def _item_outcome(layer: GuardrailLayer, item: dict) -> tuple[int, int]:
    """Return ``(with_ok, without_ok)``: 1 when protection did the desired thing.

    Desired: adversarial (jailbreak/unsafe) blocked; pii_output masked; benign passed.
    "without" is the pass-through baseline (no layer).
    """
    t, text = item["type"], item["text"]
    if t in ("jailbreak", "unsafe"):  # want BLOCKED
        return (0 if layer.process_input(text).allowed else 1, 0)
    if t == "pii_output":  # want PII MASKED at output
        return (0 if has_pii(layer.process_output(text).text) else 1,
                0 if has_pii(text) else 1)
    if t == "benign":  # want PASS (not blocked)
        return (1 if layer.process_input(text).allowed else 0, 1)
    return (0, 0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path,
                        default=Path("benchmarks/guardrails/guardrails_30.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("results/q6_guardrails.csv"))
    parser.add_argument("--runs", type=int, default=DEFAULT_EVAL_RUNS,
                        help="mandatory number of evaluation repetitions (default 5)")
    args = parser.parse_args()

    items = [json.loads(ln) for ln in args.benchmark.read_text(encoding="utf-8").splitlines() if ln.strip()]
    layers = _layers()

    # One row per (run, item, layer): keeps the per-id outcome across all runs and layers.
    per_item: list[dict] = []
    for layer_name, layer in layers.items():
        for run in range(1, args.runs + 1):
            for idx, it in enumerate(items):
                with_ok, without_ok = _item_outcome(layer, it)
                per_item.append({
                    "layer": layer_name,
                    "run": run,
                    "id": it.get("id", idx + 1),
                    "type": it["type"],
                    "text": it["text"],
                    "with_ok": with_ok,
                    "without_ok": without_ok,
                })

    write_item_results(
        args.out, per_item,
        fieldnames=["layer", "run", "id", "type", "text", "with_ok", "without_ok"],
    )
    summary = summarize_runs(per_item, ["layer", "type"], ["with_ok", "without_ok"])
    summary_path = args.out.with_name(args.out.stem + "_summary.csv")
    write_item_results(summary_path, summary)

    # Overall console report (aggregated over the last run's counts, identical across
    # runs when the layer is deterministic).
    for layer_name in layers.keys():
        print(f"\n--- Layer: {layer_name} ---")
        stat: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "with": 0})
        for r in per_item:
            if r["run"] != 1 or r["layer"] != layer_name:
                continue
            s = stat[r["type"]]
            s["n"] += 1
            s["with"] += r["with_ok"]
        adv_n = sum(s["n"] for t, s in stat.items() if t in ("jailbreak", "unsafe"))
        adv_ok = sum(s["with"] for t, s in stat.items() if t in ("jailbreak", "unsafe"))
        pii = stat.get("pii_output", {"n": 0, "with": 0})
        benign = stat.get("benign", {"n": 0, "with": 0})
        print(f"runs: {args.runs} | itens: {len(items)}")
        print(f"adversariais bloqueados: {adv_ok}/{adv_n} "
              f"({adv_ok/max(adv_n,1)*100:.0f}%) vs 0/{adv_n} sem guardrails")
        print(f"PII mascarada: {pii['with']}/{pii['n']}")
        fp = benign["n"] - benign["with"]
        print(f"benignas bloqueadas (falsos positivos): {fp}/{benign['n']} "
              f"({fp/max(benign['n'],1)*100:.0f}%)")
    
    print(f"\nwrote {args.out} (per-item, {len(per_item)} linhas) e {summary_path}")


if __name__ == "__main__":
    main()

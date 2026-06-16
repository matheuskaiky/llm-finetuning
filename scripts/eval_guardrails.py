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
import csv
import json
from collections import defaultdict
from pathlib import Path

from llm_finetuning.guardrails import GUARDRAILS, GuardrailLayer
from llm_finetuning.guardrails.pii import has_pii


def _layer() -> GuardrailLayer:
    return GuardrailLayer([
        GUARDRAILS.build("jailbreak_block"),
        GUARDRAILS.build("unsafe_block"),
        GUARDRAILS.build("pii_mask"),
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path,
                        default=Path("benchmarks/guardrails/guardrails_30.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("results/q6_guardrails.csv"))
    args = parser.parse_args()

    items = [json.loads(l) for l in args.benchmark.read_text(encoding="utf-8").splitlines() if l.strip()]
    layer = _layer()

    # per (type, protection) -> handled count (blocked for adversarial, masked for pii,
    # passed for benign). "without" = pass-through (no layer).
    stat: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "with": 0, "without": 0})

    for it in items:
        t, text = it["type"], it["text"]
        s = stat[t]
        s["n"] += 1
        if t in ("jailbreak", "unsafe"):  # want BLOCKED
            s["with"] += 0 if layer.process_input(text).allowed else 1
            s["without"] += 0  # without guardrails nothing is blocked
        elif t == "pii_output":  # want PII MASKED at output
            out = layer.process_output(text)
            s["with"] += 0 if has_pii(out.text) else 1
            s["without"] += 0 if has_pii(text) else 1  # raw text still has PII
        elif t == "benign":  # want PASS (not blocked)
            s["with"] += 1 if layer.process_input(text).allowed else 0
            s["without"] += 1  # without guardrails everything passes

    rows = []
    for t, s in sorted(stat.items()):
        rows.append({"type": t, "n": s["n"],
                     "handled_with": s["with"], "handled_without": s["without"],
                     "rate_with": round(s["with"] / s["n"], 3),
                     "rate_without": round(s["without"] / s["n"], 3)})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    adv = [s for t, s in stat.items() if t in ("jailbreak", "unsafe")]
    n_adv = sum(s["n"] for s in adv)
    blocked = sum(s["with"] for s in adv)
    pii = stat.get("pii_output", {"n": 0, "with": 0})
    benign = stat.get("benign", {"n": 0, "with": 0})
    print(f"adversariais bloqueados (com guardrails): {blocked}/{n_adv} "
          f"({blocked/max(n_adv,1)*100:.0f}%) vs 0/{n_adv} sem")
    print(f"PII mascarada: {pii['with']}/{pii['n']}")
    fp = benign["n"] - benign["with"]
    print(f"benignas bloqueadas (falsos positivos): {fp}/{benign['n']} "
          f"({fp/max(benign['n'],1)*100:.0f}%)")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

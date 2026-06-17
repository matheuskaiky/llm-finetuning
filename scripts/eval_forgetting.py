#!/usr/bin/env python3
"""Q1 catastrophic-forgetting probe: out-of-domain perplexity before vs after.

Continued pre-training on the gazette corpus (Q1) lowers in-domain perplexity. This
script measures the cost: perplexity on an out-of-domain probe (docentesDC, academic
code/text, disjoint from the gazette domain) before (base) and after (Q1 checkpoint).
A rise after training is catastrophic forgetting. The in-domain held-out is evaluated
in the same pass to contrast the trade-off (in-domain should drop while OOD rises).

Usage:
    python scripts/eval_forgetting.py --probe-size 150 --out results/q1_forgetting.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import llm_finetuning.models  # noqa: F401  registers model providers
from llm_finetuning.core import instantiate, set_global_seed
from llm_finetuning.core.config import ComponentSpec
from llm_finetuning.core.registry import MODEL_PROVIDERS
from llm_finetuning.evaluation.evaluator import LanguageModelEvaluator

# (label, params, base path, Q1 checkpoint path). Base models and checkpoints are local.
LADDER = [
    ("Qwen3-0.6B", "0.6B", "models/Qwen3-0.6B-Base", "outputs/pretrain_qwen3_0p6b_diarios"),
    ("Qwen3-1.7B", "1.7B", "models/Qwen3-1.7B-Base", "outputs/pretrain_qwen3_1p7b_diarios"),
    ("gemma-3-1b", "1.0B", "models/gemma-3-1b-pt", "outputs/pretrain_gemma3_1b_diarios"),
]

OOD_SRC = Path("data/raw/docentesDC/docentesDC.jsonl")
INDOMAIN = Path("data/processed/diarios_heldout.jsonl")


def build_ood_probe(out_path: Path, n: int, seed: int) -> Path:
    """Sample n documents from the docentesDC jsonl into an OOD probe file."""
    lines = [l for l in OOD_SRC.read_text(encoding="utf-8").splitlines() if l.strip()]
    rng = random.Random(seed)
    sample = rng.sample(lines, min(n, len(lines)))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for line in sample:
            obj = json.loads(line)
            f.write(json.dumps({"text": obj["text"]}, ensure_ascii=False) + "\n")
    return out_path


def eval_ppl(model_path: str, benchmark: Path, max_length: int) -> dict[str, float]:
    """Load a model and return its intrinsic metrics over a benchmark file."""
    cfg = ComponentSpec(type="local", params={"model_name": model_path, "dtype": "bfloat16"})
    provider = instantiate(MODEL_PROVIDERS, cfg)
    bundle = provider.load()
    ev = LanguageModelEvaluator(max_length=max_length, stride=0)
    res = ev.evaluate(bundle, benchmark)
    del bundle
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return res


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--probe-size", type=int, default=150)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--probe-path", type=Path, default=Path("data/processed/ood_docentes_probe.jsonl"))
    p.add_argument("--out", type=Path, default=Path("results/q1_forgetting.csv"))
    args = p.parse_args()

    set_global_seed(args.seed)
    probe = build_ood_probe(args.probe_path, args.probe_size, args.seed)
    print(f"OOD probe: {probe} ({args.probe_size} docs from docentesDC)", flush=True)

    # OOD only here; the in-domain held-out antes/depois is already in runs.csv
    # (heldout_150docs) and is cited alongside in the docs to show the trade-off.
    eval_sets = [("ood_docentes", probe)]

    rows = []
    for label, params, base_path, ckpt_path in LADDER:
        for set_name, bench in eval_sets:
            before = eval_ppl(base_path, bench, args.max_length)
            after = eval_ppl(ckpt_path, bench, args.max_length)
            d_ppl = after["perplexity"] - before["perplexity"]
            rows.append({
                "model": label, "params": params, "eval_set": set_name,
                "ppl_antes": round(before["perplexity"], 3),
                "ppl_depois": round(after["perplexity"], 3),
                "delta_ppl": round(d_ppl, 3),
                "ce_antes": round(before["cross_entropy"], 3),
                "ce_depois": round(after["cross_entropy"], 3),
            })
            print(f"  {label} [{set_name}]: {before['perplexity']:.3f} -> "
                  f"{after['perplexity']:.3f} (delta {d_ppl:+.3f})", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

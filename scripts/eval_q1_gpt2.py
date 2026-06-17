#!/usr/bin/env python3
"""Q1 evaluation for the GPT-2 family (continued pre-training, full parameter).

Mirrors the treatment given to the other base models: intrinsic metrics
(perplexity, cross-entropy, token accuracy) for each GPT-2 size before (base
checkpoint from the provider) and after the gazette continued pre-training (Q1),
on three benchmarks: in-domain held-out gazette text, the conceptual Q&A set, and
the out-of-domain docentesDC probe (forgetting). GPT-2 uses an English byte-level
BPE tokenizer, so its absolute perplexity is not comparable across families; the
before/after delta within GPT-2 is the meaningful quantity.

Writes one row per (model, eval_set) with the model name kept alongside the
parameter count.
"""

from __future__ import annotations

import argparse
import csv
import gc
from pathlib import Path

import llm_finetuning.models  # noqa: F401  registers model providers
from llm_finetuning.core import instantiate, set_global_seed
from llm_finetuning.core.config import ComponentSpec
from llm_finetuning.core.registry import MODEL_PROVIDERS
from llm_finetuning.evaluation.evaluator import LanguageModelEvaluator

# (model name, params, base path, Q1 checkpoint path).
LADDER = [
    ("gpt2", "124M", "models/gpt2", "outputs/pretrain_gpt2_diarios"),
    ("gpt2-medium", "355M", "models/gpt2-medium", "outputs/pretrain_gpt2_medium_diarios"),
    ("gpt2-large", "774M", "models/gpt2-large", "outputs/pretrain_gpt2_large_diarios"),
]

EVAL_SETS = [
    ("heldout", Path("data/processed/diarios_heldout.jsonl")),
    ("qa", Path("benchmarks/pre_treino/diarios_qa.jsonl")),
    ("ood_docentes", Path("data/processed/ood_docentes_probe.jsonl")),
]


def eval_metrics(model_path: str, benchmark: Path, max_length: int) -> dict[str, float]:
    cfg = ComponentSpec(type="local", params={"model_name": model_path, "dtype": "bfloat16"})
    provider = instantiate(MODEL_PROVIDERS, cfg)
    bundle = provider.load()
    ev = LanguageModelEvaluator(max_length=max_length, stride=0)
    res = ev.evaluate(bundle, benchmark)
    del bundle, provider
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return res


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=Path("results/q1_gpt2.csv"))
    args = p.parse_args()

    set_global_seed(args.seed)
    rows = []
    for name, params, base_path, ckpt_path in LADDER:
        for set_name, bench in EVAL_SETS:
            if not bench.exists():
                print(f"  skip {name} [{set_name}]: missing {bench}", flush=True)
                continue
            before = eval_metrics(base_path, bench, args.max_length)
            after = eval_metrics(ckpt_path, bench, args.max_length)
            d_ppl = after["perplexity"] - before["perplexity"]
            rows.append({
                "model": name, "params": params, "eval_set": set_name,
                "ppl_antes": round(before["perplexity"], 3),
                "ppl_depois": round(after["perplexity"], 3),
                "delta_ppl": round(d_ppl, 3),
                "ce_antes": round(before["cross_entropy"], 3),
                "ce_depois": round(after["cross_entropy"], 3),
                "tokacc_antes": round(before["token_accuracy"], 3),
                "tokacc_depois": round(after["token_accuracy"], 3),
            })
            print(f"  {name} ({params}) [{set_name}]: {before['perplexity']:.3f} -> "
                  f"{after['perplexity']:.3f} (delta {d_ppl:+.3f})", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

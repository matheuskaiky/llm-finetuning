#!/usr/bin/env python3
"""Q1 held-out text perplexity on unseen diario documents, per epoch.

Computes the intrinsic perplexity of each held-out document (disjoint from training)
under every model (base / ep1..epN / instruct), truncated to the training block size.
Perplexity of fixed text is deterministic, so there is no 5-run repetition here; the
variance comes from the 2000-document distribution (bootstrap 95% CI over documents).

Writes one row per (model, document) to a single CSV plus a summary with corpus
perplexity = exp(sum_nll / sum_tokens) and a bootstrap CI.

Usage:
    python scripts/eval_q1_heldout.py \
        --heldout data/processed/diarios_heldout_full.jsonl \
        --models base=models/Qwen2.5-0.5B ep1=outputs/.../checkpoint-16027 \
                 instruct=models/Qwen2.5-0.5B-Instruct \
        --out results/q1_heldout_qwen2p5_0p5b.csv
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from llm_finetuning.evaluation.results import write_item_results
from llm_finetuning.rag.llm_client import LocalChatLLM


def _load(path: Path, limit: int) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    return rows[:limit] if limit else rows


def _bootstrap_ppl(per_doc: list[tuple[float, int]], n_boot: int, seed: int) -> tuple[float, float]:
    """95% CI for corpus perplexity by resampling documents with replacement."""
    rng = random.Random(seed)
    m = len(per_doc)
    if m == 0:
        return (float("nan"), float("nan"))
    ppls = []
    for _ in range(n_boot):
        s_nll = s_tok = 0.0
        for _ in range(m):
            nll, tok = per_doc[rng.randrange(m)]
            s_nll += nll
            s_tok += tok
        if s_tok:
            ppls.append(math.exp(s_nll / s_tok))
    ppls.sort()
    lo = ppls[int(0.025 * len(ppls))]
    hi = ppls[int(0.975 * len(ppls)) - 1]
    return (lo, hi)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heldout", type=Path,
                        default=Path("data/processed/diarios_heldout_full.jsonl"))
    parser.add_argument("--models", nargs="+", required=True,
                        help="label=path entries (base, ep1..epN, instruct)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=0, help="limit docs (0=all)")
    parser.add_argument("--out", type=Path, default=Path("results/q1_heldout.csv"))
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    docs = _load(args.heldout, args.limit)
    print(f"held-out docs: {len(docs)}", flush=True)
    specs = [m.split("=", 1) for m in args.models]

    per_item: list[dict] = []
    summary: list[dict] = []
    for label, path in specs:
        print(f"=== {label} ({path}) ===", flush=True)
        llm = LocalChatLLM(model_name=path, device=args.device, temperature=0.0)
        per_doc: list[tuple[float, int]] = []
        for d in docs:
            text = d.get("text", "")
            nll, ntok = llm.text_nll(text, args.max_length)
            if ntok == 0:
                continue
            per_doc.append((nll, ntok))
            per_item.append({
                "model": label, "id": d.get("id", ""), "n_tokens": ntok,
                "doc_ppl": round(math.exp(nll / ntok), 4),
            })
        llm.unload()
        s_nll = sum(n for n, _ in per_doc)
        s_tok = sum(t for _, t in per_doc)
        corpus_ppl = math.exp(s_nll / s_tok) if s_tok else float("nan")
        lo, hi = _bootstrap_ppl(per_doc, args.bootstrap, args.seed)
        summary.append({"model": label, "n_docs": len(per_doc), "n_tokens": s_tok,
                        "corpus_ppl": round(corpus_ppl, 4),
                        "ci95_low": round(lo, 4), "ci95_high": round(hi, 4)})
        print(f"  {label}: corpus_ppl={corpus_ppl:.3f}  CI95=[{lo:.2f}, {hi:.2f}]", flush=True)

    write_item_results(args.out, per_item,
                       fieldnames=["model", "id", "n_tokens", "doc_ppl"])
    summary_path = args.out.with_name(args.out.stem + "_summary.csv")
    write_item_results(summary_path, summary)
    print(f"wrote {args.out} (per-doc) e {summary_path}")


if __name__ == "__main__":
    main()

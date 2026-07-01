#!/usr/bin/env python3
"""Q5 retrieval metrics: answer hit-rate@k of the vector retriever.

The RAG benchmark has no gold document ids, so retrieval quality is measured by a
standard proxy: a question is a "hit" at k if the expected answer string appears in
any of the top-k retrieved chunks (answer recall@k / context hit-rate). Reports plain
similarity search and MMR side by side. This isolates the retriever from the generator:
it measures whether the evidence reaches the prompt at all.

Usage:
    python scripts/eval_retrieval.py --config configs/rag_diarios.yaml \
        --benchmark benchmarks/rag/diarios_rag_30.jsonl --out results/q5_retrieval.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from llm_finetuning.rag import load_rag_config
from llm_finetuning.rag.vector_store import Embedder, VectorStore

KS = [1, 3, 5, 10]


def normalize(s: str) -> str:
    """Lowercase and collapse whitespace for substring matching."""
    return re.sub(r"\s+", " ", s.lower()).strip()


def is_hit(answer: str, chunks: list[str]) -> bool:
    """True if the normalized expected answer is a substring of any chunk."""
    a = normalize(answer)
    if not a:
        return False
    blob = normalize(" ".join(chunks))
    return a in blob


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/rag_diarios.yaml"))
    p.add_argument("--benchmark", type=Path, default=Path("benchmarks/rag/diarios_rag_30.jsonl"))
    p.add_argument("--out", type=Path, default=Path("results/q5_retrieval.csv"))
    p.add_argument("--device", default="cpu", help="embedder device (cpu avoids GPU contention)")
    args = p.parse_args()

    cfg = load_rag_config(args.config)
    items = [json.loads(ln) for ln in args.benchmark.read_text(encoding="utf-8").splitlines() if ln.strip()]

    embedder = Embedder(cfg.embedder.model_name, args.device, cfg.embedder.batch_size)
    store = VectorStore.load(cfg.index.vector_dir, embedder)

    maxk = max(KS)
    methods = {"plain": lambda q: store.search(q, maxk),
               "mmr": lambda q: store.search_mmr(q, maxk, cfg.agent.mmr_fetch_k, cfg.agent.mmr_lambda)}

    rows = []
    for method, retrieve in methods.items():
        hits = {k: 0 for k in KS}
        per_type: dict[str, dict[str, int]] = {}
        for it in items:
            q, exp, qtype = it["question"], it["expected_answer"], it.get("type", "all")
            retrieved = retrieve(q)
            chunks = [r.text for r in retrieved]
            per_type.setdefault(qtype, {"n": 0, **{k: 0 for k in KS}})
            per_type[qtype]["n"] += 1
            for k in KS:
                if is_hit(exp, chunks[:k]):
                    hits[k] += 1
                    per_type[qtype][k] += 1
        n = len(items)
        row = {"method": method, "n": n}
        for k in KS:
            row[f"hit@{k}"] = round(hits[k] / n, 3)
        rows.append(row)
        print(f"[{method}] " + "  ".join(f"hit@{k}={hits[k]}/{n}" for k in KS), flush=True)
        for qtype, d in sorted(per_type.items()):
            tr = {"method": f"{method}:{qtype}", "n": d["n"]}
            for k in KS:
                tr[f"hit@{k}"] = round(d[k] / d["n"], 3) if d["n"] else 0.0
            rows.append(tr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["method", "n"] + [f"hit@{k}" for k in KS])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

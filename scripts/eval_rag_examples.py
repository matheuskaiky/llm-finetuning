#!/usr/bin/env python3
"""Q5 qualitative examples: a few questions answered without RAG vs with RAG.

Shows the contribution of retrieval side by side (no-RAG baseline vs standard RAG)
on a handful of benchmark questions. Writes a Markdown table for the report.

Usage:
    python scripts/eval_rag_examples.py --limit 5 --out results/q5_qualitativos.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm_finetuning.rag import load_rag_config
from llm_finetuning.rag.llm_client import LocalChatLLM
from llm_finetuning.rag.pipelines import build_runner
from llm_finetuning.rag.retrievers import VectorRetriever
from llm_finetuning.rag.vector_store import Embedder, VectorStore

SYS = "Responda de forma objetiva e em portugues."


def baseline_answer(llm: LocalChatLLM, q: str) -> str:
    return llm.chat([{"role": "system", "content": SYS}, {"role": "user", "content": q}])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/rag_diarios.yaml"))
    p.add_argument("--benchmark", type=Path, default=Path("benchmarks/rag/diarios_rag_30.jsonl"))
    p.add_argument("--out", type=Path, default=Path("results/q5_qualitativos.md"))
    p.add_argument("--limit", type=int, default=5)
    args = p.parse_args()

    cfg = load_rag_config(args.config)
    items = [json.loads(l) for l in args.benchmark.read_text(encoding="utf-8").splitlines() if l.strip()]
    items = items[: args.limit]

    llm = LocalChatLLM.from_config(cfg.llm)
    embedder = Embedder(cfg.embedder.model_name, "cpu", cfg.embedder.batch_size)
    store = VectorStore.load(cfg.index.vector_dir, embedder)
    vec = VectorRetriever(store, cfg.agent.top_k_vector, cfg.agent.use_mmr,
                          cfg.agent.mmr_fetch_k, cfg.agent.mmr_lambda)
    runner = build_runner("standard", llm, vec, None, cfg.agent.max_reflections)

    lines = ["# Q5 - exemplos qualitativos (sem RAG vs com RAG)", "",
             f"Motor: {cfg.llm.model_name}. Modo RAG: standard (recupera + gera).", ""]
    for i, it in enumerate(items, 1):
        q, exp = it["question"], it["expected_answer"]
        base = baseline_answer(llm, q)
        rag = runner.answer(q).answer
        lines += [f"## Exemplo {i}", f"**Pergunta:** {q}", "",
                  f"**Resposta de referencia:** {exp}", "",
                  f"**Sem RAG (so o modelo):** {base}", "",
                  f"**Com RAG:** {rag}", ""]
        print(f"[{i}/{len(items)}] done", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

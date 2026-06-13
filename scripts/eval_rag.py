#!/usr/bin/env python3
"""Q5 evaluation: compare a no-RAG baseline against one or more RAG modes.

RAG modes (``--modes``, comma-separated; default ``agentic_graph``):
  - standard:       vector retrieve once + generate once (no agent, no graph);
  - agentic_vector: self-reflective agent, vector retrieval only (no graph);
  - agentic_graph:  self-reflective agent with the knowledge graph (full GraphRAG).

The same LLM answers every scenario; an LLM-as-judge scores each answer (0-5)
against the reference. The ablation isolates the effect of retrieval (standard vs
baseline), of the agent loop (agentic_vector vs standard) and of the graph
(agentic_graph vs agentic_vector). Results and a short analysis go to ``--out``.

Usage:
    python scripts/eval_rag.py --config configs/rag_diarios.yaml \
        --modes standard,agentic_vector,agentic_graph \
        --out results/benchmark_rag_compare.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from llm_finetuning.rag import load_rag_config
from llm_finetuning.rag.graph_store import KnowledgeGraph
from llm_finetuning.rag.judge import llm_judge
from llm_finetuning.rag.llm_client import LocalChatLLM
from llm_finetuning.rag.pipelines import RUNNERS, build_runner
from llm_finetuning.rag.retrievers import GraphRetriever, VectorRetriever
from llm_finetuning.rag.vector_store import Embedder, VectorStore


def load_benchmark(path: Path) -> list[dict[str, str]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def baseline_answer(llm: LocalChatLLM, question: str) -> str:
    return llm.chat(
        [
            {"role": "system", "content": "Responda a pergunta de forma concisa e factual, em portugues."},
            {"role": "user", "content": question},
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, default=Path("benchmarks/rag/diarios_rag_30.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("results/benchmark_rag_analysis.csv"))
    parser.add_argument("--modes", default="agentic_graph", help=f"comma list of {sorted(RUNNERS)}")
    parser.add_argument("--limit", type=int, default=0, help="limit questions (0=all, for testing)")
    parser.add_argument(
        "--judge-model",
        default=None,
        help="separate model for LLM-as-judge (fixed across engines for a fair "
        "comparison). Default: judge with the engine itself.",
    )
    parser.add_argument("--judge-device", default="cuda:1", help="device for the judge model")
    parser.add_argument("--vector-dir", default=None, help="override index.vector_dir")
    parser.add_argument("--graph-path", default=None, help="override index.graph_path")
    args = parser.parse_args()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    unknown = [m for m in modes if m not in RUNNERS]
    if unknown:
        raise SystemExit(f"unknown modes {unknown}; valid: {sorted(RUNNERS)}")

    cfg = load_rag_config(args.config)
    if args.vector_dir:  # reuse an engine config with a different index
        cfg.index.vector_dir = args.vector_dir
    if args.graph_path:
        cfg.index.graph_path = args.graph_path
    questions = load_benchmark(args.benchmark)
    if args.limit:
        questions = questions[: args.limit]

    llm = LocalChatLLM.from_config(cfg.llm)
    # A fixed judge (e.g. Qwen3-8B on cuda:1) keeps scoring comparable across
    # engines; without it a weak engine would also be a weak/biased judge of itself.
    judge = (
        LocalChatLLM(model_name=args.judge_model, device=args.judge_device,
                     max_new_tokens=8, temperature=0.0)
        if args.judge_model
        else llm
    )
    # Query embedder on CPU; GPUs stay free for the LLM (avoids device_map offload).
    embedder = Embedder(cfg.embedder.model_name, "cpu", cfg.embedder.batch_size)
    store = VectorStore.load(cfg.index.vector_dir, embedder)
    graph = KnowledgeGraph.load(cfg.index.graph_path)
    vec = VectorRetriever(store, cfg.agent.top_k_vector)
    gra = GraphRetriever(graph, cfg.agent.max_graph_hops)

    # One runner per requested mode (Strategy). Adding a mode = a new runner class
    # registered in rag.pipelines; this loop does not change.
    runners = {m: build_runner(m, llm, vec, gra, cfg.agent.max_reflections) for m in modes}

    rows = []
    scores: dict[str, list[int]] = {"baseline": [], **{m: [] for m in modes}}
    corrected: dict[str, int] = {m: 0 for m in modes}
    for i, item in enumerate(questions):
        q, expected = item["question"], item["expected_answer"]
        row: dict[str, object] = {"idx": i, "type": item.get("type", ""), "question": q}
        try:
            base = baseline_answer(llm, q)
            s_base = llm_judge(judge, q, expected, base)
        except Exception as exc:  # robust: one bad question must not kill the run
            base, s_base = f"<error: {exc}>", 0
        scores["baseline"].append(s_base)
        row["answer_baseline"], row["score_baseline"] = base, s_base
        for mode, runner in runners.items():
            try:
                res = runner.answer(q)
                s = llm_judge(judge, q, expected, res.answer)
            except Exception as exc:
                res, s = type("R", (), {"answer": f"<error: {exc}>", "corrected": False})(), 0
            scores[mode].append(s)
            if res.corrected:
                corrected[mode] += 1
            row[f"answer_{mode}"], row[f"score_{mode}"], row[f"corrected_{mode}"] = (
                res.answer, s, res.corrected,
            )
        rows.append(row)
        print(f"[{i + 1}/{len(questions)}] " + " ".join(f"{k}={scores[k][-1]}" for k in scores), flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    n = len(questions)
    judge_name = args.judge_model or cfg.llm.model_name
    print(f"\n## Analise Q5 - motor {cfg.llm.model_name} | juiz {judge_name} ({n} perguntas)\n")
    mean_base = sum(scores["baseline"]) / n
    print(f"- baseline (sem RAG):     {mean_base:.2f} / 5")
    for mode in modes:
        mm = sum(scores[mode]) / n
        extra = f"; auto-correcao {corrected[mode]}/{n}" if mode.startswith("agentic") else ""
        print(f"- {mode:16s} {mm:.2f} / 5  (ganho {mm - mean_base:+.2f}){extra}")
    print(f"\nCSV: {args.out}")


if __name__ == "__main__":
    main()

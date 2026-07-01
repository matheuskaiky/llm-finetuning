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
import json
from pathlib import Path

from llm_finetuning.core import set_global_seed
from llm_finetuning.evaluation.results import (
    DEFAULT_EVAL_RUNS,
    run_seeds,
    summarize_runs,
    write_item_results,
)
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
    parser.add_argument("--llm-model", default=None,
                        help="override llm.model_name (reuse one config across engines)")
    parser.add_argument("--vector-dir", default=None, help="override index.vector_dir")
    parser.add_argument("--graph-path", default=None, help="override index.graph_path")
    parser.add_argument("--mmr", action="store_true", help="enable MMR reranking on the vector retriever")
    parser.add_argument("--mmr-lambda", type=float, default=None, help="MMR lambda (relevance vs diversity)")
    parser.add_argument("--runs", type=int, default=DEFAULT_EVAL_RUNS,
                        help="mandatory number of evaluation repetitions (default 5)")
    parser.add_argument("--seed", type=int, default=42, help="base seed (run r uses seed+r-1)")
    args = parser.parse_args()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    unknown = [m for m in modes if m not in RUNNERS]
    if unknown:
        raise SystemExit(f"unknown modes {unknown}; valid: {sorted(RUNNERS)}")

    cfg = load_rag_config(args.config)
    if args.llm_model:  # reuse one engine config across different generators
        cfg.llm.model_name = args.llm_model
    if args.vector_dir:  # reuse an engine config with a different index
        cfg.index.vector_dir = args.vector_dir
    if args.graph_path:
        cfg.index.graph_path = args.graph_path
    if args.mmr:
        cfg.agent.use_mmr = True
    if args.mmr_lambda is not None:
        cfg.agent.mmr_lambda = args.mmr_lambda
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
    vec = VectorRetriever(
        store, cfg.agent.top_k_vector, cfg.agent.use_mmr,
        cfg.agent.mmr_fetch_k, cfg.agent.mmr_lambda,
    )
    # The graph is only needed for the agentic_graph mode; load it lazily so
    # retrieval-only experiments (e.g. a vector-only deduped index) need no graph.
    need_graph = "agentic_graph" in modes
    graph = KnowledgeGraph.load(cfg.index.graph_path) if need_graph else None
    gra = GraphRetriever(graph, cfg.agent.max_graph_hops) if need_graph else None

    # One runner per requested mode (Strategy). Adding a mode = a new runner class
    # registered in rag.pipelines; this loop does not change.
    runners = {m: build_runner(m, llm, vec, gra, cfg.agent.max_reflections) for m in modes}

    # One row per (run, question, scenario), scenario in {baseline, <modes>}. Keeps
    # each id's per-scenario answer/score for all runs in a single CSV.
    per_item: list[dict] = []
    seeds = run_seeds(args.seed, args.runs)
    for run, seed in enumerate(seeds, start=1):
        set_global_seed(seed)
        for i, item in enumerate(questions):
            q, expected = item["question"], item["expected_answer"]
            qid = item.get("id", i + 1)
            qtype = item.get("type", "")
            try:
                base = baseline_answer(llm, q)
                s_base = llm_judge(judge, q, expected, base)
            except Exception as exc:  # robust: one bad question must not kill the run
                base, s_base = f"<error: {exc}>", 0
            per_item.append({"run": run, "id": qid, "type": qtype, "scenario": "baseline",
                             "question": q, "answer": base, "score": s_base, "corrected": 0})
            for mode, runner in runners.items():
                try:
                    res = runner.answer(q)
                    s = llm_judge(judge, q, expected, res.answer)
                    corr = int(res.corrected)
                except Exception as exc:
                    res_answer, s, corr = f"<error: {exc}>", 0, 0
                else:
                    res_answer = res.answer
                per_item.append({"run": run, "id": qid, "type": qtype, "scenario": mode,
                                 "question": q, "answer": res_answer, "score": s,
                                 "corrected": corr})
            done = " ".join(
                f"{r['scenario']}={r['score']}"
                for r in per_item[-(len(modes) + 1):]
            )
            print(f"[run {run}/{args.runs}][{i + 1}/{len(questions)}] {done}", flush=True)

    write_item_results(
        args.out, per_item,
        fieldnames=["run", "id", "type", "scenario", "question", "answer", "score",
                    "corrected"],
    )
    summary = summarize_runs(per_item, ["scenario"], ["score", "corrected"])
    summary_path = args.out.with_name(args.out.stem + "_summary.csv")
    write_item_results(summary_path, summary)

    judge_name = args.judge_model or cfg.llm.model_name
    by = {row["scenario"]: row for row in summary}
    mean_base = by.get("baseline", {}).get("score_mean", 0.0)
    print(f"\n## Analise Q5 - motor {cfg.llm.model_name} | juiz {judge_name} "
          f"({len(questions)} perguntas x {args.runs} runs)\n")
    print(f"- baseline (sem RAG):     {mean_base:.2f} / 5")
    for mode in modes:
        row = by.get(mode, {})
        mm = row.get("score_mean", 0.0)
        extra = (f"; auto-correcao media {row.get('corrected_mean', 0.0):.2f}"
                 if mode.startswith("agentic") else "")
        print(f"- {mode:16s} {mm:.2f}+-{row.get('score_std', 0.0):.2f} / 5  "
              f"(ganho {mm - mean_base:+.2f}){extra}")
    print(f"\nCSV per-item: {args.out} ({len(per_item)} linhas) | resumo: {summary_path}")


if __name__ == "__main__":
    main()

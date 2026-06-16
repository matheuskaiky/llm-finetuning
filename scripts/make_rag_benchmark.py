#!/usr/bin/env python3
"""Generate a grounded 30-question RAG benchmark from the built index.

Two kinds of question, both with answers grounded in the indexed gazette content
(so the no-RAG baseline cannot know them, isolating the RAG contribution):
  - factual: answerable from a single chunk;
  - multihop: needs combining two graph relations (tests the knowledge graph).

Output JSONL lines: ``{"question", "expected_answer", "type"}``.

Usage:
    python scripts/make_rag_benchmark.py --config configs/rag_diarios.yaml \
        --out benchmarks/rag/diarios_rag_30.jsonl --n 30
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from llm_finetuning.rag import load_rag_config
from llm_finetuning.rag.graph_store import KnowledgeGraph
from llm_finetuning.rag.llm_client import LocalChatLLM

FACTUAL_SYS = (
    "Dado um trecho de diario oficial de municipio, escreva UMA pergunta factual "
    "especifica cuja resposta esteja EXPLICITA no trecho (nomes, valores, datas, "
    "cargos, numeros de ato). Responda APENAS com JSON: "
    '{"question": "...", "expected_answer": "..."} (resposta curta).'
)
MULTIHOP_SYS = (
    "Voce recebe uma cadeia de dois fatos: A -[r1]-> B -[r2]-> C. Escreva UMA "
    "pergunta cuja resposta seja C, formulada a partir de A, SEM mencionar o elo "
    "intermediario B (quem responde precisa descobrir B para chegar em C). Isso "
    "forca uma busca multi-hop de verdade. A resposta (expected_answer) deve ser C. "
    'Responda APENAS com JSON: {"question": "...", "expected_answer": "..."}.'
)


def _parse_qa(raw: str) -> dict[str, str] | None:
    s, e = raw.find("{"), raw.rfind("}")
    if s == -1 or e <= s:
        return None
    try:
        d = json.loads(raw[s : e + 1])
    except (json.JSONDecodeError, ValueError):
        return None
    q, a = str(d.get("question", "")).strip(), str(d.get("expected_answer", "")).strip()
    return {"question": q, "expected_answer": a} if q and a else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("benchmarks/rag/diarios_rag_30.jsonl"))
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--exclude-licitacao",
        action="store_true",
        help="generate only NON-licitacao factual questions (targeted pollution test)",
    )
    args = parser.parse_args()

    cfg = load_rag_config(args.config)
    random.seed(args.seed)
    llm = LocalChatLLM.from_config(cfg.llm)

    # Chunk texts come from the saved vector-store metadata (no embedder needed).
    meta = json.loads((Path(cfg.index.vector_dir) / "meta.json").read_text(encoding="utf-8"))
    texts = meta["texts"]

    if args.exclude_licitacao:
        from llm_finetuning.rag.doc_select import is_licitacao

        texts = [t for t in texts if not is_licitacao(t)]
        n_fact, n_multi = args.n, 0  # factual-only from non-licitacao chunks
        graph = None
    else:
        graph = KnowledgeGraph.load(cfg.index.graph_path)
        n_multi = args.n * 2 // 5  # ~40% multi-hop
        n_fact = args.n - n_multi

    items: list[dict[str, str]] = []
    seen: set[str] = set()

    # Factual from chunks (oversample, then dedupe).
    fact_chunks = random.sample(texts, min(len(texts), n_fact * 2))
    for chunk in fact_chunks:
        if sum(1 for it in items if it["type"] == "factual") >= n_fact:
            break
        raw = llm.chat(
            [{"role": "system", "content": FACTUAL_SYS},
             {"role": "user", "content": f"Trecho:\n{chunk[:1500]}\n\nJSON:"}],
            max_new_tokens=200,
        )
        qa = _parse_qa(raw)
        if qa and qa["question"].casefold() not in seen:
            seen.add(qa["question"].casefold())
            items.append({**qa, "type": "factual"})

    # Multi-hop from two-hop graph paths (skipped when graph is None / n_multi=0).
    paths = graph.two_hop_paths(limit=200) if (graph is not None and n_multi > 0) else []
    random.shuffle(paths)
    for a, r1, b, r2, c in paths:
        if sum(1 for it in items if it["type"] == "multihop") >= n_multi:
            break
        chain = (
            f"A = {a}\nr1 = {r1}\nB (NAO mencione na pergunta) = {b}\n"
            f"r2 = {r2}\nC (esta e a resposta) = {c}"
        )
        raw = llm.chat(
            [{"role": "system", "content": MULTIHOP_SYS},
             {"role": "user", "content": f"{chain}\n\nJSON:"}],
            max_new_tokens=200,
        )
        qa = _parse_qa(raw)
        if qa and qa["question"].casefold() not in seen:
            seen.add(qa["question"].casefold())
            items.append({**qa, "type": "multihop"})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for it in items[: args.n]:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")
    nf = sum(1 for it in items[: args.n] if it["type"] == "factual")
    nm = sum(1 for it in items[: args.n] if it["type"] == "multihop")
    print(f"wrote {min(len(items), args.n)} questions to {args.out} ({nf} factual, {nm} multihop)")


if __name__ == "__main__":
    main()

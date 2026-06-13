#!/usr/bin/env python3
"""Q5 Phase 1: build the GraphRAG index (vector store + knowledge graph).

Reads a bounded subset of the dom-pi gazette corpus, chunks it, embeds all chunks
into a FAISS store, and extracts an entity/relation graph from the first
``graph_max_chunks`` chunks with the instruct LLM. Both artifacts are saved under
``index.vector_dir`` / ``index.graph_path``.

Usage:
    python scripts/build_rag_index.py --config configs/rag_diarios.yaml
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

from llm_finetuning.rag import load_rag_config
from llm_finetuning.rag.chunking import chunk_document
from llm_finetuning.rag.extraction import (
    build_extraction_messages,
    ingest_into_graph,
    parse_extraction,
)
from llm_finetuning.rag.graph_store import KnowledgeGraph


def iter_docs(src_dir: str, config: str, limit: int, skip: int, min_tokens: int):
    """Yield ``(doc_id, text)`` for the selected gazette documents."""
    import pyarrow.parquet as pq

    files = sorted(glob.glob(f"{src_dir}/{config}/{config}-*.parquet"))
    if not files:
        raise SystemExit(f"no parquet found for config {config!r} under {src_dir}")
    seen = 0
    yielded = 0
    for path in files:
        table = pq.read_table(path, columns=["id", "n_tokens", "texto"])
        ids = table.column("id").to_pylist()
        ntok = table.column("n_tokens").to_pylist()
        texts = table.column("texto").to_pylist()
        for doc_id, n, text in zip(ids, ntok, texts, strict=False):
            if not text or (n is not None and n < min_tokens):
                continue
            if seen < skip:
                seen += 1
                continue
            if yielded >= limit:
                return
            yielded += 1
            yield str(doc_id), str(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--vectors-only",
        action="store_true",
        help="build only the FAISS vector store, skip the (slow) LLM graph extraction",
    )
    args = parser.parse_args()
    cfg = load_rag_config(args.config)

    # 1. Read + chunk the subset.
    if cfg.source.balance_licitacoes:
        # Read a larger pool, then downsample licitacao docs to pair with the rest.
        from llm_finetuning.rag.doc_select import balance_by_licitacao, is_licitacao

        pool = list(
            iter_docs(cfg.source.src_dir, cfg.source.config, cfg.source.pool_size,
                      cfg.source.skip, cfg.source.min_tokens)
        )
        docs = balance_by_licitacao(pool, seed=cfg.seed, max_total=cfg.source.limit)
        n_lic = sum(1 for _, t in docs if is_licitacao(t))
        print(f"balanced pool {len(pool)} -> {len(docs)} docs ({n_lic} licitacao, "
              f"{len(docs) - n_lic} outros)")
    else:
        docs = list(
            iter_docs(cfg.source.src_dir, cfg.source.config, cfg.source.limit,
                      cfg.source.skip, cfg.source.min_tokens)
        )

    chunk_texts: list[str] = []
    chunk_doc_ids: list[str] = []
    for doc_id, text in docs:
        for ch in chunk_document(doc_id, text, cfg.chunking.chunk_size, cfg.chunking.overlap):
            chunk_texts.append(ch.text)
            chunk_doc_ids.append(ch.doc_id)
    print(f"docs read; {len(chunk_texts)} chunks total")

    if cfg.chunking.dedup_near:
        from llm_finetuning.rag.doc_select import near_dup_keep_mask

        mask = near_dup_keep_mask(chunk_texts, cfg.chunking.dedup_threshold)
        chunk_texts = [t for t, k in zip(chunk_texts, mask, strict=False) if k]
        chunk_doc_ids = [d for d, k in zip(chunk_doc_ids, mask, strict=False) if k]
        print(f"near-dup dedup: kept {len(chunk_texts)} chunks ({sum(mask)} of {len(mask)})")

    # 2. Vector store over all chunks.
    from llm_finetuning.rag.vector_store import Embedder, VectorStore

    embedder = Embedder(cfg.embedder.model_name, cfg.embedder.device, cfg.embedder.batch_size)
    store = VectorStore(embedder)
    store.build(chunk_texts, chunk_doc_ids)
    store.save(cfg.index.vector_dir)
    print(f"vector store saved to {cfg.index.vector_dir}")
    # Free the embedder's GPU memory so the LLM's device_map="auto" keeps all
    # layers on the GPUs (a resident embedder triggers slow CPU offload).
    embedder.unload()

    if args.vectors_only:
        print("vectors-only: skipping graph extraction")
        return

    # 3. Knowledge graph from the first graph_max_chunks chunks (the costly step).
    from llm_finetuning.rag.llm_client import LocalChatLLM

    llm = LocalChatLLM.from_config(cfg.llm)
    graph = KnowledgeGraph()
    n_graph = min(cfg.graph_max_chunks, len(chunk_texts))
    for i in range(n_graph):
        raw = llm.chat(build_extraction_messages(chunk_texts[i]))
        ingest_into_graph(parse_extraction(raw), graph, chunk_doc_ids[i])
        if (i + 1) % 10 == 0:
            print(f"  extracted {i + 1}/{n_graph} chunks; "
                  f"{graph.num_entities()} entities, {graph.num_relations()} relations",
                  flush=True)
    graph.save(cfg.index.graph_path)
    print(f"graph saved to {cfg.index.graph_path}: "
          f"{graph.num_entities()} entities, {graph.num_relations()} relations")


if __name__ == "__main__":
    main()

"""Typed configuration for the RAG pipeline (``configs/rag_*.yaml``).

Defined in the rag package (not in core) so the RAG layer is added by extension
without changing the core experiment config. Validates with pydantic, same style
as ``core.config``.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class SourceConfig(BaseModel):
    """Which documents feed the index (a bounded subset of the corpus)."""

    src_dir: str = "data/raw/dom-pi-corpus-2025"
    config: str = "curated"
    limit: int = 150
    skip: int = 0
    min_tokens: int = 64
    #: When true, downsample licitacao-heavy documents so they are paired in count
    #: with the other document types (reads ``pool_size`` docs, then balances).
    balance_licitacoes: bool = False
    pool_size: int = 1200


class ChunkingConfig(BaseModel):
    """Character-based chunking of each document."""

    chunk_size: int = 1200
    overlap: int = 200
    #: Drop near-duplicate chunks (MinHash/LSH) before indexing, collapsing the
    #: repetitive licitacao chunks while keeping the unique ones.
    dedup_near: bool = False
    dedup_threshold: float = 0.85


class EmbedderConfig(BaseModel):
    model_name: str = "BAAI/bge-m3"
    device: str = "cuda"
    batch_size: int = 32


class LlmConfig(BaseModel):
    """Local instruct LLM used for extraction, generation and judging.

    ``device_map`` (e.g. "auto") splits a large model across both GPUs for
    inference (no NCCL needed); leave null to pin to a single ``device``.
    ``load_in_8bit``/``load_in_4bit`` quantize at load time via bitsandbytes; a
    checkpoint that is already FP8 needs neither (just ``device_map: auto``).
    """

    model_name: str = "models/Qwen3-8B"
    device: str = "cuda"
    device_map: str | None = None
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    max_new_tokens: int = 512
    temperature: float = 0.0


class IndexConfig(BaseModel):
    vector_dir: str = "data/processed/rag/diarios_faiss"
    graph_path: str = "data/processed/rag/diarios_graph.json"


class AgentConfig(BaseModel):
    top_k_vector: int = 5
    max_graph_hops: int = 2
    max_reflections: int = 2
    #: MMR reranking on the vector retriever (diversity, anti near-duplicate).
    use_mmr: bool = False
    mmr_fetch_k: int = 20
    mmr_lambda: float = 0.5


class RagConfig(BaseModel):
    """Top-level RAG configuration."""

    name: str = "rag_diarios"
    seed: int = 42
    #: Cap on chunks sent to the LLM for graph extraction (the costly step). The
    #: vector store still indexes all chunks; only the KG build is bounded.
    graph_max_chunks: int = 200
    source: SourceConfig = Field(default_factory=SourceConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    embedder: EmbedderConfig = Field(default_factory=EmbedderConfig)
    llm: LlmConfig = Field(default_factory=LlmConfig)
    index: IndexConfig = Field(default_factory=IndexConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)


def load_rag_config(path: str | Path) -> RagConfig:
    """Load and validate a RAG config from a YAML file."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return RagConfig.model_validate(raw or {})

"""Unit tests for the pure RAG logic (no ML stack needed).

Covers chunking, the knowledge graph traversal, and the tolerant extraction
parser. networkx is a light dependency; faiss/transformers/langgraph are not
imported here.
"""

from __future__ import annotations

import pytest

from llm_finetuning.rag.chunking import chunk_document, chunk_text
from llm_finetuning.rag.extraction import parse_extraction
from llm_finetuning.rag.graph_store import KnowledgeGraph, normalize_name


def test_chunk_text_overlap_and_coverage():
    text = "palavra " * 500  # ~4000 chars
    chunks = chunk_text(text, chunk_size=1000, overlap=200)
    assert len(chunks) > 1
    assert all(len(c) <= 1100 for c in chunks)  # +slack for word extension
    assert all(c.strip() for c in chunks)


def test_chunk_text_short_and_empty():
    assert chunk_text("   ") == []
    assert chunk_text("curto") == ["curto"]


def test_chunk_text_rejects_bad_overlap():
    with pytest.raises(ValueError):
        chunk_text("abc", chunk_size=10, overlap=10)


def test_chunk_document_records_provenance():
    chunks = chunk_document("doc42", "a " * 2000, chunk_size=1000, overlap=100)
    assert chunks[0].doc_id == "doc42"
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_normalize_name():
    assert normalize_name("  Prefeitura  DE   Teresina ") == "prefeitura de teresina"
    assert normalize_name("EMPRESA X") == normalize_name("empresa x")


def test_graph_relations_and_multihop():
    kg = KnowledgeGraph()
    kg.add_relation("Empresa X", "venceu", "Licitacao 01", "d1")
    kg.add_relation("Licitacao 01", "da", "Prefeitura de Teresina", "d1")
    kg.add_relation("Empresa Y", "venceu", "Licitacao 02", "d2")
    assert kg.num_entities() == 5
    assert kg.num_relations() == 3
    # Two hops from Empresa X reaches the city through the bid.
    key = normalize_name("Empresa X")
    reachable = kg.neighbors(key, hops=2)
    assert normalize_name("Prefeitura de Teresina") in reachable
    # One hop does not.
    assert normalize_name("Prefeitura de Teresina") not in kg.neighbors(key, hops=1)


def test_graph_context_text():
    kg = KnowledgeGraph()
    kg.add_relation("Empresa X", "venceu", "Licitacao 01", "d1")
    ctx = kg.context_for_entities(["Empresa X"], hops=1)
    assert "Empresa X" in ctx and "venceu" in ctx and "Licitacao 01" in ctx


def test_graph_save_load_roundtrip(tmp_path):
    kg = KnowledgeGraph()
    kg.add_relation("A", "rel", "B", "d1")
    p = kg.save(tmp_path / "g.json")
    kg2 = KnowledgeGraph.load(p)
    assert kg2.num_entities() == 2
    assert kg2.num_relations() == 1


def test_parse_extraction_plain_json():
    raw = '{"entities":[{"name":"Empresa X","type":"EMPRESA"}],"relations":[{"source":"Empresa X","relation":"venceu","target":"Licitacao 01"}]}'
    out = parse_extraction(raw)
    assert out["entities"][0]["name"] == "Empresa X"
    assert out["relations"][0]["target"] == "Licitacao 01"


def test_parse_extraction_with_fences_and_prose():
    raw = 'Claro! Aqui esta:\n```json\n{"entities": [{"name": "Prefeitura", "type": "PREFEITURA"}], "relations": []}\n```'
    out = parse_extraction(raw)
    assert out["entities"] == [{"name": "Prefeitura", "type": "PREFEITURA"}]
    assert out["relations"] == []


def test_parse_extraction_malformed_returns_empty():
    assert parse_extraction("desculpe, nao consegui") == {"entities": [], "relations": []}
    assert parse_extraction("") == {"entities": [], "relations": []}


def test_parse_extraction_drops_incomplete_relations():
    raw = '{"relations":[{"source":"A","target":""},{"source":"A","relation":"r","target":"B"}]}'
    out = parse_extraction(raw)
    assert out["relations"] == [{"source": "A", "relation": "r", "target": "B"}]


def test_graph_all_relations_and_two_hop_paths():
    kg = KnowledgeGraph()
    kg.add_relation("Empresa X", "venceu", "Licitacao 01", "d1")
    kg.add_relation("Licitacao 01", "da", "Prefeitura de Teresina", "d1")
    rels = kg.all_relations()
    assert ("Empresa X", "venceu", "Licitacao 01", "d1") in rels
    paths = kg.two_hop_paths(limit=10)
    # A -> B -> C with A != C: Empresa X -> Licitacao 01 -> Prefeitura.
    assert any(
        p[0] == "Empresa X" and p[2] == "Licitacao 01" and p[4] == "Prefeitura de Teresina"
        for p in paths
    )


def test_two_hop_paths_excludes_trivial_cycles():
    kg = KnowledgeGraph()
    kg.add_relation("A", "r", "B", "d1")
    kg.add_relation("B", "r2", "A", "d1")  # back to A; should be excluded
    paths = kg.two_hop_paths(limit=10)
    assert all(p[0] != p[4] for p in paths)

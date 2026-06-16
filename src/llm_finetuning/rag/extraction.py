"""Entity/relation extraction from text chunks via an instruct LLM.

The prompt asks for a strict JSON object; :func:`parse_extraction` is tolerant of
the usual instruct-model noise (code fences, leading prose) so a single malformed
chunk does not abort indexing. Prompt building and parsing are pure and tested
without the LLM.
"""

from __future__ import annotations

import json
from typing import Any

ENTITY_TYPES = (
    "PESSOA",
    "PREFEITURA",
    "EMPRESA",
    "ORGAO",
    "CARGO",
    "LICITACAO",
    "VALOR",
    "OUTRO",
)

EXTRACTION_SYSTEM = (
    "Voce extrai um grafo de conhecimento de trechos de diarios oficiais de "
    "municipios. Responda APENAS com um objeto JSON valido, sem texto fora dele, "
    "sem comentarios e sem cercas de codigo. Esquema:\n"
    '{"entities": [{"name": "...", "type": "<TIPO>"}], '
    '"relations": [{"source": "...", "relation": "...", "target": "..."}]}\n'
    f"Tipos permitidos: {', '.join(ENTITY_TYPES)}. Use nomes completos e "
    "canonicos. Extraia relacoes factuais explicitas (ex.: EMPRESA venceu "
    "LICITACAO; PREFEITURA nomeou PESSOA; CONTRATO no VALOR). Se nao houver nada "
    "relevante, devolva listas vazias."
)


def build_extraction_messages(chunk_text: str) -> list[dict[str, str]]:
    """Chat messages for extracting a knowledge graph from one chunk."""
    return [
        {"role": "system", "content": EXTRACTION_SYSTEM},
        {"role": "user", "content": f"Trecho:\n{chunk_text}\n\nJSON:"},
    ]


def _extract_json_object(raw: str) -> str | None:
    """Return the outermost ``{...}`` substring, tolerating fences/prose."""
    if not raw:
        return None
    text = raw.strip()
    if "```" in text:
        # Drop code fences, keeping inner content.
        parts = text.split("```")
        # Pick the part that looks most like JSON.
        candidates = [p[4:] if p.lstrip().startswith("json") else p for p in parts]
        text = max(candidates, key=lambda p: p.count("{") + p.count("}"))
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


def parse_extraction(raw: str) -> dict[str, list[dict[str, str]]]:
    """Parse an extraction response into ``{"entities": [...], "relations": [...]}``.

    Returns empty lists on any malformed or missing content (never raises).
    """
    empty: dict[str, list[dict[str, str]]] = {"entities": [], "relations": []}
    blob = _extract_json_object(raw)
    if blob is None:
        return empty
    try:
        data: Any = json.loads(blob)
    except (json.JSONDecodeError, ValueError):
        return empty
    if not isinstance(data, dict):
        return empty

    entities: list[dict[str, str]] = []
    for e in data.get("entities", []) or []:
        if isinstance(e, dict) and str(e.get("name", "")).strip():
            entities.append(
                {"name": str(e["name"]).strip(), "type": str(e.get("type", "")).strip()}
            )
    relations: list[dict[str, str]] = []
    for r in data.get("relations", []) or []:
        if not isinstance(r, dict):
            continue
        s = str(r.get("source", "")).strip()
        t = str(r.get("target", "")).strip()
        rel = str(r.get("relation", "")).strip()
        if s and t:
            relations.append({"source": s, "relation": rel, "target": t})
    return {"entities": entities, "relations": relations}


def ingest_into_graph(parsed: dict[str, list[dict[str, str]]], graph: Any, doc_id: str) -> None:
    """Add parsed entities and relations to a :class:`KnowledgeGraph`."""
    for e in parsed["entities"]:
        graph.add_entity(e["name"], e.get("type", ""))
    for r in parsed["relations"]:
        graph.add_relation(r["source"], r["relation"], r["target"], doc_id)

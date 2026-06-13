"""Knowledge graph over extracted entities and relations (NetworkX-backed).

Entities are nodes (keyed by a normalized name, carrying a display name and type);
relations are directed edges carrying a label and the source document id. Used for
multi-hop relational retrieval (e.g. "which companies won bids from city X").
"""

from __future__ import annotations

import json
from pathlib import Path


def normalize_name(name: str) -> str:
    """Canonical key for an entity: trimmed, collapsed spaces, casefolded."""
    return " ".join(name.split()).casefold()


class KnowledgeGraph:
    """Directed multigraph of entities and relations."""

    def __init__(self) -> None:
        import networkx as nx

        self._g = nx.MultiDiGraph()

    def add_entity(self, name: str, entity_type: str = "") -> str:
        """Add or update an entity; returns its normalized key."""
        key = normalize_name(name)
        if not key:
            return key
        if self._g.has_node(key):
            if entity_type and not self._g.nodes[key].get("type"):
                self._g.nodes[key]["type"] = entity_type
        else:
            self._g.add_node(key, display=name.strip(), type=entity_type)
        return key

    def add_relation(
        self, source: str, relation: str, target: str, doc_id: str = ""
    ) -> bool:
        """Add a directed ``source -[relation]-> target`` edge. Returns False if
        either endpoint is empty."""
        s = self.add_entity(source)
        t = self.add_entity(target)
        if not s or not t:
            return False
        self._g.add_edge(s, t, label=relation.strip(), doc_id=doc_id)
        return True

    def num_entities(self) -> int:
        return self._g.number_of_nodes()

    def num_relations(self) -> int:
        return self._g.number_of_edges()

    def find_entities(self, query: str) -> list[str]:
        """Normalized keys of entities whose display name contains ``query``."""
        q = query.casefold().strip()
        if not q:
            return []
        return [
            key
            for key, data in self._g.nodes(data=True)
            if q in data.get("display", key).casefold()
        ]

    def neighbors(self, key: str, hops: int = 1) -> set[str]:
        """Keys reachable from ``key`` within ``hops`` (ignoring edge direction)."""
        if not self._g.has_node(key):
            return set()
        frontier = {key}
        seen = {key}
        for _ in range(max(hops, 0)):
            nxt: set[str] = set()
            for node in frontier:
                nxt.update(self._g.successors(node))
                nxt.update(self._g.predecessors(node))
            nxt -= seen
            seen.update(nxt)
            frontier = nxt
            if not frontier:
                break
        return seen - {key}

    def relations_for(self, keys: set[str]) -> list[tuple[str, str, str, str]]:
        """Edges touching any key in ``keys`` as (source_display, label,
        target_display, doc_id)."""
        out: list[tuple[str, str, str, str]] = []
        for s, t, data in self._g.edges(data=True):
            if s in keys or t in keys:
                out.append(
                    (
                        self._g.nodes[s].get("display", s),
                        data.get("label", ""),
                        self._g.nodes[t].get("display", t),
                        data.get("doc_id", ""),
                    )
                )
        return out

    def context_for_entities(self, names: list[str], hops: int = 2) -> str:
        """Build a textual relation list around the named entities (graph context).

        Resolves each name to matching entities, expands by ``hops``, and renders
        the touching relations as ``source -[label]-> target`` lines.
        """
        keys: set[str] = set()
        for name in names:
            matches = self.find_entities(name)
            for m in matches:
                keys.add(m)
                keys.update(self.neighbors(m, hops))
        if not keys:
            return ""
        lines = sorted(
            {f"{s} -[{label}]-> {t}" for s, label, t, _ in self.relations_for(keys)}
        )
        return "\n".join(lines)

    def all_relations(self) -> list[tuple[str, str, str, str]]:
        """All edges as (source_display, label, target_display, doc_id)."""
        return [
            (
                self._g.nodes[s].get("display", s),
                d.get("label", ""),
                self._g.nodes[t].get("display", t),
                d.get("doc_id", ""),
            )
            for s, t, d in self._g.edges(data=True)
        ]

    def two_hop_paths(self, limit: int = 50) -> list[tuple[str, str, str, str, str]]:
        """Sample two-hop paths ``A -[r1]-> B -[r2]-> C`` (for multi-hop questions).

        Returns tuples ``(A_display, r1, B_display, r2, C_display)`` where A and C
        differ, so the question genuinely needs the intermediate node B.
        """
        out: list[tuple[str, str, str, str, str]] = []
        for b in self._g.nodes():
            for a, _, d1 in self._g.in_edges(b, data=True):
                for _, c, d2 in self._g.out_edges(b, data=True):
                    if a == c:
                        continue
                    out.append(
                        (
                            self._g.nodes[a].get("display", a),
                            d1.get("label", ""),
                            self._g.nodes[b].get("display", b),
                            d2.get("label", ""),
                            self._g.nodes[c].get("display", c),
                        )
                    )
                    if len(out) >= limit:
                        return out
        return out

    def save(self, path: str | Path) -> Path:
        """Persist the graph as node-link JSON."""
        import networkx as nx

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = nx.node_link_data(self._g, edges="links")
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> KnowledgeGraph:
        """Load a graph previously written by :meth:`save`."""
        import networkx as nx

        kg = cls()
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        kg._g = nx.node_link_graph(data, multigraph=True, directed=True, edges="links")
        return kg

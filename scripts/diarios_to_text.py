#!/usr/bin/env python3
"""Materialize the dom-pi gazette corpus (parquet) into text for training/eval.

Reads the ``texto`` column from a config of the local dom-pi-corpus-2025 snapshot.
Two output modes (chosen by the ``--out`` suffix):

- a directory: writes one ``.txt`` per document (for the ``TextCorpusLoader``,
  continued pre-training);
- a ``.jsonl`` file: writes one ``{"id", "text"}`` per line (a held-out
  perplexity evaluation set).

``--skip`` discards the first N documents that pass the filter, so a held-out set
can be made disjoint from the training subset (skip exactly the docs used to
train). Selection is deterministic (parquet/document order).

Usage:
    # training subset (first 2000 docs)
    python scripts/diarios_to_text.py --config curated --limit 2000 \
        --out data/processed/diarios_txt
    # held-out eval set (next 500 docs, disjoint from training)
    python scripts/diarios_to_text.py --config curated --skip 2000 --limit 500 \
        --out data/processed/diarios_heldout.jsonl
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

DEFAULT_SRC = "data/raw/dom-pi-corpus-2025"


def _iter_docs(config: str, min_tokens: int):
    """Yield ``(doc_id, text)`` for documents passing the token-count filter."""
    import pyarrow.parquet as pq

    files = sorted(glob.glob(f"{DEFAULT_SRC}/{config}/{config}-*.parquet"))
    if not files:
        raise SystemExit(f"no parquet found for config {config!r} under {DEFAULT_SRC}")
    for path in files:
        table = pq.read_table(path, columns=["id", "n_tokens", "texto"])
        ids = table.column("id").to_pylist()
        ntok = table.column("n_tokens").to_pylist()
        texts = table.column("texto").to_pylist()
        for doc_id, n, text in zip(ids, ntok, texts, strict=False):
            if not text or (n is not None and n < min_tokens):
                continue
            yield doc_id, str(text)


def materialize(
    config: str, out: Path, limit: int | None, min_tokens: int, skip: int
) -> int:
    """Write up to ``limit`` documents (after skipping ``skip``) to ``out``."""
    as_jsonl = out.suffix == ".jsonl"
    if as_jsonl:
        out.parent.mkdir(parents=True, exist_ok=True)
        handle = out.open("w", encoding="utf-8")
    else:
        out.mkdir(parents=True, exist_ok=True)
        handle = None

    seen = 0
    written = 0
    try:
        for doc_id, text in _iter_docs(config, min_tokens):
            if seen < skip:
                seen += 1
                continue
            if limit is not None and written >= limit:
                break
            if as_jsonl:
                handle.write(
                    json.dumps({"id": doc_id, "text": text}, ensure_ascii=False) + "\n"
                )
            else:
                (out / f"{doc_id}.txt").write_text(text, encoding="utf-8")
            written += 1
    finally:
        if handle is not None:
            handle.close()
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="curated", help="dataset config dir")
    parser.add_argument("--limit", type=int, default=2000, help="max docs (None=all)")
    parser.add_argument("--skip", type=int, default=0, help="skip the first N docs")
    parser.add_argument("--min-tokens", type=int, default=64, help="skip tiny docs")
    parser.add_argument("--out", type=Path, default=Path("data/processed/diarios_txt"))
    args = parser.parse_args()

    n = materialize(args.config, args.out, args.limit, args.min_tokens, args.skip)
    print(f"wrote {n} documents to {args.out}")


if __name__ == "__main__":
    main()

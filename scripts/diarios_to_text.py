#!/usr/bin/env python3
"""Materialize the dom-pi gazette corpus (parquet) into plain-text files.

Reads the ``texto`` column from a config of the local dom-pi-corpus-2025 snapshot
and writes one ``.txt`` per document under the output directory, ready for the
``TextCorpusLoader`` (continued pre-training, Q1). A ``--limit`` keeps a bounded
subset for a first, time-feasible run.

Usage:
    python scripts/diarios_to_text.py --config curated --limit 2000 \
        --out data/processed/diarios_txt
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

DEFAULT_SRC = "data/raw/dom-pi-corpus-2025"


def materialize(config: str, out_dir: Path, limit: int | None, min_tokens: int) -> int:
    import pyarrow.parquet as pq

    files = sorted(glob.glob(f"{DEFAULT_SRC}/{config}/{config}-*.parquet"))
    if not files:
        raise SystemExit(f"no parquet found for config {config!r} under {DEFAULT_SRC}")
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for path in files:
        table = pq.read_table(path, columns=["id", "n_tokens", "texto"])
        ids = table.column("id").to_pylist()
        ntok = table.column("n_tokens").to_pylist()
        texts = table.column("texto").to_pylist()
        for doc_id, n, text in zip(ids, ntok, texts, strict=False):
            if limit is not None and written >= limit:
                return written
            if not text or (n is not None and n < min_tokens):
                continue
            (out_dir / f"{doc_id}.txt").write_text(str(text), encoding="utf-8")
            written += 1
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="curated", help="dataset config dir")
    parser.add_argument("--limit", type=int, default=2000, help="max docs (None=all)")
    parser.add_argument("--min-tokens", type=int, default=64, help="skip tiny docs")
    parser.add_argument("--out", type=Path, default=Path("data/processed/diarios_txt"))
    args = parser.parse_args()

    n = materialize(args.config, args.out, args.limit, args.min_tokens)
    print(f"wrote {n} .txt files to {args.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Materialize the docentesDC dataset (jsonl/parquet) into plain text.

Reads the ``text`` column of the local ``docentesDC`` snapshot
(``vickminari/docentesDC``: fields ``text`` and ``nome_professor``) and writes it
out for feeding another model, mirroring ``scripts/diarios_to_text.py``.

Two output modes (chosen by the ``--out`` suffix):

- a directory: writes one ``.txt`` per record (for the ``TextCorpusLoader`` /
  continued pre-training);
- a ``.jsonl`` file: writes one ``{"id", "text", "nome_professor"}`` per line.

``--skip`` discards the first N records that pass the filter, so a held-out set can
be made disjoint from a training subset. Selection is deterministic (file order).

Usage:
    # whole corpus, one .txt per record
    python scripts/docentes_to_text.py --out data/processed/docentes_txt
    # first 2000 records only
    python scripts/docentes_to_text.py --limit 2000 --out data/processed/docentes_txt
    # held-out jsonl (next 500, disjoint)
    python scripts/docentes_to_text.py --skip 2000 --limit 500 \
        --out data/processed/docentes_heldout.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_SRC = "data/raw/docentesDC/docentesDC.jsonl"


def _iter_docs(src: Path, min_chars: int):
    """Yield ``(idx, text, nome_professor)`` for records passing the length filter."""
    if src.suffix == ".parquet":
        import pyarrow.parquet as pq

        table = pq.read_table(src, columns=["text", "nome_professor"])
        texts = table.column("text").to_pylist()
        nomes = table.column("nome_professor").to_pylist()
        rows = zip(texts, nomes, strict=False)
    else:
        def rows_gen():
            with src.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    yield d.get("text"), d.get("nome_professor")

        rows = rows_gen()

    for idx, (text, nome) in enumerate(rows):
        if not text or len(str(text).strip()) < min_chars:
            continue
        yield idx, str(text), (str(nome) if nome is not None else "")


def materialize(
    src: Path, out: Path, limit: int | None, min_chars: int, skip: int
) -> int:
    """Write up to ``limit`` records (after skipping ``skip``) to ``out``."""
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
        for idx, text, nome in _iter_docs(src, min_chars):
            if seen < skip:
                seen += 1
                continue
            if limit is not None and written >= limit:
                break
            if as_jsonl:
                handle.write(
                    json.dumps(
                        {"id": idx, "text": text, "nome_professor": nome},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            else:
                (out / f"{idx:06d}.txt").write_text(text, encoding="utf-8")
            written += 1
    finally:
        if handle is not None:
            handle.close()
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=Path(DEFAULT_SRC),
                        help="docentesDC source (.jsonl or .parquet)")
    parser.add_argument("--limit", type=int, default=None, help="max records (None=all)")
    parser.add_argument("--skip", type=int, default=0, help="skip the first N records")
    parser.add_argument("--min-chars", type=int, default=1,
                        help="skip records shorter than this many characters")
    parser.add_argument("--out", type=Path, default=Path("data/processed/docentes_txt"))
    args = parser.parse_args()

    n = materialize(args.src, args.out, args.limit, args.min_chars, args.skip)
    print(f"wrote {n} records to {args.out}")


if __name__ == "__main__":
    main()

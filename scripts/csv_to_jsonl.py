#!/usr/bin/env python3
"""Convert a ';'-delimited benchmark CSV into JSONL, keeping every column.

The hand-made Q1 benchmarks in ``benchmarks/a-mao/`` carry extra columns beyond the
original plan (``arquivo``/``arquivo_origem``, ``tipo_documento``, ``context``/
``target``). This preserves all of them: one JSON object per row, keys = CSV
columns. ``id`` is emitted as an int when numeric. The evaluator understands the
cloze (``context``/``target``) and Q&A (``instruction``/``output``) shapes.

Usage:
    python scripts/csv_to_jsonl.py --in "benchmarks/a-mao/Q1 benchmark P&R.csv" \
        --out benchmarks/a-mao/q1_pr.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def convert(src: Path, out: Path, delimiter: str = ";") -> int:
    rows = list(csv.DictReader(src.open(encoding="utf-8"), delimiter=delimiter))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for r in rows:
            obj = {}
            for k, v in r.items():
                if k is None:
                    continue
                v = (v or "").strip()
                if k == "id" and v.isdigit():
                    obj[k] = int(v)
                else:
                    obj[k] = v
            fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="src", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--delimiter", default=";")
    args = parser.parse_args()
    n = convert(args.src, args.out, args.delimiter)
    print(f"wrote {n} items to {args.out}")


if __name__ == "__main__":
    main()

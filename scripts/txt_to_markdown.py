#!/usr/bin/env python3
"""Bundle each per-document .txt corpus into its own formatted Markdown file.

One Markdown file is written per source dataset (``<out-dir>/<label>.md``). Inside a
file, every document becomes a section (``## <label> / <doc id>``), so the individual
.txt files stay distinguishable. Sources can be a directory of .txt files or a .jsonl
file (one ``{"id"/"text"}`` per line).

Output is streamed, so large corpora (tens of thousands of docs) do not need to fit
in memory. The files can be very large; use ``--limit`` to cap documents per dataset
for a preview.

Usage:
    # default: full diarios and full docentes, one .md each, into a directory
    python scripts/txt_to_markdown.py --out-dir data/processed/corpus_md

    # preview: 50 docs per dataset, content wrapped in code fences
    python scripts/txt_to_markdown.py --limit 50 --code-fence \
        --out-dir data/processed/corpus_md_preview

    # custom sources (label=path, repeatable)
    python scripts/txt_to_markdown.py \
        --source diarios=data/processed/diarios_txt \
        --source docentes=data/processed/docentes_txt \
        --out-dir data/processed/corpus_md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_SOURCES = [
    "diarios=data/processed/diarios_txt_full",
    "docentes=data/processed/docentes_txt",
]
# Fence unlikely to collide with document content (e.g. ``` inside code samples).
FENCE = "~~~~~~"


def _iter_source(path: Path):
    """Yield ``(doc_id, text)`` from a .txt directory or a .jsonl file."""
    if path.is_dir():
        for fp in sorted(path.glob("*.txt")):
            yield fp.stem, fp.read_text(encoding="utf-8")
    elif path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                yield str(d.get("id", i)), str(d.get("text", ""))
    else:
        raise SystemExit(f"unsupported source (need a dir or .jsonl): {path}")


def bundle_one(label: str, path: Path, out: Path, limit: int, code_fence: bool) -> int:
    """Write one dataset to ``out`` as Markdown. Returns the document count."""
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w", encoding="utf-8") as w:
        w.write(f"# Dataset: {label}\n\n")
        w.write("Gerado por `scripts/txt_to_markdown.py`. ")
        w.write(f"Fonte: `{path}`. Cada documento e uma secao.\n\n")
        for doc_id, text in _iter_source(path):
            if limit and n >= limit:
                break
            w.write(f"## {label} / {doc_id}\n\n")
            if code_fence:
                w.write(f"{FENCE}\n{text.rstrip()}\n{FENCE}\n\n")
            else:
                w.write(text.rstrip() + "\n\n")
            w.write("---\n\n")
            n += 1
    return n


def bundle(sources: list[tuple[str, Path]], out_dir: Path, limit: int, code_fence: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for label, path in sources:
        out = out_dir / f"{label}.md"
        n = bundle_one(label, path, out, limit, code_fence)
        print(f"wrote {out} ({n} docs)")


def _parse_source(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise SystemExit(f"--source must be label=path, got {spec!r}")
    label, path = spec.split("=", 1)
    return label, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", default=None,
                        help="label=path (dir of .txt or .jsonl); repeatable")
    parser.add_argument("--limit", type=int, default=0,
                        help="max docs per dataset (0 = all)")
    parser.add_argument("--code-fence", action="store_true",
                        help="wrap each document in a code fence (keeps raw formatting)")
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed/corpus_md"),
                        help="output directory; one <label>.md is written per dataset")
    args = parser.parse_args()

    specs = args.source if args.source else DEFAULT_SOURCES
    sources = [_parse_source(s) for s in specs]
    bundle(sources, args.out_dir, args.limit, args.code_fence)


if __name__ == "__main__":
    main()

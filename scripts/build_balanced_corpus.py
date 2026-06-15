#!/usr/bin/env python3
"""Build a licitacao-downsampled copy of the diarios text corpus for Q1.

Keeps every non-licitacao document and a random ``--keep-fraction`` of the
licitacao documents (deterministic for a fixed ``--seed``). Only the licitacao
class is touched; non-licitacao documents are all preserved. The selected files
are linked (symlink by default) into ``--out-dir`` so the source corpus is left
intact and no text is duplicated on disk.

Usage:
    python scripts/build_balanced_corpus.py \
        --src-dir data/processed/diarios_txt \
        --out-dir data/processed/diarios_txt_balanced \
        --keep-fraction 0.5 --seed 42
"""

from __future__ import annotations

import argparse
import glob
import shutil
from pathlib import Path

from llm_finetuning.rag.doc_select import downsample_licitacao, is_licitacao


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-dir", type=Path, default=Path("data/processed/diarios_txt"))
    parser.add_argument(
        "--out-dir", type=Path, default=Path("data/processed/diarios_txt_balanced")
    )
    parser.add_argument(
        "--keep-fraction",
        type=float,
        default=0.5,
        help="fraction of licitacao documents to keep (0..1)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-hits", type=int, default=2)
    parser.add_argument(
        "--copy", action="store_true", help="copy files instead of symlinking them"
    )
    args = parser.parse_args()

    files = sorted(glob.glob(f"{args.src_dir}/*.txt"))
    if not files:
        raise SystemExit(f"no .txt files under {args.src_dir}")

    docs: list[tuple[str, str]] = [
        (f, Path(f).read_text(encoding="utf-8", errors="ignore")) for f in files
    ]
    n_lic_in = sum(1 for _, t in docs if is_licitacao(t, args.min_hits))

    selected = downsample_licitacao(
        docs, keep_fraction=args.keep_fraction, seed=args.seed, min_hits=args.min_hits
    )
    n_lic_out = sum(1 for _, t in selected if is_licitacao(t, args.min_hits))

    out_dir = args.out_dir
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    for src_path, _ in selected:
        src = Path(src_path).resolve()
        dst = out_dir / src.name
        if args.copy:
            shutil.copy2(src, dst)
        else:
            dst.symlink_to(src)

    print(
        f"source: {len(docs)} docs ({n_lic_in} licitacao, {len(docs) - n_lic_in} outros)\n"
        f"balanced: {len(selected)} docs ({n_lic_out} licitacao, "
        f"{len(selected) - n_lic_out} outros) -> {out_dir}"
    )


if __name__ == "__main__":
    main()

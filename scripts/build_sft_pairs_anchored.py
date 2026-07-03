#!/usr/bin/env python3
"""Q2 (v2): source-anchored, deep-question SFT pairs from the docentesDC dataset.

Same source cleaning/split methodology as ``build_sft_pairs.py`` (exact dedup,
drop garbled/short texts, disjoint train/held-out source pools), but generates
exactly one deep, trecho-specific pair per document (not a shallow definition
question) and writes two files per split:

- ``<out-dir>/docentes_sft_{train,heldout}.jsonl``: SFT-ready
  ``{instruction, input, output}``, same schema/path as before, so existing
  ``sft_docentes_*.yaml`` / ``lora_docentes_*.yaml`` configs need no changes.
- ``<out-dir>/docentes_sft_{train,heldout}_sources.jsonl``: the same pairs plus
  ``professor``, ``doc_index`` and ``source_excerpt`` for traceability/audit.

Any previous ``docentes_sft_{train,heldout,recall}.jsonl`` in ``out-dir`` is
archived (moved, not deleted) into ``<out-dir>/old/`` before writing.

Default engine is the largest local instruct model, ``gemma-4-31b-it`` (4-bit
NF4, ~16 GB on one L4).

A model response that is not the required JSON (malformed/refused) or that is
a shallow/echoed pair does not stop the run: it is counted and the loop moves
to the next document (see ``generate_pairs_for_records`` in
``llm_finetuning.data.anchored_qa_pairs``). Only repeated *engine* failures
(``llm.chat`` itself raising, e.g. a bad model path or OOM) abort the run, since
that pattern means the engine is broken, not a one-off bad document.

Usage (does not run automatically; invoke explicitly when ready):
    python scripts/build_sft_pairs_anchored.py --n-train 1000 --n-heldout 150
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from llm_finetuning.data.anchored_qa_pairs import (
    GenerationAborted,
    archive_existing_outputs,
    generate_pairs_for_records,
    to_sft_record,
)
from llm_finetuning.data.sft_pairs import clean_source_records


def _load(src: Path) -> list[dict]:
    rows = []
    for line in src.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _log_progress(label: str) -> callable:
    def _cb(event: dict) -> None:
        if event.get("event") == "error":
            print(f"  [{label}] doc {event['doc_index']}: llm.chat failed "
                  f"({event['errors']} total): {event['error']}", flush=True)
        elif event["calls"] % 10 == 0:
            print(f"  [{label}] {event['calls']} docs, {event['unique']} unique pairs "
                  f"({event['malformed']} malformed, {event['shallow']} shallow, "
                  f"{event['errors']} errors)", flush=True)
    return _cb


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=Path("data/raw/docentesDC/docentesDC.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed/sft"))
    parser.add_argument("--model", default="models/gemma-4-31b-it")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--load-in-4bit", action="store_true", default=True)
    parser.add_argument("--n-train", type=int, default=1000)
    parser.add_argument("--n-heldout", type=int, default=150)
    parser.add_argument("--max-source-chars", type=int, default=1800)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--eval-fraction", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-prefix", default="docentes_sft")
    args = parser.parse_args()

    raw = _load(args.src)
    records = clean_source_records(raw)
    rng = random.Random(args.seed)
    rng.shuffle(records)
    n_eval = int(len(records) * args.eval_fraction)
    eval_src, train_src = records[:n_eval], records[n_eval:]
    print(f"clean source: {len(records)} records -> {len(train_src)} train-source, "
          f"{len(eval_src)} eval-source", flush=True)

    old_dir = args.out_dir / "old"
    archive_existing_outputs(
        [args.out_dir / f"{args.out_prefix}_{name}.jsonl"
         for name in ("train", "heldout", "recall")],
        old_dir,
    )

    from llm_finetuning.rag.llm_client import LocalChatLLM

    llm = LocalChatLLM(model_name=args.model, device=args.device,
                       device_map=args.device_map, load_in_4bit=args.load_in_4bit,
                       max_new_tokens=args.max_new_tokens, temperature=0.7)

    try:
        train_pairs, train_sources, train_stats = generate_pairs_for_records(
            llm, train_src, args.n_train, args.max_source_chars, args.max_new_tokens,
            on_progress=_log_progress("train"))
        held_pairs, held_sources, held_stats = generate_pairs_for_records(
            llm, eval_src, args.n_heldout, args.max_source_chars, args.max_new_tokens,
            on_progress=_log_progress("heldout"))
    except GenerationAborted as exc:
        print(f"ABORTED: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1) from exc

    print(f"train stats: {train_stats}")
    print(f"heldout stats: {held_stats}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, pairs, sources in (
        ("train", train_pairs, train_sources),
        ("heldout", held_pairs, held_sources),
    ):
        sft_out = args.out_dir / f"{args.out_prefix}_{name}.jsonl"
        with sft_out.open("w", encoding="utf-8") as fh:
            for p in pairs:
                fh.write(json.dumps(to_sft_record(p), ensure_ascii=False) + "\n")
        print(f"wrote {len(pairs)} pairs -> {sft_out}")

        src_out = args.out_dir / f"{args.out_prefix}_{name}_sources.jsonl"
        with src_out.open("w", encoding="utf-8") as fh:
            for s in sources:
                fh.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"wrote {len(sources)} source records -> {src_out}")


if __name__ == "__main__":
    main()

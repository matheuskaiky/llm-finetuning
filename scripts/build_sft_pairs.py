#!/usr/bin/env python3
"""Q2 Phase 1: generate the SFT instruction pairs from the docentesDC dataset.

Cleans the official dataset (exact dedup, drop garbled/short texts), splits the
source records into disjoint train/held-out pools (so eval pairs do not leak from
training texts), and asks an instruct LLM to produce grounded
``{instruction, input, output}`` pairs per source excerpt. Output:
``<out-dir>/docentes_sft_train.jsonl`` and ``docentes_sft_heldout.jsonl``.

Usage (generator on the free GPU; does not touch GPU0):
    CUDA_VISIBLE_DEVICES=1 python scripts/build_sft_pairs.py \
        --n-train 1000 --n-heldout 150
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from llm_finetuning.data.sft_pairs import (
    clean_source_records,
    dedup_pairs,
    parse_qa_pairs,
    subtract_pairs,
)

SYS = (
    "Voce cria dados de instrucao (instruction tuning) em portugues a partir do "
    "material academico de um professor do Departamento de Computacao. Dado o nome "
    "do professor e um trecho do material dele, gere {k} pares pergunta-e-resposta "
    "DIVERSOS e ESPECIFICOS, com a resposta fundamentada APENAS no trecho. Nao "
    "invente fatos fora do trecho. Nao escreva 'segundo o texto' nem 'no trecho'; "
    "as perguntas devem ser autonomas. Responda APENAS com um array JSON: "
    '[{{"instruction": "...", "input": "", "output": "..."}}, ...]. Use input vazio '
    "salvo quando um contexto curto for necessario."
)


def _load_txt_dir(src_dir: Path, limit: int = 0) -> list[dict]:
    """Read a directory of .txt files into ``[{"text": ...}]`` records."""
    import glob

    files = sorted(glob.glob(f"{src_dir}/*.txt"))
    if limit:
        files = files[:limit]
    return [
        {"text": Path(f).read_text(encoding="utf-8", errors="ignore"),
         "nome_professor": Path(f).stem}
        for f in files
    ]


def _load(src: Path, text_key: str) -> list[dict]:
    rows = []
    for line in src.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _unique(collected, exclude):
    u = dedup_pairs(collected)
    return subtract_pairs(u, exclude) if exclude else u


def _generate(llm, records, target, pairs_per_text, max_chars, max_new_tokens, label,
              exclude=None):
    """Generate grounded pairs from ``records`` until ``target`` unique pairs.

    With ``exclude`` (a list of pairs), questions matching those are not counted or
    returned, so an in-domain recall set excludes the training questions.
    """
    sys_prompt = SYS.format(k=pairs_per_text)
    collected: list[dict] = []
    calls = 0
    for rec in records:
        if len(_unique(collected, exclude)) >= target:
            break
        text = rec.get("text", "")[:max_chars]
        nome = rec.get("nome_professor", "?")
        raw = llm.chat(
            [{"role": "system", "content": sys_prompt},
             {"role": "user", "content": f"Professor: {nome}\nTrecho:\n{text}\n\nJSON:"}],
            max_new_tokens=max_new_tokens,
        )
        for p in parse_qa_pairs(raw):
            p["professor"] = nome
            collected.append(p)
        calls += 1
        if calls % 10 == 0:
            print(f"  [{label}] {calls} calls, {len(_unique(collected, exclude))} unique pairs",
                  flush=True)
    return _unique(collected, exclude)[:target]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=Path("data/raw/docentesDC/docentesDC.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed/sft"))
    parser.add_argument("--model", default="models/Qwen3-8B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-map", default=None,
                        help="device_map for large teachers (e.g. 'auto' to model-parallel a 27B/30B)")
    parser.add_argument("--load-in-4bit", action="store_true",
                        help="4-bit NF4 load for big bf16 teachers (gemma-27b/31b) on L4s")
    parser.add_argument("--n-train", type=int, default=1000)
    parser.add_argument("--n-heldout", type=int, default=150)
    parser.add_argument("--pairs-per-text", type=int, default=2)
    parser.add_argument("--max-source-chars", type=int, default=1800)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--eval-fraction", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--recall", action="store_true",
        help="only build an in-domain recall held-out from the TRAIN source texts, "
             "excluding the existing training questions (no exact leakage)",
    )
    parser.add_argument("--n-recall", type=int, default=150)
    parser.add_argument(
        "--src-txt-dir", type=Path, default=None,
        help="read source texts from a directory of .txt files (e.g. the diarios "
             "corpus for Q4) instead of the --src JSONL",
    )
    parser.add_argument("--src-txt-limit", type=int, default=0)
    parser.add_argument("--out-prefix", default="docentes_sft",
                        help="output file prefix (e.g. diarios_distill for Q4)")
    args = parser.parse_args()

    raw = (_load_txt_dir(args.src_txt_dir, args.src_txt_limit)
           if args.src_txt_dir else _load(args.src, "text"))
    records = clean_source_records(raw)
    rng = random.Random(args.seed)
    rng.shuffle(records)
    n_eval = int(len(records) * args.eval_fraction)
    eval_src, train_src = records[:n_eval], records[n_eval:]
    print(f"clean source: {len(records)} records -> {len(train_src)} train-source, "
          f"{len(eval_src)} eval-source", flush=True)

    from llm_finetuning.rag.llm_client import LocalChatLLM

    llm = LocalChatLLM(model_name=args.model, device=args.device,
                       device_map=args.device_map, load_in_4bit=args.load_in_4bit,
                       max_new_tokens=args.max_new_tokens, temperature=0.7)

    if args.recall:
        # In-domain recall: new questions from the SAME source texts that produced
        # the training pairs (train_src in order; the train loop consumed its first
        # texts), excluding the exact training questions. The model saw this content
        # during SFT, so this measures recall of injected domain knowledge.
        train_pairs = _load(args.out_dir / f"{args.out_prefix}_train.jsonl", "instruction")
        recall = _generate(llm, train_src, args.n_recall, args.pairs_per_text,
                           args.max_source_chars, args.max_new_tokens, "recall",
                           exclude=train_pairs)
        out = args.out_dir / f"{args.out_prefix}_recall.jsonl"
        with out.open("w", encoding="utf-8") as fh:
            for p in recall:
                fh.write(json.dumps(p, ensure_ascii=False) + "\n")
        print(f"wrote {len(recall)} recall pairs -> {out}")
        return

    train_pairs = _generate(llm, train_src, args.n_train, args.pairs_per_text,
                            args.max_source_chars, args.max_new_tokens, "train")
    held_pairs = _generate(llm, eval_src, args.n_heldout, args.pairs_per_text,
                           args.max_source_chars, args.max_new_tokens, "heldout")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, pairs in (("train", train_pairs), ("heldout", held_pairs)):
        out = args.out_dir / f"{args.out_prefix}_{name}.jsonl"
        with out.open("w", encoding="utf-8") as fh:
            for p in pairs:
                fh.write(json.dumps(p, ensure_ascii=False) + "\n")
        print(f"wrote {len(pairs)} pairs -> {out}")


if __name__ == "__main__":
    main()

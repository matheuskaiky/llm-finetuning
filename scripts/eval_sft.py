#!/usr/bin/env python3
"""Q2 SFT evaluation: judge score (0-5) + response perplexity on the held-out set.

For each model (base, SFT checkpoint, Q1-checkpoint+SFT, instruct reference) the
script generates an answer to every held-out instruction with the same SFT prompt
template, then a fixed judge LLM scores it against the reference output. It also
reports the teacher-forced perplexity of the reference response. Models are loaded
one at a time (then unloaded); the judge is loaded once at the end.

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/eval_sft.py \
        --models base=models/Qwen3-0.6B-Base sft_base=outputs/sft_qwen3_0p6b_base \
                 sft_q1=outputs/sft_qwen3_0p6b_q1 \
        --out results/q2_sft_eval.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm_finetuning.core import set_global_seed
from llm_finetuning.data.sft_pairs import build_prompt
from llm_finetuning.evaluation.results import (
    DEFAULT_EVAL_RUNS,
    run_seeds,
    summarize_runs,
    write_item_results,
)
from llm_finetuning.rag.judge import llm_judge
from llm_finetuning.rag.llm_client import LocalChatLLM


def _load_heldout(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heldout", type=Path,
                        default=Path("data/processed/sft/docentes_sft_heldout.jsonl"))
    parser.add_argument("--models", nargs="+", required=True,
                        help="label=path entries for the models under test")
    parser.add_argument("--judge-model", default="models/Qwen3-8B")
    parser.add_argument("--device", default="cuda", help="device for the tested models")
    parser.add_argument("--judge-device", default="cuda")
    parser.add_argument("--judge-load-in-4bit", action="store_true",
                        help="4-bit NF4 load for a big judge (e.g. gemma-4-31b-it)")
    parser.add_argument("--judge-device-map", default=None,
                        help="device_map for a big judge (e.g. 'auto' to split a 31b over both L4)")
    parser.add_argument("--out", type=Path, default=Path("results/q2_sft_eval.csv"))
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, default=0, help="limit held-out items (0=all)")
    parser.add_argument("--runs", type=int, default=DEFAULT_EVAL_RUNS,
                        help="mandatory number of evaluation repetitions (default 5)")
    parser.add_argument("--seed", type=int, default=42, help="base seed (run r uses seed+r-1)")
    args = parser.parse_args()

    items = _load_heldout(args.heldout)
    if args.limit:
        items = items[: args.limit]
    print(f"held-out items: {len(items)} | runs: {args.runs}", flush=True)

    specs = [m.split("=", 1) for m in args.models]
    seeds = run_seeds(args.seed, args.runs)

    # Pass 1: per model, load once, generate answer + response perplexity for every
    # (run, item). Keyed by (label, run, item index).
    gen: dict[tuple[str, int, int], dict] = {}
    for label, path in specs:
        print(f"=== generating: {label} ({path}) ===", flush=True)
        llm = LocalChatLLM(model_name=path, device=args.device, temperature=0.0,
                           max_new_tokens=args.max_new_tokens)
        for run, seed in enumerate(seeds, start=1):
            set_global_seed(seed)
            for idx, it in enumerate(items):
                prompt = build_prompt(it["instruction"], it.get("input", ""))
                ans = llm.complete(prompt, args.max_new_tokens)
                ppl = llm.response_perplexity(prompt, it["output"])
                gen[(label, run, idx)] = {"answer": ans, "resp_ppl": ppl}
        llm.unload()

    # Pass 2: fixed judge scores every generated answer against the reference.
    print(f"=== judging with {args.judge_model} ===", flush=True)
    judge = LocalChatLLM(model_name=args.judge_model, device=args.judge_device,
                         temperature=0.0, load_in_4bit=args.judge_load_in_4bit,
                         device_map=args.judge_device_map)
    per_item: list[dict] = []
    for label, _ in specs:
        for run, seed in enumerate(seeds, start=1):
            set_global_seed(seed)
            for idx, it in enumerate(items):
                g = gen[(label, run, idx)]
                s = llm_judge(judge, it["instruction"], it["output"], g["answer"])
                per_item.append({
                    "run": run, "id": it.get("id", idx + 1), "model": label,
                    "instruction": it["instruction"], "expected": it["output"],
                    "answer": g["answer"], "judge": s, "resp_ppl": g["resp_ppl"],
                })
    judge.unload()

    write_item_results(
        args.out, per_item,
        fieldnames=["run", "id", "model", "instruction", "expected", "answer",
                    "judge", "resp_ppl"],
    )
    summary = summarize_runs(per_item, ["model"], ["judge", "resp_ppl"])
    summary_path = args.out.with_name(args.out.stem + "_summary.csv")
    write_item_results(summary_path, summary)
    for row in summary:
        print(f"  {row['model']}: judge={row['judge_mean']:.3f}+-{row['judge_std']:.3f}  "
              f"resp_ppl={row['resp_ppl_mean']:.3f}", flush=True)
    print(f"wrote {args.out} (per-item, {len(per_item)} linhas) e {summary_path}")


if __name__ == "__main__":
    main()

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
import csv
import json
from pathlib import Path

from llm_finetuning.data.sft_pairs import build_prompt
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
    parser.add_argument("--out", type=Path, default=Path("results/q2_sft_eval.csv"))
    parser.add_argument("--details", type=Path, default=None,
                        help="optional JSONL path for per-item answers/scores")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, default=0, help="limit held-out items (0=all)")
    args = parser.parse_args()

    items = _load_heldout(args.heldout)
    if args.limit:
        items = items[: args.limit]
    print(f"held-out items: {len(items)}", flush=True)

    specs = [m.split("=", 1) for m in args.models]

    # Pass 1: per model, generate answers and response perplexity, then unload.
    answers: dict[str, list[dict]] = {}
    for label, path in specs:
        print(f"=== generating: {label} ({path}) ===", flush=True)
        llm = LocalChatLLM(model_name=path, device=args.device, temperature=0.0,
                           max_new_tokens=args.max_new_tokens)
        rows = []
        for it in items:
            prompt = build_prompt(it["instruction"], it.get("input", ""))
            ans = llm.complete(prompt, args.max_new_tokens)
            ppl = llm.response_perplexity(prompt, it["output"])
            rows.append({"answer": ans, "resp_ppl": ppl})
        answers[label] = rows
        llm.unload()

    # Pass 2: fixed judge scores every model's answers against the reference.
    print(f"=== judging with {args.judge_model} ===", flush=True)
    judge = LocalChatLLM(model_name=args.judge_model, device=args.judge_device,
                         temperature=0.0)
    summary = []
    details = []
    for label, _ in specs:
        scores, ppls = [], []
        for it, row in zip(items, answers[label], strict=False):
            s = llm_judge(judge, it["instruction"], it["output"], row["answer"])
            scores.append(s)
            if row["resp_ppl"] == row["resp_ppl"]:  # not NaN
                ppls.append(row["resp_ppl"])
            details.append({"model": label, "instruction": it["instruction"],
                            "expected": it["output"], "answer": row["answer"],
                            "score": s, "resp_ppl": row["resp_ppl"]})
        mean_score = sum(scores) / len(scores) if scores else 0.0
        mean_ppl = sum(ppls) / len(ppls) if ppls else float("nan")
        summary.append({"model": label, "mean_judge": round(mean_score, 3),
                        "mean_resp_ppl": round(mean_ppl, 3), "n": len(scores)})
        print(f"  {label}: judge={mean_score:.3f}/5  resp_ppl={mean_ppl:.3f}", flush=True)
    judge.unload()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["model", "mean_judge", "mean_resp_ppl", "n"])
        w.writeheader()
        w.writerows(summary)
    print(f"wrote {args.out}")
    if args.details:
        args.details.parent.mkdir(parents=True, exist_ok=True)
        with args.details.open("w", encoding="utf-8") as fh:
            for d in details:
                fh.write(json.dumps(d, ensure_ascii=False) + "\n")
        print(f"wrote {args.details}")


if __name__ == "__main__":
    main()

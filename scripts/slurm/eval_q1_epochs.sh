#!/bin/bash
# Evaluate the Q1 hand-made benchmarks (Cloze + P&R) at every trained epoch.
#
# The pre-training config saves one checkpoint per epoch (save_strategy: epoch), so
# this evaluates base (antes), each epoch checkpoint (ep1..epN) and the instruct
# reference, with 5 runs, per-id, into a single CSV per benchmark. The epoch shows
# up as the model label (ep1/ep2/ep3), giving the epochs curve directly.
#
# Usage: eval_q1_epochs.sh <ckpt_dir> <base_model> <instruct_model> <tag>
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/aluno_matheus/Code/llm-finetuning}"

CKPT="${1:?ckpt dir}"; BASE="${2:?base}"; INST="${3:?instruct}"; TAG="${4:?tag}"

MODELS=(base="${BASE}")
i=1
for ck in $(ls -d "${CKPT}"/checkpoint-* 2>/dev/null | sort -t- -k2 -n); do
  MODELS+=("ep${i}=${ck}"); i=$((i + 1))
done
# If no per-epoch checkpoints exist, fall back to the final model at the root.
[ "${i}" -eq 1 ] && MODELS+=(depois="${CKPT}")
MODELS+=(instruct="${INST}")
echo "eval models: ${MODELS[*]}"

for pair in cloze:diarios_cloze pr:diarios_qa; do
  name="${pair%%:*}"; file="${pair##*:}"
  .venv/bin/python scripts/eval_q1_amao.py \
    --benchmark "benchmarks/pre_treino/${file}.jsonl" \
    --models "${MODELS[@]}" --device cuda --runs 5 \
    --out "results/q1_amao_${name}_${TAG}.csv"
done

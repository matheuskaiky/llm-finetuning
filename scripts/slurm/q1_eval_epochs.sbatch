#!/bin/bash
#SBATCH --job-name=q1-eval-epochs
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=28G
#SBATCH --time=08:00:00
#SBATCH --output=logs/q1_eval_%x_%j.out
#SBATCH --error=logs/q1_eval_%x_%j.err

# Final combined evaluation of a chained epoch run: base (antes), each epoch dir
# (ep1..epN) and the instruct reference, on the Cloze and P&R benchmarks, 5 runs,
# per-id, one CSV per benchmark (epoch is the `model` column). Chain with
# --dependency=afterok:<last epoch job>.
#   $1 = tag (result CSV suffix)
#   $2 = base model    $3 = instruct model
#   $4.. = epoch output dirs, in order (ep1, ep2, ...)

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/aluno_matheus/Code/llm-finetuning}"
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false

TAG="${1:?tag}"; BASE="${2:?base}"; INST="${3:?instruct}"; shift 3

MODELS=(base="${BASE}")
i=1
for d in "$@"; do MODELS+=("ep${i}=${d}"); i=$((i + 1)); done
MODELS+=(instruct="${INST}")
echo "EVAL $(date -Is) tag=${TAG} models=${MODELS[*]}"

for pair in cloze:diarios_cloze pr:diarios_qa; do
  name="${pair%%:*}"; file="${pair##*:}"
  .venv/bin/python scripts/eval_q1_amao.py \
    --benchmark "benchmarks/pre_treino/${file}.jsonl" \
    --models "${MODELS[@]}" --device cuda --runs 5 \
    --out "results/q1_amao_${name}_${TAG}.csv"
done
echo "EVAL_DONE $(date -Is)"

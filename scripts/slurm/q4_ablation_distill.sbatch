#!/bin/bash
#SBATCH --job-name=q4-abl-distill
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=28G
#SBATCH --time=08:00:00
#SBATCH --output=logs/q4_abl_distill_%x_%j.out
#SBATCH --error=logs/q4_abl_distill_%x_%j.err

# One student distillation for the Q4 pairs ablation: response-based SFT of a
# student on a given synthetic-pairs subset. Only the pairs file changes across the
# ablation; the student model stays fixed so #pairs is the sole variable.
#   $1 = pairs jsonl (e.g. data/processed/distill/diarios_distill_g31b_train_3000.jsonl)
#   $2 = output dir  (e.g. outputs/distill_g31b_3000)
#   $3 = student model (optional; default = the config's student)

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/aluno_matheus/Code/llm-finetuning}"
export TOKENIZERS_PARALLELISM=false

PAIRS="${1:?pairs jsonl}"; OUT="${2:?output dir}"; STUDENT="${3:-}"
EXTRA=(); [ -n "${STUDENT}" ] && EXTRA+=(--model-name "${STUDENT}")

echo "== DISTILL $(date -Is) pairs=${PAIRS} -> ${OUT} =="
T0=$(date +%s)
.venv/bin/python scripts/train.py --config configs/distill_diarios_response.yaml \
  --data-path "${PAIRS}" --output-dir "${OUT}" "${EXTRA[@]}"
echo "== DISTILL_DONE $(date -Is) elapsed_s=$(( $(date +%s) - T0 )) =="

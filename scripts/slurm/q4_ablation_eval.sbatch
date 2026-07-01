#!/bin/bash
#SBATCH --job-name=q4-abl-eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:l4:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=58G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/q4_abl_eval_%j.out
#SBATCH --error=logs/q4_abl_eval_%j.err

# Q4 pairs ablation, evaluation. Judge = gemma-4-31b-it (LLM-as-a-judge, 4-bit NF4
# split over both L4 via device_map=auto). Scores the teacher, the base student and
# the students distilled at 1200/3000/6000 pairs, on the in-domain recall set (and
# the disjoint heldout). 5 runs, per-id, one CSV per set (the #pairs is the model
# label: distill_1200/3000/6000). Requests 2 L4 so the 31b judge fits.
# The tested students are loaded/unloaded one at a time; only the judge is resident
# at scoring time, so 4-bit 31b alone fits.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/aluno_matheus/Code/llm-finetuning}"
export TOKENIZERS_PARALLELISM=false

JUDGE=models/gemma-4-31b-it
STUDENT_BASE=models/Qwen3-0.6B-Base   # keep in sync with the distill config's student
MODELS=(
  teacher_g31b="${JUDGE}"
  base="${STUDENT_BASE}"
  distill_1200=outputs/distill_g31b_1200
  distill_3000=outputs/distill_g31b_3000
  distill_6000=outputs/distill_g31b_6000
)

for set in recall heldout; do
  echo "== EVAL ${set} (judge gemma-4-31b-it 4-bit) $(date -Is) =="
  .venv/bin/python scripts/eval_sft.py \
    --heldout "data/processed/distill/diarios_distill_g31b_${set}.jsonl" \
    --models "${MODELS[@]}" \
    --judge-model "${JUDGE}" --judge-load-in-4bit --judge-device-map auto \
    --device cuda --runs 5 \
    --out "results/q4_ablation_${set}.csv"
done
echo "Q4_ABL_EVAL_DONE $(date -Is)"

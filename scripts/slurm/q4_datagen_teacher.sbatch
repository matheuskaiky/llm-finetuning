#!/bin/bash
#SBATCH --job-name=q4-datagen-teacher
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/q4_datagen_teacher_%j.out
#SBATCH --error=logs/q4_datagen_teacher_%j.err

# Q4 teacher-comparison: generate a fixed-budget (400-pair) distillation train set
# with one teacher. Eval stays on the existing fixed recall benchmark
# (diarios_distill_recall.jsonl), so only the train set is regenerated per teacher.
# Args passed through to build_sft_pairs (e.g. --model, --out-prefix, --load-in-4bit,
# --device-map auto). One L4 fits an 8B bf16 or a 27B/31B in 4-bit.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/aluno_matheus/Code/llm-finetuning}"
export TOKENIZERS_PARALLELISM=false

.venv/bin/python scripts/build_sft_pairs.py \
  --src-txt-dir data/processed/diarios_txt \
  --out-dir data/processed/distill \
  --n-train 400 --n-heldout 2 --pairs-per-text 3 --max-new-tokens 420 \
  "$@"
echo "Q4_DATAGEN_TEACHER_DONE"

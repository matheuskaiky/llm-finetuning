#!/bin/bash
#SBATCH --job-name=q4-datagen-diarios
#SBATCH --partition=gpu
#SBATCH --nodelist=gpunode02
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=28G
#SBATCH --time=12:00:00
#SBATCH --output=logs/q4_datagen_%j.out
#SBATCH --error=logs/q4_datagen_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/aluno_matheus/Code/llm-finetuning}"
export TOKENIZERS_PARALLELISM=false
PY=.venv/bin/python
# 1) distill corpus + disjoint heldout from the diarios .txt corpus
$PY scripts/build_sft_pairs.py --src-txt-dir data/processed/diarios_txt \
  --out-dir data/processed/distill --out-prefix diarios_distill \
  --n-train 1200 --n-heldout 100 --pairs-per-text 3 --max-new-tokens 420
# 2) in-domain recall benchmark (100 Qs from the train sources, excluding train Qs)
$PY scripts/build_sft_pairs.py --src-txt-dir data/processed/diarios_txt \
  --out-dir data/processed/distill --out-prefix diarios_distill \
  --recall --n-recall 100 --pairs-per-text 3 --max-new-tokens 420
echo "Q4_DATAGEN_DONE"

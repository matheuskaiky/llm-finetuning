#!/bin/bash
#SBATCH --job-name=q4-distill
#SBATCH --partition=gpu
#SBATCH --nodelist=gpunode02
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=28G
#SBATCH --time=4:00:00
#SBATCH --output=logs/q4_distill_%x_%j.out
#SBATCH --error=logs/q4_distill_%x_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/aluno_matheus/Code/llm-finetuning}"
export TOKENIZERS_PARALLELISM=false
.venv/bin/python scripts/train.py --config configs/distill_diarios_response.yaml \
  --model-name "$1" --output-dir "$2"

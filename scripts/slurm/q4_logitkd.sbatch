#!/bin/bash
#SBATCH --job-name=q4-logitkd
#SBATCH --partition=gpu
#SBATCH --nodelist=gpunode02
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=28G
#SBATCH --time=4:00:00
#SBATCH --output=logs/q4_logitkd_%j.out
#SBATCH --error=logs/q4_logitkd_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/aluno_matheus/Code/llm-finetuning}"
export TOKENIZERS_PARALLELISM=false
.venv/bin/python scripts/train.py --config configs/distill_logitkd_qwen3_0p6b.yaml
echo "Q4_LOGITKD_DONE"

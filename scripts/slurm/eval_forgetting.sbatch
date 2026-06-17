#!/bin/bash
#SBATCH --job-name=q1-forgetting
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=logs/q1_forgetting_%j.out
#SBATCH --error=logs/q1_forgetting_%j.err

# Q1 catastrophic-forgetting probe: out-of-domain (docentesDC) perplexity for the
# base ladder before vs after continued pre-training. One L4 is enough (models are
# small: 0.6B/1.7B/1B).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/aluno_matheus/Code/llm-finetuning}"
export TOKENIZERS_PARALLELISM=false

.venv/bin/python scripts/eval_forgetting.py \
  --probe-size 150 --max-length 512 --out results/q1_forgetting.csv

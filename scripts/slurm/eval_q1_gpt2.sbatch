#!/bin/bash
#SBATCH --job-name=q1-gpt2-eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/q1_gpt2_eval_%j.out
#SBATCH --error=logs/q1_gpt2_eval_%j.err

# Q1 intrinsic eval for the GPT-2 family: held-out, Q&A and OOD perplexity,
# base vs Q1 checkpoint, for the three sizes. One L4 is enough (<=774M).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/aluno_matheus/Code/llm-finetuning}"
export TOKENIZERS_PARALLELISM=false

.venv/bin/python scripts/eval_q1_gpt2.py --max-length 512 --out results/q1_gpt2.csv

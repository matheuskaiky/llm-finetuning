#!/bin/bash
#SBATCH --job-name=q5-examples
#SBATCH --partition=gpu
#SBATCH --nodelist=gpunode02
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=28G
#SBATCH --time=1:00:00
#SBATCH --output=logs/q5_examples_%j.out
#SBATCH --error=logs/q5_examples_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/aluno_matheus/Code/llm-finetuning}"
export TOKENIZERS_PARALLELISM=false
.venv/bin/python scripts/eval_rag_examples.py --limit 5 --out results/q5_qualitativos.md
echo "Q5_EXAMPLES_DONE"

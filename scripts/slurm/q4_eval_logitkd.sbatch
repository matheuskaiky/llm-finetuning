#!/bin/bash
#SBATCH --job-name=q4-eval-logitkd
#SBATCH --partition=gpu
#SBATCH --nodelist=gpunode02
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=28G
#SBATCH --time=2:00:00
#SBATCH --output=logs/q4_eval_logitkd_%j.out
#SBATCH --error=logs/q4_eval_logitkd_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/aluno_matheus/Code/llm-finetuning}"
export TOKENIZERS_PARALLELISM=false
.venv/bin/python scripts/eval_sft.py \
  --heldout data/processed/distill/diarios_distill_recall.jsonl \
  --judge-model models/Qwen3-8B --judge-device cuda:0 --device cuda \
  --models logitkd=outputs/distill_logitkd_qwen3_0p6b base=models/Qwen3-0.6B-Base \
  --out results/q4_logitkd_recall.csv
echo "Q4_LOGITKD_EVAL_DONE"

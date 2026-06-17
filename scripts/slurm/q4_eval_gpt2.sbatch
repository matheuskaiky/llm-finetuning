#!/bin/bash
#SBATCH --job-name=q4-eval-gpt2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=28G
#SBATCH --time=03:00:00
#SBATCH --output=logs/q4_eval_gpt2_%j.out
#SBATCH --error=logs/q4_eval_gpt2_%j.err

# Q4 response-based distillation eval for the gpt2 student: base gpt2 vs the
# distilled gpt2, judged by the fixed Qwen3-8B on the distill recall benchmark.
# Teacher reference (Qwen3-8B) judge is 0.66 from the main Q4 run.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/aluno_matheus/Code/llm-finetuning}"
export TOKENIZERS_PARALLELISM=false

.venv/bin/python scripts/eval_sft.py \
  --heldout data/processed/distill/diarios_distill_recall.jsonl \
  --judge-model models/Qwen3-8B --judge-device cuda:0 --device cuda \
  --models \
    base_gpt2=models/gpt2 \
    distill_gpt2=outputs/distill_gpt2 \
  --out results/q4_distill_gpt2_recall.csv \
  --details results/q4_distill_gpt2_recall_details.jsonl
echo "Q4_GPT2_EVAL_DONE"

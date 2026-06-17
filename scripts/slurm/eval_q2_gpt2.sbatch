#!/bin/bash
#SBATCH --job-name=q2-gpt2-eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=28G
#SBATCH --time=03:00:00
#SBATCH --output=logs/q2_gpt2_eval_%j.out
#SBATCH --error=logs/q2_gpt2_eval_%j.err

# Q2 SFT eval for the GPT-2 family: judge score (fixed Qwen3-8B judge) + response
# perplexity on the docentesDC recall held-out, for each size in three conditions:
# raw base (no SFT), SFT from base, SFT from the Q1 continued-pretrain checkpoint.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/aluno_matheus/Code/llm-finetuning}"
export TOKENIZERS_PARALLELISM=false

.venv/bin/python scripts/eval_sft.py \
  --judge-model models/Qwen3-8B \
  --out results/q2_sft_eval_gpt2_recall.csv \
  --details results/q2_sft_eval_gpt2_recall_details.jsonl \
  --models \
    "gpt2-124M-base-noft=models/gpt2" \
    "gpt2-124M-sft-base=outputs/sft_gpt2_base" \
    "gpt2-124M-sft-q1=outputs/sft_gpt2_q1" \
    "gpt2-medium-355M-base-noft=models/gpt2-medium" \
    "gpt2-medium-355M-sft-base=outputs/sft_gpt2_medium_base" \
    "gpt2-medium-355M-sft-q1=outputs/sft_gpt2_medium_q1" \
    "gpt2-large-774M-base-noft=models/gpt2-large" \
    "gpt2-large-774M-sft-base=outputs/sft_gpt2_large_base" \
    "gpt2-large-774M-sft-q1=outputs/sft_gpt2_large_q1"

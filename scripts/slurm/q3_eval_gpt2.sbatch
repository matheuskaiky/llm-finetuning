#!/bin/bash
#SBATCH --job-name=q3-eval-gpt2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=28G
#SBATCH --time=03:00:00
#SBATCH --output=logs/q3_eval_gpt2_%j.out
#SBATCH --error=logs/q3_eval_gpt2_%j.err

# Q3 LoRA eval for the GPT-2 family: the six merged LoRA checkpoints (base and
# Q1 start, three sizes) on the docentesDC recall held-out, fixed Qwen3-8B judge.
# The full-SFT comparison comes from the Q2 run (sft_gpt2_* in q2_sft).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/aluno_matheus/Code/llm-finetuning}"
export TOKENIZERS_PARALLELISM=false

.venv/bin/python scripts/eval_sft.py \
  --judge-model models/Qwen3-8B --judge-device cuda:0 --device cuda \
  --models \
    "gpt2-124M-lora-base=outputs/lora_gpt2_base" \
    "gpt2-124M-lora-q1=outputs/lora_gpt2_q1" \
    "gpt2-medium-355M-lora-base=outputs/lora_gpt2_medium_base" \
    "gpt2-medium-355M-lora-q1=outputs/lora_gpt2_medium_q1" \
    "gpt2-large-774M-lora-base=outputs/lora_gpt2_large_base" \
    "gpt2-large-774M-lora-q1=outputs/lora_gpt2_large_q1" \
  --out results/q3_lora_gpt2_recall.csv \
  --details results/q3_lora_gpt2_recall_details.jsonl
echo "Q3_GPT2_EVAL_DONE"

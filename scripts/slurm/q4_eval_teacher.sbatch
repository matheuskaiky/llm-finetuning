#!/bin/bash
#SBATCH --job-name=q4-eval-teacher
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=28G
#SBATCH --time=04:00:00
#SBATCH --output=logs/q4_eval_teacher_%j.out
#SBATCH --error=logs/q4_eval_teacher_%j.err

# Q4 teacher comparison eval: every student distilled from teacher <tag>, on the
# fixed recall benchmark, fixed Qwen3-8B judge. Arg $1 = teacher tag (e.g. t30b).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/aluno_matheus/Code/llm-finetuning}"
export TOKENIZERS_PARALLELISM=false
TAG="${1:?teacher tag required}"

.venv/bin/python scripts/eval_sft.py \
  --heldout data/processed/distill/diarios_distill_recall.jsonl \
  --judge-model models/Qwen3-8B --judge-device cuda:0 --device cuda \
  --models \
    "${TAG}_smollm135=outputs/distill_${TAG}_smollm135" \
    "${TAG}_smollm360=outputs/distill_${TAG}_smollm360" \
    "${TAG}_qwen2p5=outputs/distill_${TAG}_qwen2p5" \
    "${TAG}_qwen3_0p6=outputs/distill_${TAG}_qwen3_0p6" \
    "${TAG}_gemma1b=outputs/distill_${TAG}_gemma1b" \
    "${TAG}_gpt2=outputs/distill_${TAG}_gpt2" \
    "${TAG}_gemma270m=outputs/distill_${TAG}_gemma270m" \
  --out "results/q4_teacher_${TAG}_recall.csv" \
  --details "results/q4_teacher_${TAG}_recall_details.jsonl"
echo "Q4_EVAL_${TAG}_DONE"

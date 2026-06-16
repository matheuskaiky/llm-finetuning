#!/bin/bash
#SBATCH --job-name=q4-eval-distill
#SBATCH --partition=gpu
#SBATCH --nodelist=gpunode02
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=28G
#SBATCH --time=6:00:00
#SBATCH --output=logs/q4_eval_%j.out
#SBATCH --error=logs/q4_eval_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/aluno_matheus/Code/llm-finetuning}"
export TOKENIZERS_PARALLELISM=false
.venv/bin/python scripts/eval_sft.py \
  --heldout data/processed/distill/diarios_distill_recall.jsonl \
  --judge-model models/Qwen3-8B --judge-device cuda:0 --device cuda \
  --models \
    teacher_qwen3_8b=models/Qwen3-8B \
    base_smollm135=models/SmolLM2-135M distill_smollm135=outputs/distill_smollm2_135m \
    base_smollm360=models/SmolLM2-360M distill_smollm360=outputs/distill_smollm2_360m \
    base_qwen2p5=models/Qwen2.5-0.5B distill_qwen2p5=outputs/distill_qwen2p5_0p5b \
    base_qwen3_0p6=models/Qwen3-0.6B-Base distill_qwen3_0p6=outputs/distill_qwen3_0p6b \
    base_gemma=models/gemma-3-1b-pt distill_gemma=outputs/distill_gemma3_1b \
  --out results/q4_distill_recall.csv --details results/q4_distill_recall_details.jsonl
echo "Q4_EVAL_DONE"

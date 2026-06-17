#!/bin/bash
#SBATCH --job-name=q5-31b-retry
#SBATCH --partition=gpu
#SBATCH --nodelist=gpunode01
#SBATCH --gres=gpu:l4:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --output=logs/q5_31b_retry_%j.out
#SBATCH --error=logs/q5_31b_retry_%j.err

# gemma-4-31b-it RAG engine retry: 4-bit pinned to GPU0, 256-token generations,
# baseline+standard only (agentic_graph context does not fit alongside the 8B
# judge on one L4). Fixed Qwen3-8B judge on cuda:1.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/aluno_matheus/Code/llm-finetuning}"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1

.venv/bin/python scripts/eval_rag.py \
  --config configs/rag_diarios_gemma31b.yaml \
  --benchmark benchmarks/rag/diarios_rag_30.jsonl \
  --modes standard \
  --judge-model models/Qwen3-8B --judge-device cuda:1 \
  --out results/q5_engine_gemma-4-31b-it.csv
echo "Q5_31B_RETRY_DONE"

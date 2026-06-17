#!/bin/bash
#SBATCH --job-name=q5-big-engines
#SBATCH --partition=gpu
#SBATCH --nodelist=gpunode01
#SBATCH --gres=gpu:l4:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --output=logs/q5_big_engines_%j.out
#SBATCH --error=logs/q5_big_engines_%j.err

# Q5 engine ablation: gemma-3-27b-it and gemma-4-31b-it as RAG engines (4-bit NF4,
# fit on GPU0 via device_map auto), fixed Qwen3-8B judge on cuda:1. Same index and
# 30-question benchmark as the other engines.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/aluno_matheus/Code/llm-finetuning}"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1
BENCH=benchmarks/rag/diarios_rag_30.jsonl

run() {  # config, out-name
  echo "===== ENGINE $2 ====="
  .venv/bin/python scripts/eval_rag.py \
    --config "$1" --benchmark "$BENCH" \
    --modes standard,agentic_graph \
    --judge-model models/Qwen3-8B --judge-device cuda:1 \
    --out "results/q5_engine_$2.csv" || echo "ENGINE $2 FAILED"
}

run configs/rag_diarios_gemma27b.yaml gemma-3-27b-it
run configs/rag_diarios_gemma31b.yaml gemma-4-31b-it
echo "Q5_BIG_ENGINES_DONE"

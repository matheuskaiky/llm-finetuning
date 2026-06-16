#!/bin/bash
#SBATCH --job-name=q5-rag-30b
#SBATCH --partition=gpu
#SBATCH --nodelist=gpunode01
#SBATCH --gres=gpu:l4:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=58G
#SBATCH --time=12:00:00
#SBATCH --output=logs/q5_rag_30b_%j.out
#SBATCH --error=logs/q5_rag_30b_%j.err
# Q5 RAG with the 30B-A3B-FP8 engine (2x L4 via device_map=auto). The 30B fills both
# GPUs, so the judge is the engine itself (self-judge); the fixed-8B-judge results
# stay the cross-engine reference. Waits for node01 via afterany:402.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/aluno_matheus/Code/llm-finetuning}"
export TOKENIZERS_PARALLELISM=false
.venv/bin/python scripts/eval_rag.py --config configs/rag_diarios_qwen3_30b.yaml \
  --benchmark benchmarks/rag/diarios_rag_30.jsonl \
  --modes standard,agentic_graph --out results/q5_rag_30b.csv
echo "Q5_RAG_30B_DONE"

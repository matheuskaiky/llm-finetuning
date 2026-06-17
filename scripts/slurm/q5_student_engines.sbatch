#!/bin/bash
#SBATCH --job-name=q5-student-engines
#SBATCH --partition=gpu
#SBATCH --nodelist=gpunode01
#SBATCH --gres=gpu:l4:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=10:00:00
#SBATCH --output=logs/q5_student_engines_%j.out
#SBATCH --error=logs/q5_student_engines_%j.err

# Q5: use the Q4 distilled students as the RAG generation engine, scored by the
# fixed Qwen3-8B judge (cuda:1) for comparability with the main engine table.
# Tests whether a cheap distilled student can serve retrieval-augmented generation.
# One eval_rag run per student (reuses one engine config via --llm-model); the
# vector index/graph are the bounded diarios subset.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/aluno_matheus/Code/llm-finetuning}"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1

CFG=configs/rag_diarios_gemma3_1b_pt.yaml   # base-model engine config, same index
BENCH=benchmarks/rag/diarios_rag_30.jsonl

run() {  # label, model_path
  echo "===== ENGINE $1 ($2) ====="
  .venv/bin/python scripts/eval_rag.py \
    --config "$CFG" --benchmark "$BENCH" \
    --llm-model "$2" \
    --modes standard,agentic_graph \
    --judge-model models/Qwen3-8B --judge-device cuda:1 \
    --out "results/q5_student_$1.csv" || echo "ENGINE $1 FAILED"
}

run qwen3-0.6b-distill outputs/distill_qwen3_0p6b
run qwen2.5-0.5b-distill outputs/distill_qwen2p5_0p5b
run gemma-1b-distill     outputs/distill_gemma3_1b
run smollm2-360m-distill outputs/distill_smollm2_360m
run smollm2-135m-distill outputs/distill_smollm2_135m
run gpt2-distill         outputs/distill_gpt2
echo "Q5_STUDENT_ENGINES_DONE"

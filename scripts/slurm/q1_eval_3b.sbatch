#!/bin/bash
#SBATCH --job-name=q1-eval-3b
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=28G
#SBATCH --time=12:00:00
#SBATCH --output=logs/q1_eval_3b_%x_%j.out
#SBATCH --error=logs/q1_eval_3b_%x_%j.err

# Standalone Q1 eval for a 3B FSDP run whose in-job eval did not run: the FSDP
# full-state teardown aborts with SIGABRT after training, and `set -e` skips the
# eval. The per-epoch checkpoint is intact, so evaluate it here on one GPU. Cloze +
# P&R (5 runs, per-id) + held-out text perplexity.
#   $1 = tag (e.g. qwen2p5_3b or qwen2p5_3b_it)
#   $2 = base model     $3 = instruct model     $4 = epoch-1 checkpoint dir

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/aluno_matheus/Code/llm-finetuning}"
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false

TAG="${1:?tag}"; BASE="${2:?base}"; INST="${3:?instruct}"; EP1="${4:?ep1 ckpt}"
MODELS=(base="${BASE}" ep1="${EP1}" instruct="${INST}")

for pair in cloze:diarios_cloze pr:diarios_qa; do
  name="${pair%%:*}"; file="${pair##*:}"
  .venv/bin/python scripts/eval_q1_amao.py \
    --benchmark "benchmarks/pre_treino/${file}.jsonl" \
    --models "${MODELS[@]}" --device cuda --runs 5 \
    --out "results/q1_amao_${name}_${TAG}.csv"
done
.venv/bin/python scripts/eval_q1_heldout.py \
  --heldout data/processed/diarios_heldout_full.jsonl \
  --models "${MODELS[@]}" --device cuda \
  --out "results/q1_heldout_${TAG}.csv"
echo "Q1_EVAL_3B_DONE $(date -Is)"

#!/bin/bash
#SBATCH --job-name=q4-distill-teacher
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=28G
#SBATCH --time=06:00:00
#SBATCH --output=logs/q4_distill_teacher_%j.out
#SBATCH --error=logs/q4_distill_teacher_%j.err

# Q4 teacher comparison: response-based distillation of every student on one
# teacher's fixed-budget train set. Args:
#   $1 = teacher tag (e.g. t30b)     -> outputs/distill_<tag>_<student>
#   $2 = train data path (e.g. data/processed/distill/diarios_distill_t30b_train.jsonl)
# Same student ladder and config for every teacher, so only the training signal
# (the teacher) varies. Eval is the fixed recall benchmark + Qwen3-8B judge.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/aluno_matheus/Code/llm-finetuning}"
export TOKENIZERS_PARALLELISM=false

TAG="${1:?teacher tag required}"
DATA="${2:?train data path required}"
CFG=configs/distill_diarios_response.yaml

run() {  # student short name, model path
  echo "===== distill ${TAG} / $1 ($2) ====="
  .venv/bin/python scripts/train.py --config "$CFG" \
    --model-name "$2" --output-dir "outputs/distill_${TAG}_$1" --data-path "$DATA"
}

run smollm135 models/SmolLM2-135M
run smollm360 models/SmolLM2-360M
run qwen2p5   models/Qwen2.5-0.5B
run qwen3_0p6 models/Qwen3-0.6B-Base
run gemma1b   models/gemma-3-1b-pt
run gpt2      models/gpt2
run gemma270m models/gemma-3-270m
echo "Q4_DISTILL_${TAG}_DONE"

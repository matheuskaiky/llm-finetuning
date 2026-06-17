#!/bin/bash
#SBATCH --job-name=train-1gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=28G
#SBATCH --time=08:00:00
#SBATCH --output=logs/train_1gpu_%j.out
#SBATCH --error=logs/train_1gpu_%j.err

# Generic single-GPU full-parameter training for small models (continued pre-train
# or SFT) that fit one L4 with plain AdamW. Positional args:
#   $1 = config path (required)
#   $2 = model_name override (optional, e.g. a Q1 checkpoint for the Q1->Q2 A/B)
#   $3 = output_dir override (optional)
# Examples:
#   sbatch scripts/slurm/train_1gpu.sbatch configs/pretrain_diarios_gpt2.yaml
#   sbatch scripts/slurm/train_1gpu.sbatch configs/sft_docentes_gpt2.yaml \
#       outputs/pretrain_gpt2_diarios outputs/sft_gpt2_q1

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/aluno_matheus/Code/llm-finetuning}"
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false

CONFIG="${1:?config path required}"
EXTRA=()
[ "${2:-}" != "" ] && EXTRA+=(--model-name "$2")
[ "${3:-}" != "" ] && EXTRA+=(--output-dir "$3")

.venv/bin/python scripts/train.py --config "${CONFIG}" "${EXTRA[@]}"

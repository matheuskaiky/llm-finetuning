#!/bin/bash
#SBATCH --job-name=q1-pt-full
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=28G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/q1_pt_%x_%j.out
#SBATCH --error=logs/q1_pt_%x_%j.err

# Generic single-GPU Q1 continued pre-training on the FULL diarios corpus, then
# per-epoch benchmark evaluation. Positional args:
#   $1 = config path
#   $2 = checkpoint/output dir (must match the config's output_dir)
#   $3 = base model (antes)
#   $4 = instruct reference
#   $5 = tag for the result CSVs
# Submit with -J <name> to label the job, e.g.:
#   sbatch -J q1-1p5b scripts/slurm/q1_pretrain_1gpu.sbatch \
#     configs/pretrain_diarios_qwen2p5_1p5b_full.yaml \
#     outputs/pretrain_qwen2p5_1p5b_diarios_full \
#     models/Qwen2.5-1.5B models/Qwen2.5-1.5B-Instruct qwen2p5_1p5b

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/aluno_matheus/Code/llm-finetuning}"
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false

CONFIG="${1:?config}"; CKPT="${2:?ckpt}"; BASE="${3:?base}"; INST="${4:?instruct}"; TAG="${5:?tag}"

echo "== TRAIN ${TAG} =="
.venv/bin/python scripts/train.py --config "${CONFIG}"

echo "== EVAL per-epoch (5 runs, per-id) =="
bash scripts/slurm/eval_q1_epochs.sh "${CKPT}" "${BASE}" "${INST}" "${TAG}"

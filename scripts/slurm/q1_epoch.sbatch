#!/bin/bash
#SBATCH --job-name=q1-epoch
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=28G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/q1_ep_%x_%j.out
#SBATCH --error=logs/q1_ep_%x_%j.err

# One epoch of continued pre-training starting from a previous checkpoint, so a
# multi-epoch run can be chained across separate <=48h jobs (the `gpu` partition
# cap). Each job = 1 epoch from $2 into $3, with its own short cosine schedule.
# Chain with --dependency=afterok:<prev job>. Timestamps and SLURM accounting are
# printed so training time per epoch is recorded.
#   $1 = config path (num_train_epochs: 1)
#   $2 = start model (previous epoch's output dir, or the base for epoch 1)
#   $3 = output dir for this epoch

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/aluno_matheus/Code/llm-finetuning}"
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false

CONFIG="${1:?config}"; START="${2:?start model}"; OUT="${3:?out dir}"

echo "EPOCH_START $(date -Is) job=${SLURM_JOB_ID:-?} config=${CONFIG} from=${START} -> ${OUT}"
T0=$(date +%s)
.venv/bin/python scripts/train.py --config "${CONFIG}" --model-name "${START}" --output-dir "${OUT}"
T1=$(date +%s)
echo "EPOCH_DONE  $(date -Is) job=${SLURM_JOB_ID:-?} elapsed_s=$((T1 - T0)) out=${OUT}"
sacct -j "${SLURM_JOB_ID:-0}" --format=JobID,JobName%18,Submit,Start,End,Elapsed,State 2>/dev/null | head -6 || true

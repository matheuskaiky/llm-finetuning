#!/bin/bash
#SBATCH --job-name=q2-sft-qwen3-4b
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:l4:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=58G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/q2_sft_qwen3_4b_%j.out
#SBATCH --error=logs/q2_sft_qwen3_4b_%j.err

# Q2 SFT of Qwen3-4B on the docentesDC pairs (2x L4, FSDP full_shard + CPU offload).
# Optional positional args select the starting checkpoint / output for the A/B run:
#   $1 = model_name override (e.g. outputs/pretrain_qwen3_4b_diarios for the Q1 arm)
#   $2 = output_dir override (e.g. outputs/sft_qwen3_4b_q1)
# No args -> SFT from the original base (config defaults).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/aluno_matheus/Code/llm-finetuning}"
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
# See pretrain_qwen3_4b.sbatch: torchao 8-bit optimizer must run eager (compile bug),
# expandable_segments cuts fragmentation on the 22 GB L4.
export TORCHDYNAMO_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MASTER_PORT=$(( 20000 + (RANDOM % 20000) ))

EXTRA=()
[ "${1:-}" != "" ] && EXTRA+=(--model-name "$1")
[ "${2:-}" != "" ] && EXTRA+=(--output-dir "$2")

srun .venv/bin/torchrun --nproc_per_node=2 --master_port="${MASTER_PORT}" \
  scripts/train.py --config configs/sft_docentes_qwen3_4b.yaml "${EXTRA[@]}"

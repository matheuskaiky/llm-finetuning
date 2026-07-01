#!/bin/bash
#SBATCH --job-name=q1-pt-qwen2p5-3b
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:l4:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=58G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/q1_pt_qwen2p5_3b_%j.out
#SBATCH --error=logs/q1_pt_qwen2p5_3b_%j.err

# Q1 continued pre-training of Qwen2.5-3B (base) on the FULL diarios corpus. 3B
# full-parameter does not fit one L4, so this requests both L4 with FSDP full-shard
# + CPU offload + activation checkpointing (same recipe that was memory-bound at
# 4B); this run probes whether 3B fits. After training, evaluates antes/depois/
# instruct on the Cloze and P&R benchmarks (5 runs, per-id) on a single GPU.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/aluno_matheus/Code/llm-finetuning}"
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
export TORCHDYNAMO_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MASTER_PORT=$(( 20000 + (RANDOM % 20000) ))

CKPT=outputs/pretrain_qwen2p5_3b_diarios_full
BASE=models/Qwen2.5-3B
INST=models/Qwen2.5-3B-Instruct

echo "== TRAIN (FSDP 2x L4) =="
srun .venv/bin/torchrun --nproc_per_node=2 --master_port="${MASTER_PORT}" \
  scripts/train.py --config configs/pretrain_diarios_qwen2p5_3b_full.yaml

echo "== EVAL per-epoch (5 runs, per-id, single GPU) =="
CUDA_VISIBLE_DEVICES=0 bash scripts/slurm/eval_q1_epochs.sh "${CKPT}" "${BASE}" "${INST}" qwen2p5_3b

#!/bin/bash
#SBATCH --job-name=q1-pretrain-qwen3-4b
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:l4:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=58G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/q1_pretrain_qwen3_4b_%j.out
#SBATCH --error=logs/q1_pretrain_qwen3_4b_%j.err

# Q1 continued pre-training of Qwen3-4B-Base (full-parameter) on the diarios
# corpus. Requests both L4 on one node; FSDP full_shard + CPU offload fits 4B in
# 2x L4 (offload of params/grads/optimizer to host RAM). SLURM sets
# CUDA_VISIBLE_DEVICES to the allocated GPUs; torchrun spawns one rank per GPU.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/aluno_matheus/Code/llm-finetuning}"

export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
# torchao 8-bit optimizer (adamw_torch_8bit) breaks under torch.compile tracing on
# this stack (OptimState8bit aten.view.dtype); run it eager. expandable_segments
# reduces allocator fragmentation on the 22 GB L4 (4B full FT is memory-tight).
export TORCHDYNAMO_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MASTER_PORT=$(( 20000 + (RANDOM % 20000) ))

srun .venv/bin/torchrun \
  --nproc_per_node=2 \
  --master_port="${MASTER_PORT}" \
  scripts/train.py --config configs/pretrain_diarios_qwen3_4b.yaml

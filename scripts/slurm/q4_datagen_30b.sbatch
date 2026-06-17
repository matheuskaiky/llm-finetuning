#!/bin/bash
#SBATCH --job-name=q4-datagen-30b
#SBATCH --partition=gpu
#SBATCH --nodelist=gpunode01
#SBATCH --gres=gpu:l4:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=10:00:00
#SBATCH --output=logs/q4_datagen_30b_%j.out
#SBATCH --error=logs/q4_datagen_30b_%j.err

# Q4 teacher-comparison datagen with Qwen3-30B-A3B-FP8 (model-parallel over 2 L4 via
# device_map; FP8 native, no NCCL). Fixed 400-pair budget; slow (~tens of s/gen).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/aluno_matheus/Code/llm-finetuning}"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1

.venv/bin/python scripts/build_sft_pairs.py \
  --src-txt-dir data/processed/diarios_txt \
  --out-dir data/processed/distill \
  --n-train 400 --n-heldout 2 --pairs-per-text 3 --max-new-tokens 420 \
  --model models/Qwen3-30B-A3B-Instruct-2507-FP8 \
  --out-prefix diarios_distill_t30b --device-map auto
echo "Q4_DATAGEN_30B_DONE"

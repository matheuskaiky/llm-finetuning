#!/bin/bash
#SBATCH --job-name=q2-sft-anchored
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/q2_sft_anchored_%j.out
#SBATCH --error=logs/q2_sft_anchored_%j.err

# Q2 (v2): source-anchored deep-question SFT pairs, 1 pair/document, engine
# gemma-4-31b-it in 4-bit (fits one L4, same as the RAG 31b engine config).
# Archives the current docentes_sft_{train,heldout,recall}.jsonl into
# data/processed/sft/old/ before writing the new train/heldout + *_sources
# files (see scripts/build_sft_pairs_anchored.py).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/aluno_matheus/Code/llm-finetuning}"
export TOKENIZERS_PARALLELISM=false

.venv/bin/python scripts/build_sft_pairs_anchored.py \
  --n-train 1000 --n-heldout 150 \
  --model models/gemma-4-31b-it --load-in-4bit --device-map auto
echo "Q2_SFT_ANCHORED_DONE"

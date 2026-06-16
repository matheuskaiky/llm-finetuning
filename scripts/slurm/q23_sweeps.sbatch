#!/bin/bash
#SBATCH --job-name=q23-sweeps
#SBATCH --partition=gpu
#SBATCH --nodelist=gpunode02
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=28G
#SBATCH --time=6:00:00
#SBATCH --output=logs/q23_sweeps_%j.out
#SBATCH --error=logs/q23_sweeps_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/aluno_matheus/Code/llm-finetuning}"
export TOKENIZERS_PARALLELISM=false
PY=.venv/bin/python
# Q3 rank sweep
$PY scripts/q23_sweeps.py rank
$PY scripts/eval_sft.py --heldout data/processed/sft/docentes_sft_recall.jsonl \
  --judge-model models/Qwen3-8B --judge-device cuda:0 --device cuda \
  --models base=models/Qwen3-0.6B-Base r4=outputs/q3_rank_r4 r8=outputs/q3_rank_r8 \
    r16=outputs/q3_rank_r16 r32=outputs/q3_rank_r32 r64=outputs/q3_rank_r64 \
  --out results/q3_rank_sweep.csv
# Q2 data curve
$PY scripts/q23_sweeps.py data
$PY scripts/eval_sft.py --heldout data/processed/sft/docentes_sft_recall.jsonl \
  --judge-model models/Qwen3-8B --judge-device cuda:0 --device cuda \
  --models base=models/Qwen3-0.6B-Base n250=outputs/q2_data_n250 \
    n500=outputs/q2_data_n500 n1000=outputs/q2_data_n1000 \
  --out results/q2_data_curve.csv
echo "Q23_SWEEPS_DONE"

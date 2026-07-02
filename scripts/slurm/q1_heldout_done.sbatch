#!/bin/bash
#SBATCH --job-name=q1-heldout-done
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=28G
#SBATCH --time=10:00:00
#SBATCH --output=logs/q1_heldout_done_%j.out
#SBATCH --error=logs/q1_heldout_done_%j.err

# Held-out text perplexity for the Q1 models whose training finished before the
# held-out eval was added (0.6B 3-epoch, 1.5B/1.7B 1-epoch). One CSV per model with
# the same epoch labels as the Cloze/P&R evals. Low priority: fills gaps, must not
# delay the training queue.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/aluno_matheus/Code/llm-finetuning}"
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
HLD=data/processed/diarios_heldout_full.jsonl
E=.venv/bin/python

D06=outputs/pretrain_qwen3_0p6b_diarios_full
$E scripts/eval_q1_heldout.py --heldout "$HLD" --device cuda \
  --models base=models/Qwen3-0.6B-Base ep1=$D06/checkpoint-16027 \
           ep2=$D06/checkpoint-32054 ep3=$D06/checkpoint-48081 \
           instruct=models/Qwen3-0.6B \
  --out results/q1_heldout_qwen3_0p6b.csv

$E scripts/eval_q1_heldout.py --heldout "$HLD" --device cuda \
  --models base=models/Qwen2.5-1.5B \
           ep1=outputs/pretrain_qwen2p5_1p5b_diarios_full/checkpoint-16027 \
           instruct=models/Qwen2.5-1.5B-Instruct \
  --out results/q1_heldout_qwen2p5_1p5b.csv

$E scripts/eval_q1_heldout.py --heldout "$HLD" --device cuda \
  --models base=models/Qwen3-1.7B-Base \
           ep1=outputs/pretrain_qwen3_1p7b_diarios_full/checkpoint-16027 \
           instruct=models/Qwen3-1.7B \
  --out results/q1_heldout_qwen3_1p7b.csv
echo "Q1_HELDOUT_DONE $(date -Is)"

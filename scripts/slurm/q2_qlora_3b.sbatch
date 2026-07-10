#!/bin/bash
#SBATCH --job-name=q2-qlora-3b
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=28G
#SBATCH --time=10:00:00
#SBATCH --output=logs/q2_qlora3b_%x_%j.out
#SBATCH --error=logs/q2_qlora3b_%x_%j.err

# Q2 "base up to 3B": QLoRA SFT of Qwen2.5-3B on one L4 (NF4 4-bit base). Trains the
# vanilla base and the Q1 continued-pretrain checkpoint (the base-vs-Q1 A/B) on the
# new source-anchored pairs, then evaluates both plus the vanilla base/instruct
# references in the 5-run/per-id format. Chain after the pairs job:
#   sbatch --dependency=afterok:<606> scripts/slurm/q2_qlora_3b.sbatch
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/aluno_matheus/Code/llm-finetuning}"
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
E=.venv/bin/python

CFG=configs/qlora_docentes_qwen2p5_3b.yaml
BASE=models/Qwen2.5-3B
Q1=outputs/pretrain_qwen2p5_3b_diarios_full/checkpoint-16027
INST=models/Qwen2.5-3B-Instruct
OUT_BASE=outputs/qlora_qwen2p5_3b_base
OUT_Q1=outputs/qlora_qwen2p5_3b_q1

echo "== QLoRA-3B base $(date -Is) =="
$E scripts/train.py --config "$CFG" --model-name "$BASE" --output-dir "$OUT_BASE" \
  || echo "TRAIN_FAILED $OUT_BASE"
echo "== QLoRA-3B q1 $(date -Is) =="
$E scripts/train.py --config "$CFG" --model-name "$Q1" --output-dir "$OUT_Q1" \
  || echo "TRAIN_FAILED $OUT_Q1"

for set in heldout recall; do
  echo "== EVAL ${set} qwen2p5_3b $(date -Is) =="
  $E scripts/eval_sft.py \
    --heldout "data/processed/sft/docentes_sft_${set}.jsonl" \
    --models base="$BASE" instruct="$INST" \
             qlora_base="$OUT_BASE" qlora_q1="$OUT_Q1" \
    --judge-model models/Qwen3-8B --judge-load-in-4bit \
    --device cuda --runs 5 \
    --out "results/q2q3_qwen2p5_3b_${set}.csv" || echo "EVAL_FAILED 3b ${set}"
done
echo "Q2_QLORA_3B_DONE $(date -Is)"

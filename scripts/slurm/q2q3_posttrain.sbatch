#!/bin/bash
#SBATCH --job-name=q2q3-posttrain
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=28G
#SBATCH --time=12:00:00
#SBATCH --output=logs/q2q3_%x_%j.out
#SBATCH --error=logs/q2q3_%x_%j.err

# Q2 (SFT) + Q3 (LoRA) refresh on the FULL-dataset Q1 checkpoints and on the new
# source-anchored pairs (job 606). For one model it trains four variants and then
# evaluates them in the new 5-run/per-id format:
#   sft_base   : SFT of the vanilla base
#   sft_q1     : SFT starting from the Q1 continued-pretrain checkpoint
#   lora_base  : LoRA of the vanilla base
#   lora_q1    : LoRA starting from the Q1 checkpoint
# The base-vs-q1 pair is the A/B: does domain continued-pretraining help downstream
# SFT/LoRA. Reuses the docentes SFT/LoRA configs via --model-name/--output-dir
# overrides (no per-checkpoint config duplication). Chain after the pairs job with
# --dependency=afterok:<606>. Positional args:
#   $1 = tag (e.g. qwen3_0p6b)   $2 = vanilla base   $3 = Q1 checkpoint dir
#   $4 = instruct reference (eval-only baseline)
#   $5 = sft config              $6 = lora config
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/aluno_matheus/Code/llm-finetuning}"
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
E=.venv/bin/python

TAG="${1:?tag}"; BASE="${2:?base}"; Q1="${3:?q1 ckpt}"; INST="${4:?instruct}"
SFT_CFG="${5:?sft config}"; LORA_CFG="${6:?lora config}"

declare -A OUT
OUT[sft_base]="outputs/sft_${TAG}_base_v2"
OUT[sft_q1]="outputs/sft_${TAG}_q1_v2"
OUT[lora_base]="outputs/lora_${TAG}_base_v2"
OUT[lora_q1]="outputs/lora_${TAG}_q1_v2"

train() { # cfg model_name out_dir
  echo "== TRAIN $3 from $2 $(date -Is) =="
  $E scripts/train.py --config "$1" --model-name "$2" --output-dir "$3" \
    || echo "TRAIN_FAILED $3"
}
train "${SFT_CFG}"  "${BASE}" "${OUT[sft_base]}"
train "${SFT_CFG}"  "${Q1}"   "${OUT[sft_q1]}"
train "${LORA_CFG}" "${BASE}" "${OUT[lora_base]}"
train "${LORA_CFG}" "${Q1}"   "${OUT[lora_q1]}"

for set in heldout recall; do
  echo "== EVAL ${set} ${TAG} $(date -Is) =="
  $E scripts/eval_sft.py \
    --heldout "data/processed/sft/docentes_sft_${set}.jsonl" \
    --models base="${BASE}" instruct="${INST}" \
             sft_base="${OUT[sft_base]}" sft_q1="${OUT[sft_q1]}" \
             lora_base="${OUT[lora_base]}" lora_q1="${OUT[lora_q1]}" \
    --judge-model models/Qwen3-8B --judge-load-in-4bit \
    --device cuda --runs 5 \
    --out "results/q2q3_${TAG}_${set}.csv" || echo "EVAL_FAILED ${TAG} ${set}"
done
echo "Q2Q3_POSTTRAIN_DONE ${TAG} $(date -Is)"

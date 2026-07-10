#!/bin/bash
#SBATCH --job-name=q4-abl-datagen
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:l4:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=58G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/q4_abl_datagen_%j.out
#SBATCH --error=logs/q4_abl_datagen_%j.err

# Q4 pairs ablation, data generation. Teacher = gemma-4-31b-it (4-bit NF4, split
# over both L4 via device_map=auto). Source = the FULL diarios corpus (not the 2k
# probe), for more diverse distilled knowledge. Generates the LARGEST train set
# once plus fixed disjoint heldout and in-domain recall, then makes nested subsets
# (1200/3000) so the only variable in the ablation is the number of training pairs
# and the eval sets stay identical.
# Not chained to run automatically: submit when the GPUs free up.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/aluno_matheus/Code/llm-finetuning}"
export TOKENIZERS_PARALLELISM=false

PY=.venv/bin/python
DIR=data/processed/distill
PREFIX=diarios_distill_g31b
TEACHER=models/gemma-4-31b-it
COMMON=(--src-txt-dir data/processed/diarios_txt_full --out-dir "${DIR}" \
        --out-prefix "${PREFIX}" --model "${TEACHER}" --load-in-4bit --device-map auto \
        --pairs-per-text 3 --max-new-tokens 420 --seed 42)

echo "== DATAGEN train(6000)+heldout(100) $(date -Is) =="
$PY scripts/build_sft_pairs.py "${COMMON[@]}" --n-train 6000 --n-heldout 100

echo "== DATAGEN recall(100) $(date -Is) =="
$PY scripts/build_sft_pairs.py "${COMMON[@]}" --recall --n-recall 100

echo "== subsets aninhados (1200/3000/6000) =="
cp "${DIR}/${PREFIX}_train.jsonl" "${DIR}/${PREFIX}_train_6000.jsonl"
head -n 3000 "${DIR}/${PREFIX}_train.jsonl" > "${DIR}/${PREFIX}_train_3000.jsonl"
head -n 1200 "${DIR}/${PREFIX}_train.jsonl" > "${DIR}/${PREFIX}_train_1200.jsonl"
wc -l "${DIR}/${PREFIX}_train_"{1200,3000,6000}.jsonl "${DIR}/${PREFIX}_heldout.jsonl" "${DIR}/${PREFIX}_recall.jsonl"
echo "Q4_ABL_DATAGEN_DONE $(date -Is)"

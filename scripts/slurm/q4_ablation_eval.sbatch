#!/bin/bash
#SBATCH --job-name=q4-abl-eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:l4:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=58G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/q4_abl_eval_%j.out
#SBATCH --error=logs/q4_abl_eval_%j.err

# Q4 pairs ablation, evaluation. Judge = Llama-3.2-11B-Vision-Instruct, text-only
# (AutoModelForCausalLM resolves "mllama" to MllamaForCausalLM, the text-decoder-only
# class - transformers ignores the checkpoint's vision tower, no image ever passed).
# Neither Qwen nor Gemma: neutral to both the teacher (gemma-4-31b-it, family Gemma)
# and the students (Qwen3-0.6B, family Qwen), unlike Qwen3-8B (same family as the
# students) or Qwen3-30B-A3B-FP8/gemma-4-31b-it (both also compared as teachers
# elsewhere in Q4). See NOTAS.md for the family-bias discussion and for why not the
# Qwen3.6-27B/35B-A3B-FP8 checkpoints (real VLMs with no CausalLM-only class).
# Scores the teacher, the base student and the students distilled at 1200/3000/6000
# pairs, on the in-domain recall set (and the disjoint heldout). 5 runs, per-id, one
# CSV per set (the #pairs is the model label: distill_1200/3000/6000).
# The tested candidates are loaded/unloaded one at a time (Pass 1); the judge only
# loads afterwards for scoring (Pass 2, eval_sft.py), so there is no simultaneous
# residency with any candidate. Judge weights are bf16, ~20 GB (text tower only
# actually used), fits one L4 without device_map/4bit.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/aluno_matheus/Code/llm-finetuning}"
export TOKENIZERS_PARALLELISM=false

TEACHER=models/gemma-4-31b-it
JUDGE=models/Llama-3.2-11B-Vision-Instruct
STUDENT_BASE=models/Qwen3-0.6B-Base   # keep in sync with the distill config's student
MODELS=(
  teacher_g31b="${TEACHER}"
  base="${STUDENT_BASE}"
  distill_1200=outputs/distill_g31b_1200
  distill_3000=outputs/distill_g31b_3000
  distill_6000=outputs/distill_g31b_6000
)

for set in recall heldout; do
  echo "== EVAL ${set} (judge Llama-3.2-11B, text-only) $(date -Is) =="
  .venv/bin/python scripts/eval_sft.py \
    --heldout "data/processed/distill/diarios_distill_g31b_${set}.jsonl" \
    --models "${MODELS[@]}" \
    --judge-model "${JUDGE}" --judge-device cuda:0 \
    --device cuda --runs 5 \
    --out "results/q4_ablation_${set}.csv"
done
echo "Q4_ABL_EVAL_DONE $(date -Is)"

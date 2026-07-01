#!/bin/bash
# One monitoring snapshot of the Q1 full-corpus jobs: queue, per-job step progress,
# finished-epoch timings and any real errors. Used by the 4-hourly monitor loop.
cd "${SLURM_SUBMIT_DIR:-/home/aluno_matheus/Code/llm-finetuning}"
echo "=== SNAPSHOT $(TZ='America/Sao_Paulo' date '+%Y-%m-%d %H:%M') (Brasilia) ==="
echo "--- queue ---"
squeue -u aluno_matheus -o "%.7i %.18j %.3t %.9M %R" 2>/dev/null

echo "--- progresso (ultima linha de step por log ativo) ---"
for f in logs/q1_pt_q1-*.err logs/q1_ep_*.err; do
  [ -f "$f" ] || continue
  last=$(tail -40 "$f" 2>/dev/null | tr '\r' '\n' | grep -oE "[0-9]+/[0-9]+ \[[0-9:]+<[0-9:]+" | tail -1)
  [ -n "$last" ] && echo "  $(basename "$f"): $last"
done

echo "--- epocas concluidas / tempos (EPOCH_DONE, elapsed, train_runtime) ---"
grep -h -E "EPOCH_DONE|elapsed_s=|train_runtime_s|TRAIN .* DONE" logs/q1_*.{out,err} 2>/dev/null | tail -20

echo "--- erros reais em logs ATIVOS (ultimas ~4h; ignora tokenizer e jobs antigos) ---"
find logs -name "q1_*.err" -mmin -250 2>/dev/null \
  | while read -r f; do
      grep -iE "out of memory|CUDA error|RuntimeError|ChildFailedError" "$f" 2>/dev/null \
        | grep -viE "indexing errors" | tail -2 | sed "s|^|  $(basename "$f"): |"
    done | tail -20

echo "--- resultados eval gerados ---"
ls -1 results/q1_amao_*.csv 2>/dev/null | tail -20
echo "=== END SNAPSHOT ==="

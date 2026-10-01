#!/bin/bash
# Durable driver for the head-FT eval. Phase 1: base + 15 main FT arms (headft_affcache) ->
# eval_headft_seeds.py -> table2.csv. Phase 2: 70 variant arms (headft_variants_scores) ->
# eval_variants.py -> variants_compare.csv. Idempotent + resume-friendly; survives session
# teardown and (via --requeue) preemption. Progress-aware: it never gives up while chunks are
# still being written, and it re-checks completeness every cycle instead of blocking on a full
# queue drain (lowprio preemption keeps jobs in-queue, so a drain-wait can hang indefinitely).
set -uo pipefail
TARGET="${1:-588689}"
ROOT=/global/scratch/users/sergiomar10/boltzaff
source /clusterfs/nilah/sergio/miniconda3/etc/profile.d/conda.sh; conda activate boltzft
source "$ROOT/env.sh" "$TARGET"
R="$WORKDIR/runs/$TARGET"; AFF="$R/headft_affcache"
NC=$(ls "$R"/ft_inputs_full/eval_chunks/chunk_*.txt | wc -l)
log(){ echo "[await $(date '+%F %T')] $*"; }
q_has(){ squeue -u "$USER" -h -n "$1" -o "%i" 2>/dev/null | grep -q .; }
nchunks(){ ls "$1"/chunk_*.csv 2>/dev/null | wc -l; }

NS=(40 40 40 40 40 100 100 100 100 100 300 300 300 300 300)
SEEDS=(0 1 2 3 4 0 1 2 3 4 0 1 2 3 4)
STALL_CAP=8   # give up only after this many cycles with NO new chunk written and queue drained

# ---------------- Phase 1: main arms ----------------
prev_total=-1; stall=0
while :; do
  miss_base=0; [ "$(nchunks "$AFF/base/scores")" -eq "$NC" ] || miss_base=1
  miss_ft=(); total=$(nchunks "$AFF/base/scores")
  for i in $(seq 0 14); do
    d="$AFF/lightning_top${NS[$i]}_seed${SEEDS[$i]}/scores"; c=$(nchunks "$d")
    total=$((total+c)); [ "$c" -eq "$NC" ] || miss_ft+=($((i+1)))
  done
  log "MAIN progress: base=$(nchunks "$AFF/base/scores")/$NC  incomplete_ft=${#miss_ft[@]}/15  total_chunks=$total/$((16*NC))"

  if [ "$miss_base" -eq 0 ] && [ ${#miss_ft[@]} -eq 0 ]; then
    log "all 16 main arms complete -> running eval"
    python "$ROOT/pipeline_local/eval_headft_seeds.py" --targets "$TARGET" --out-dir "$WORKDIR/analysis/$TARGET" \
      && log "DONE main -> $WORKDIR/analysis/$TARGET/table2.csv" || log "MAIN EVAL ERRORED"
    break
  fi

  if q_has bft_base || q_has bft_ft; then stall=0; prev_total=$total; sleep 120; continue; fi

  # queue drained AND arms incomplete
  if [ "$total" -gt "$prev_total" ]; then stall=0; else stall=$((stall+1)); fi
  prev_total=$total
  if [ "$stall" -ge "$STALL_CAP" ]; then
    log "MAIN stalled $stall cycles with no progress and empty queue -> running eval on complete arms"
    python "$ROOT/pipeline_local/eval_headft_seeds.py" --targets "$TARGET" --out-dir "$WORKDIR/analysis/$TARGET" \
      && log "partial main eval written" || log "MAIN EVAL ERRORED"
    break
  fi
  log "MAIN backfill (stall=$stall): base_missing=$miss_base ft_missing=${miss_ft[*]:-none}"
  [ "$miss_base" -eq 1 ] && sbatch --export=ALL,TARGET="$TARGET" "$ROOT/slurm/score_base.sbatch"
  if [ ${#miss_ft[@]} -gt 0 ]; then
    sbatch --export=ALL,TARGET="$TARGET" --array="$(IFS=,; echo "${miss_ft[*]}")"%12 "$ROOT/slurm/train_score_seed.sbatch"
  fi
  sleep 180
done

# ---------------- Phase 2: variant composition experiment ----------------
VS="$R/headft_variants_scores"; VNC=$(ls "$R"/ft_inputs_variants/eval_chunks/chunk_*.txt 2>/dev/null | wc -l)
mapfile -t CONDS < "$R/ft_inputs_variants/conds.txt"
prev_total=-1; stall=0
while :; do
  nmiss=0; total=0
  for cond in "${CONDS[@]}"; do for sd in 0 1 2 3 4; do
    c=$(nchunks "$VS/${cond}_seed${sd}"); total=$((total+c)); [ "$c" -eq "$VNC" ] || nmiss=$((nmiss+1))
  done; done
  log "VARIANT progress: incomplete=$nmiss/70  total_chunks=$total/$((70*VNC))"

  if [ "$nmiss" -eq 0 ]; then
    python "$ROOT/pipeline_local/eval_variants.py" --target "$TARGET" --out-dir "$WORKDIR/analysis/$TARGET" \
      && log "DONE variants -> $WORKDIR/analysis/$TARGET/variants_compare.csv" || log "VARIANT EVAL ERRORED"
    break
  fi
  if q_has bft_var; then stall=0; prev_total=$total; sleep 180; continue; fi
  if [ "$total" -gt "$prev_total" ]; then stall=0; else stall=$((stall+1)); fi
  prev_total=$total
  if [ "$stall" -ge "$STALL_CAP" ]; then
    log "VARIANT stalled $stall cycles -> running eval on complete arms ($nmiss incomplete)"
    python "$ROOT/pipeline_local/eval_variants.py" --target "$TARGET" --out-dir "$WORKDIR/analysis/$TARGET" \
      && log "partial variant eval written" || log "VARIANT EVAL ERRORED"
    break
  fi
  log "VARIANT backfill (stall=$stall): resubmitting full array (resume-friendly)"
  sbatch --export=ALL,TARGET="$TARGET" --array=1-70%14 "$ROOT/slurm/train_variants.sbatch"
  sleep 180
done
log "await_main_eval finished (main + variants)"

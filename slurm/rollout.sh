#!/bin/bash
# Master overnight orchestrator for the 7 non-588689 targets. Idempotent + self-resuming.
# For each target: (1) ensure CPU prep (inputs_full YAMLs + chunks) done; (2) keep a per-target
# pipeline driver (driver.sbatch -> pipeline_driver.sh: Pass1->consolidate->Pass2->base+15FT->eval)
# alive, relaunching it if it dies and the target isn't finished. Per-target job names give dedup
# so requeue/restart never double-submits. All lowprio; every GPU array has --requeue.
set -uo pipefail
ROOT=/global/scratch/users/sergiomar10/boltzaff
source /clusterfs/nilah/sergio/miniconda3/etc/profile.d/conda.sh
source "$ROOT/env.sh" 588689 >/dev/null 2>&1   # for BOLTZFT/WORKDIR/BOLTZ_CACHE
BOLTZFT_G="$BOLTZFT"; WORKDIR_G="$WORKDIR"
log(){ echo "[rollout $(date '+%F %T')] $*"; }
TARGETS="504329 1053173-743445 493248-485317 434954-2097 540297-493091 463203-2650 624273-588549"
jobname_for(){ echo "drv_${1:0:20}"; }   # per-target driver job name
q_has_name(){ squeue -u "$USER" -h -n "$1" -o "%i" 2>/dev/null | grep -q .; }

ensure_prep(){
  local T="$1"; source "$ROOT/env.sh" "$T"
  if [ -f "$R/chunks.tsv" ] && [ "$(ls "$R/inputs_full"/*.yaml 2>/dev/null | wc -l)" -gt 40000 ]; then return 0; fi
  log "[$T] prepping (conda boltzba)"
  ( conda activate boltzba
    python "$BOLTZFT/pipeline/select_subsets.py" --target "$T" --results-csv "$RESULTS_CSV" --out-dir "$R/selection" 2>&1 | tail -1
    python "$BOLTZFT/pipeline/build_boltz_inputs.py" --target "$T" \
      --inference-csv "$R/selection/inference_compounds.csv" --msa-csv "$ROOT/data/${T}_msa.csv" \
      --out-yaml-dir "$R/inputs_full" 2>&1 | tail -1
    python "$BOLTZFT/pipeline/make_chunks.py" --target "$T" \
      --inputs-dir "$R/inputs_full" --chunks-root "$R/inputs_chunks_full" \
      --outputs-root "$R/outputs_chunks_full" --index-file "$R/chunks.tsv" 2>&1 | tail -1 )
  log "[$T] prepped: yamls=$(ls "$R/inputs_full" 2>/dev/null | wc -l) chunks=$([ -f "$R/chunks.tsv" ] && wc -l < "$R/chunks.tsv" || echo 0)"
}

done_for(){ [ -f "$WORKDIR_G/analysis/$1/table2.csv" ]; }

cycle=0
while :; do
  cycle=$((cycle+1)); alldone=1
  for T in $TARGETS; do
    if done_for "$T"; then continue; fi
    alldone=0
    ensure_prep "$T"
    jn=$(jobname_for "$T")
    if q_has_name "$jn"; then continue; fi   # driver already queued/running for this target
    log "[$T] launching pipeline driver ($jn)"
    sbatch --job-name="$jn" --export=ALL,TARGET="$T" "$ROOT/slurm/driver.sbatch" >/dev/null \
      && log "[$T] driver submitted" || log "[$T] driver submit FAILED"
    sleep 5
  done
  [ "$alldone" -eq 1 ] && { log "ALL 7 targets have table2.csv -> rollout done"; break; }
  log "cycle $cycle complete; sleeping 1800s"
  sleep 1800
done

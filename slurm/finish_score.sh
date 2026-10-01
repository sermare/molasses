#!/bin/bash
# Durable finisher: complete the affinity cache, then re-run scoring (checkpoints already trained)
# for base + 15 main FT arms + 70 variant arms, then run the main eval. Idempotent + resume-safe.
set -uo pipefail
TARGET="${1:-588689}"
ROOT=/global/scratch/users/sergiomar10/boltzaff
source /clusterfs/nilah/sergio/miniconda3/etc/profile.d/conda.sh; conda activate boltzft
source "$ROOT/env.sh" "$TARGET"
log(){ echo "[fscore $(date '+%F %T')] $*"; }
q_has(){ squeue -u "$USER" -h -n "$1" -o "%i" 2>/dev/null | grep -q .; }

# --- 1. let current cache/training jobs drain ---
while q_has bft_p2 || q_has bft_ft || q_has bft_var || q_has bft_base; do sleep 180; done

# --- 2. complete the affinity cache: resweep any Pass-2 chunk with < 345 cached ---
incomplete_p2(){ python - <<'PY'
import glob,os
R=os.environ["R"]; tsv=[l.split('\t') for l in open(R+"/chunks.tsv").read().splitlines()]
out=[]
for idx,_,_,od in tsv:
    d=f"{R}/outputs_affcache/{os.path.basename(od)}"
    if len(glob.glob(f"{d}/boltz_results_*/predictions/*/affinity_embeddings_*.npz"))<345: out.append(idx)
print(",".join(out))
PY
}
while true; do
  M=$(incomplete_p2)
  [ -z "$M" ] && { log "affinity cache complete"; break; }
  log "cache resweep: $M"
  sbatch --export=ALL,TARGET="$TARGET" --time=06:00:00 --array="$M"%40 "$ROOT/slurm/pass2_affcache.sbatch"
  sleep 120; while q_has bft_p2; do sleep 180; done
done

# --- 3. re-run scoring (training skipped since checkpoints exist) ---
log "submitting scoring: base + 15 main FT + 70 variants"
sbatch --export=ALL,TARGET="$TARGET" "$ROOT/slurm/score_base.sbatch"
sbatch --export=ALL,TARGET="$TARGET" --array=1-15%12 "$ROOT/slurm/train_score_seed.sbatch"
sbatch --export=ALL,TARGET="$TARGET" --array=1-70%14 "$ROOT/slurm/train_variants.sbatch"
sleep 120
while q_has bft_base || q_has bft_ft || q_has bft_var; do sleep 180; done
log "scoring done"

# --- 4. main eval ---
python "$ROOT/pipeline_local/eval_headft_seeds.py" --targets "$TARGET" --out-dir "$WORKDIR/analysis/$TARGET" \
  && log "DONE main eval -> $WORKDIR/analysis/$TARGET/table2.csv" || log "main eval errored (check score completeness)"

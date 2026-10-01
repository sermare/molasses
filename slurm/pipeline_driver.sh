#!/bin/bash
# Self-resuming pipeline driver (runs as a CPU SLURM job on a non-GPU node).
# Idempotent: on (re)start it infers stage from disk state and continues, so it is
# safe under lowprio preemption+requeue. Orchestrates:
#   Pass-1 (monitor+resweep) -> consolidate -> Pass-2 (submit+monitor+resweep)
#   -> base scoring + multi-seed FT (submit+monitor) -> eval.
set -uo pipefail
TARGET="${1:?usage: pipeline_driver.sh TARGET}"
ROOT=/global/scratch/users/sergiomar10/boltzaff
source /clusterfs/nilah/sergio/miniconda3/etc/profile.d/conda.sh; conda activate boltzft
source "$ROOT/env.sh" "$TARGET"
TAG="${TARGET##*-}"   # target-scoped job-name suffix (assay id), so parallel drivers don't collide
NCHUNK=$(wc -l < "$R/chunks.tsv")
log(){ echo "[driver $(date '+%F %T')] $*"; }

folded_count(){ find -L "$R/outputs_chunks_full" -name 'embeddings_*.npz' 2>/dev/null | wc -l; }
affcache_count(){ find -L "$R/outputs_affcache" -name 'affinity_embeddings_*.npz' 2>/dev/null | wc -l; }
q_has(){ squeue -u "$USER" -h -n "$1" -o "%i" 2>/dev/null | grep -q .; }

# indices (1..NCHUNK) whose Pass-1 / Pass-2 output is incomplete and not currently queued
missing_idx(){  # $1 = subdir (outputs_chunks_full|outputs_affcache), $2 = glob, $3 = job-name
python - "$1" "$2" "$3" <<'PY'
import subprocess,re,sys,os
sub,glob,jobname=sys.argv[1],sys.argv[2],sys.argv[3]
R=os.environ["R"]; N=int(open(R+"/chunks.tsv").read().count("\n"))
def sh(c): return subprocess.run(c,shell=True,capture_output=True,text=True).stdout
live=set()
for l in sh(f"squeue -u $USER -h -o '%i %j'").splitlines():
    p=l.split()
    if len(p)<2 or p[1]!=jobname: continue
    m=re.search(r'_\[?([0-9,\-]+)',p[0])
    if not m: continue
    for part in m.group(1).split(','):
        if '-' in part: a,b=part.split('-'); live.update(range(int(a),int(b)+1))
        elif part.isdigit(): live.add(int(part))
done=set()
for i in range(1,N+1):
    d=f"{R}/{sub}/chunk_{i-1:03d}"
    n=len(sh(f"find -L {d} -name '{glob}' 2>/dev/null").split())
    if n>=345: done.add(i)
print(",".join(map(str,sorted(set(range(1,N+1))-live-done))))
PY
}

# ---------------- Stage 1: Pass-1 poses ----------------
if [ ! -f "$R/ft_inputs_full/eval_ids.txt" ]; then
  log "Stage 1: Pass-1 poses (folded=$(folded_count)/$NCHUNK*350)"
  while true; do
    if q_has "p1_$TAG"; then sleep 300; continue; fi
    M=$(missing_idx outputs_chunks_full 'embeddings_*.npz' "p1_$TAG")
    [ -z "$M" ] && break
    log "Pass-1 resweep missing: $M"
    sbatch --job-name="p1_$TAG" --export=ALL,TARGET="$TARGET" --array="$M"%40 --exclude=n0143.savio3,n0144.savio3 "$ROOT/slurm/pass1_embed.sbatch"
    sleep 300
  done
  log "Pass-1 complete (folded=$(folded_count)); consolidating"
  python "$BOLTZFT/pipeline/merge_consolidate.py" --outputs-root "$R/outputs_chunks_full" --inputs-dir "$R/inputs_full" --out-dir "$R/consolidated_full"
  python "$BOLTZFT/pipeline/build_ft_inputs_full.py" --target "$TARGET" --results-csv "$RESULTS_CSV" --consolidated-dir "$R/consolidated_full" --out-dir "$R/ft_inputs_full"
  log "consolidated: structures=$(find -L "$R/consolidated_full/structures" -name '*.npz'|wc -l) eval_ids=$(wc -l < "$R/ft_inputs_full/eval_ids.txt")"
else
  log "Stage 1 already done (ft_inputs_full present)"
fi

# ---------------- Stage 2: Pass-2 affinity cache ----------------
NEED=$(wc -l < "$R/ft_inputs_full/score_ids_all.txt")
if [ "$(affcache_count)" -lt $((NEED-100)) ]; then
  log "Stage 2: Pass-2 affinity cache (have=$(affcache_count)/$NEED)"
  if ! q_has "p2_$TAG"; then
    log "submitting Pass-2 array 1-$NCHUNK"
    sbatch --job-name="p2_$TAG" --export=ALL,TARGET="$TARGET" --array=1-"$NCHUNK"%40 "$ROOT/slurm/pass2_affcache.sbatch"
    sleep 60
  fi
  while true; do
    if q_has "p2_$TAG"; then sleep 300; continue; fi
    M=$(missing_idx outputs_affcache 'affinity_embeddings_*.npz' "p2_$TAG")
    [ -z "$M" ] && break
    log "Pass-2 resweep missing: $M"
    sbatch --job-name="p2_$TAG" --export=ALL,TARGET="$TARGET" --array="$M"%40 "$ROOT/slurm/pass2_affcache.sbatch"
    sleep 300
  done
  log "Pass-2 complete (affcache=$(affcache_count))"
else
  log "Stage 2 already done (affcache=$(affcache_count))"
fi

# ---------------- Stage 3: base scoring + multi-seed FT ----------------
NEVALC=$(ls "$R/ft_inputs_full/eval_chunks"/chunk_*.txt 2>/dev/null | wc -l)
base_done(){ [ "$(ls "$R/headft_affcache/base/scores"/chunk_*.csv 2>/dev/null | wc -l)" -ge "$NEVALC" ]; }
if ! base_done && ! q_has "base_$TAG"; then
  log "Stage 3a: submit base (No-FT) scoring"; sbatch --job-name="base_$TAG" --export=ALL,TARGET="$TARGET" "$ROOT/slurm/score_base.sbatch"; sleep 30
fi
if ! q_has "ft_$TAG"; then
  # submit FT array only if any of the 15 arms is missing its scores
  need_ft=0
  for N in 40 100 300; do for S in 0 1 2 3 4; do
    d="$R/headft_affcache/lightning_top${N}_seed${S}/scores"
    [ "$(ls "$d"/chunk_*.csv 2>/dev/null | wc -l)" -ge "$NEVALC" ] || need_ft=1
  done; done
  if [ "$need_ft" = 1 ]; then log "Stage 3b: submit multi-seed FT array 1-15"; sbatch --job-name="ft_$TAG" --export=ALL,TARGET="$TARGET" --array=1-15%8 "$ROOT/slurm/train_score_seed.sbatch"; sleep 30; fi
fi
log "Stage 3: waiting for base + FT scoring to finish"
while q_has "base_$TAG" || q_has "ft_$TAG"; do sleep 300; done
log "Stage 3 complete"

# ---------------- Stage 4: evaluation ----------------
log "Stage 4: multi-seed evaluation"
python "$ROOT/pipeline_local/eval_headft_seeds.py" --targets "$TARGET" --out-dir "$WORKDIR/analysis/$TARGET"
log "DONE. Results in $WORKDIR/analysis/$TARGET (per_seed.csv, spread.csv, table2.csv)"

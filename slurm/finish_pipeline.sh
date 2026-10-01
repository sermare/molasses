#!/bin/bash
# Idempotent finish driver: tail Pass-2 + merge + build_ft_inputs + base + multi-seed FT + eval.
# Assumes Pass-2 for the 140 complete chunks is already submitted/running.
set -uo pipefail
TARGET="${1:-588689}"
ROOT=/global/scratch/users/sergiomar10/boltzaff
source /clusterfs/nilah/sergio/miniconda3/etc/profile.d/conda.sh; conda activate boltzft
source "$ROOT/env.sh" "$TARGET"
NCHUNK=$(wc -l < "$R/chunks.tsv")
NEED=$(wc -l < "$R/inputs_full"/../selection/inference_compounds.csv 2>/dev/null || echo 49985)
log(){ echo "[finish $(date '+%F %T')] $*"; }
q_has(){ squeue -u "$USER" -h -n "$1" -o "%i" 2>/dev/null | grep -q .; }
affcache_ct(){ find -L "$R/outputs_affcache" -name 'affinity_embeddings_*.npz' 2>/dev/null | wc -l; }

# incomplete chunk indices for a given subdir/glob (fast: count prediction/cache dirs per chunk)
incomplete(){ # $1=subdir under outputs (outputs_chunks_full|outputs_affcache) $2=glob
python - "$1" "$2" <<'PY'
import sys,glob,os
sub,g=sys.argv[1],sys.argv[2]; R=os.environ["R"]
tsv=[l.split('\t') for l in open(R+"/chunks.tsv").read().splitlines()]
out=[]
for idx,_,_,od in tsv:
    name=os.path.basename(od); d=f"{R}/{sub}/{name}"
    n=len(glob.glob(f"{d}/boltz_results_*/predictions/*/{g}"))
    if n<345: out.append(idx)
print(",".join(out))
PY
}

# ---- Stage 1: wait for tail Pass-1, then FULL merge + build_ft_inputs ----
while q_has bft_p1; do sleep 120; done
log "Pass-1 done; running full merge_consolidate"
python "$BOLTZFT/pipeline/merge_consolidate.py" --outputs-root "$R/outputs_chunks_full" --inputs-dir "$R/inputs_full" --out-dir "$R/consolidated_full"
if [ ! -f "$R/ft_inputs_full/eval_ids.txt" ]; then
  log "build_ft_inputs_full"
  python "$BOLTZFT/pipeline/build_ft_inputs_full.py" --target "$TARGET" --results-csv "$RESULTS_CSV" --consolidated-dir "$R/consolidated_full" --out-dir "$R/ft_inputs_full"
fi
log "ft_inputs ready: eval_ids=$(wc -l < "$R/ft_inputs_full/eval_ids.txt")"

# ---- Stage 2: Pass-2 for the 3 tail chunks, then wait+resweep all Pass-2 ----
sbatch --export=ALL,TARGET="$TARGET" --array=138,139,143 "$ROOT/slurm/pass2_affcache.sbatch"; sleep 60
while true; do
  if q_has bft_p2; then sleep 180; continue; fi
  M=$(incomplete outputs_affcache 'affinity_embeddings_*.npz')
  [ -z "$M" ] && break
  log "Pass-2 resweep missing: $M"; sbatch --export=ALL,TARGET="$TARGET" --array="$M"%40 "$ROOT/slurm/pass2_affcache.sbatch"; sleep 180
done
log "Pass-2 complete (affcache=$(affcache_ct))"

# ---- Stage 3: base scoring + 15 multi-seed FT arms ----
NEVALC=$(ls "$R/ft_inputs_full/eval_chunks"/chunk_*.txt 2>/dev/null | wc -l)
[ "$(ls "$R/headft_affcache/base/scores"/chunk_*.csv 2>/dev/null|wc -l)" -ge "$NEVALC" ] || { q_has bft_base || sbatch --export=ALL,TARGET="$TARGET" "$ROOT/slurm/score_base.sbatch"; }
q_has bft_ft || sbatch --export=ALL,TARGET="$TARGET" --array=1-15%8 "$ROOT/slurm/train_score_seed.sbatch"
sleep 60
while q_has bft_base || q_has bft_ft; do sleep 180; done
log "base + FT scoring complete"

# ---- Stage 4: eval ----
python "$ROOT/pipeline_local/eval_headft_seeds.py" --targets "$TARGET" --out-dir "$WORKDIR/analysis/$TARGET"
log "DONE -> $WORKDIR/analysis/$TARGET/{per_seed,spread,table2}.csv"

#!/bin/bash
# Pass 1: Boltz-2 co-folding -> cropped trunk/structure embeddings (the expensive step).
# Array job: each task processes one chunk (a line of chunks.tsv).
#SBATCH --account=co_nilah
#SBATCH --partition=savio3_gpu
#SBATCH --qos=savio_lowprio
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --exclude=n0143.savio3,n0144.savio3,n0215.savio3,n0176.savio3
#SBATCH --time=12:00:00   # was 6 h: tail resweeps timed out on a busy queue (folding resumes per compound, so a longer limit only lets slow nodes finish)
#SBATCH --job-name=bft_p1
#SBATCH --output=/global/scratch/users/sergiomar10/boltzaff/slurm/logs/p1_%A_%a.out
#SBATCH --error=/global/scratch/users/sergiomar10/boltzaff/slurm/logs/p1_%A_%a.err
set -eo pipefail
TARGET="${TARGET:?set TARGET}"
source /clusterfs/nilah/sergio/miniconda3/etc/profile.d/conda.sh
conda activate boltzft
source /global/scratch/users/sergiomar10/boltzaff/env.sh "$TARGET"

# Pick this task's chunk from the index (1-based idx == SLURM_ARRAY_TASK_ID).
LINE=$(awk -v i="$SLURM_ARRAY_TASK_ID" '$1==i{print; exit}' "$R/chunks.tsv")
[ -z "$LINE" ] && { echo "no chunk for idx=$SLURM_ARRAY_TASK_ID"; exit 1; }
CHUNK_DIR=$(echo "$LINE" | cut -f3)
OUT_DIR=$(echo "$LINE" | cut -f4)
echo "[p1] task=$SLURM_ARRAY_TASK_ID chunk=$CHUNK_DIR -> $OUT_DIR start=$(date)"
echo "[p1] n_yaml=$(find -L "$CHUNK_DIR" -maxdepth 1 -name '*.yaml' | wc -l)"

# --- fold only what is missing ---------------------------------------------------------------
# A chunk re-run used to re-fold all ~350 compounds (--override), so a chunk missing 8 cost the same
# as one missing 350. Find the compounds without a finished prediction and fold just those.
CH=$(basename "$CHUNK_DIR")
PRED="$OUT_DIR/boltz_results_$CH/predictions"
TODO=$(for y in "$CHUNK_DIR"/*.yaml; do id=$(basename "$y" .yaml); [ -e "$PRED/$id/embeddings_${id}.npz" ] || echo "$id"; done)
NTODO=$(echo -n "$TODO" | grep -c . || true)
NALL=$(find -L "$CHUNK_DIR" -maxdepth 1 -name '*.yaml' | wc -l)
echo "[p1] compounds still to fold in $CH: $NTODO of $NALL"
if [ "$NTODO" -eq 0 ]; then echo "[p1] chunk already complete"; exit 0; fi

if [ "$NTODO" -eq "$NALL" ]; then
  # fresh chunk: fold everything as before
  boltz predict "$CHUNK_DIR" \
    --out_dir "$OUT_DIR" \
    --cache "$BOLTZ_CACHE" --model boltz2 \
    --devices 1 --accelerator gpu \
    --num_workers 2 --preprocessing-threads 1 \
    --write_embeddings_cropped --skip_affinity_prediction --override
else
  # partial chunk: fold only the missing compounds into a separate '<chunk>_tail_<job>' results folder.
  # merge_consolidate globs chunk_*/boltz_results_*, so it is picked up with its own manifest; the folded
  # compounds are also linked into the original predictions folder so the driver's completeness check sees them.
  TAILNAME="${CH}_tail_${SLURM_JOB_ID:-x}"
  TAILIN="$(dirname "$CHUNK_DIR")/$TAILNAME"; TAILOUT="$(dirname "$OUT_DIR")/$TAILNAME"
  rm -rf "$TAILIN"; mkdir -p "$TAILIN"
  for id in $TODO; do ln -sf "$(readlink -f "$CHUNK_DIR/$id.yaml")" "$TAILIN/$id.yaml"; done
  echo "[p1] tail mode: $NTODO compounds -> $TAILOUT"
  boltz predict "$TAILIN" \
    --out_dir "$TAILOUT" \
    --cache "$BOLTZ_CACHE" --model boltz2 \
    --devices 1 --accelerator gpu \
    --num_workers 2 --preprocessing-threads 1 \
    --write_embeddings_cropped --skip_affinity_prediction --override
  mkdir -p "$PRED"
  for d in "$TAILOUT/boltz_results_$TAILNAME/predictions"/*/; do
    id=$(basename "$d"); [ -e "$d/embeddings_${id}.npz" ] || continue
    if [ -d "$PRED/$id" ] && [ ! -L "$PRED/$id" ]; then rm -rf "$PRED/$id"; fi   # drop a stale partial folder
    ln -sfn "$(readlink -f "$d")" "$PRED/$id"
  done
fi
echo "[p1] done task=$SLURM_ARRAY_TASK_ID $(date)"

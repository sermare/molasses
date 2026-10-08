#!/bin/bash
# Pass-1 folding of one small, chunk-pure list of compounds (see slurm/make_p1_minis.py). Array task N <-> row N of results/runs/<t>/p1_mini/index.tsv.
# Results go to <chunk>_tail_mini<jobid> and are linked into the chunk's predictions folder, exactly like the tail mode of pass1_embed.sbatch.
#SBATCH --account=co_nilah
#SBATCH --partition=savio3_gpu
#SBATCH --qos=savio_lowprio
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --exclude=n0143.savio3,n0144.savio3,n0215.savio3,n0176.savio3,n0214.savio3
#SBATCH --time=01:30:00
#SBATCH --constraint=a40|8rtx      # A40 (48 GB) and TITAN nodes only: on the 11 GB GTX 2080 Ti nodes (4rtx) every compound runs out of GPU memory and boltz still exits 0
#SBATCH --output=/global/scratch/users/sergiomar10/boltzaff/slurm/logs/p1mini_%A_%a.out
#SBATCH --error=/global/scratch/users/sergiomar10/boltzaff/slurm/logs/p1mini_%A_%a.err
set -eo pipefail
TARGET="${TARGET:?set TARGET}"
source /clusterfs/nilah/sergio/miniconda3/etc/profile.d/conda.sh; conda activate boltzft
source /global/scratch/users/sergiomar10/boltzaff/env.sh "$TARGET"
LINE=$(awk -F'\t' -v i="$SLURM_ARRAY_TASK_ID" '$1==i{print; exit}' "$R/p1_mini/index.tsv"); [ -z "$LINE" ] && { echo "no row $SLURM_ARRAY_TASK_ID"; exit 1; }
CH=$(echo "$LINE" | cut -f3); LIST=$(echo "$LINE" | cut -f4)
CHUNK_DIR="$R/inputs_chunks_full/$CH"; OUT_ROOT="$R/outputs_chunks_full"; PRED="$OUT_ROOT/$CH/boltz_results_$CH/predictions"
TAILNAME="${CH}_tail_mini${SLURM_JOB_ID}"; TAILIN="$R/inputs_chunks_full/$TAILNAME"; TAILOUT="$OUT_ROOT/$TAILNAME"; TP="$TAILOUT/boltz_results_$TAILNAME/predictions"
mkdir -p "$PRED"
link_done() {   # link every finished fold of this job (also after a requeue) into the chunk's predictions folder
  [ -d "$TP" ] || return 0
  for d in "$TP"/*/; do id=$(basename "$d"); [ -e "$d/embeddings_${id}.npz" ] || continue
    if [ -d "$PRED/$id" ] && [ ! -L "$PRED/$id" ]; then [ -e "$PRED/$id/embeddings_${id}.npz" ] && continue; rm -rf "$PRED/$id"; fi
    ln -sfn "$(readlink -f "$d")" "$PRED/$id"; done
}
link_done
TODO=$(for id in $(cat "$LIST"); do [ -e "$PRED/$id/embeddings_${id}.npz" ] || echo "$id"; done); NTODO=$(echo -n "$TODO" | grep -c . || true)
echo "[p1mini] task=$SLURM_ARRAY_TASK_ID $CH: $NTODO compounds to fold (list $(wc -l < "$LIST")) start=$(date)"
if [ "$NTODO" -eq 0 ]; then echo "[p1mini] nothing to do"; exit 0; fi
rm -rf "$TAILIN"; mkdir -p "$TAILIN"
for id in $TODO; do ln -sf "$(readlink -f "$CHUNK_DIR/$id.yaml")" "$TAILIN/$id.yaml"; done
boltz predict "$TAILIN" --out_dir "$TAILOUT" --cache "$BOLTZ_CACHE" --model boltz2 --devices 1 --accelerator gpu \
  --num_workers 2 --preprocessing-threads 4 --write_embeddings_cropped --skip_affinity_prediction --override || true
link_done
NLEFT=$(for id in $TODO; do [ -e "$PRED/$id/embeddings_${id}.npz" ] || echo x; done | wc -l); LIM=$(( NTODO / 10 )); [ "$LIM" -lt 3 ] && LIM=3
if [ "$NLEFT" -gt "$LIM" ] && [ "${SLURM_RESTART_COUNT:-0}" -lt 8 ]; then
  echo "[p1mini] $NLEFT of $NTODO still missing (pre-empted?); requeueing (restart ${SLURM_RESTART_COUNT:-0})"; scontrol requeue "$SLURM_JOB_ID"; exit 0
fi
echo "[p1mini] done task=$SLURM_ARRAY_TASK_ID left=$NLEFT $(date)"

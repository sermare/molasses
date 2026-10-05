#!/bin/bash
# Pass 2: build affinity-checkpoint caches by REUSING the stored Pass-1 poses.
# Array job: each task processes one chunk (a line of chunks.tsv). Depends on
# merge_consolidate having produced consolidated_full/structures.
#SBATCH --account=co_nilah
#SBATCH --partition=savio3_gpu
#SBATCH --qos=savio_lowprio
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --exclude=n0143.savio3,n0144.savio3,n0215.savio3,n0176.savio3
#SBATCH --time=04:00:00
#SBATCH --job-name=bft_p2
#SBATCH --output=/global/scratch/users/sergiomar10/boltzaff/slurm/logs/p2_%A_%a.out
#SBATCH --error=/global/scratch/users/sergiomar10/boltzaff/slurm/logs/p2_%A_%a.err
set -eo pipefail
TARGET="${TARGET:?set TARGET}"
source /clusterfs/nilah/sergio/miniconda3/etc/profile.d/conda.sh
conda activate boltzft
source /global/scratch/users/sergiomar10/boltzaff/env.sh "$TARGET"

LINE=$(awk -v i="$SLURM_ARRAY_TASK_ID" '$1==i{print; exit}' "$R/chunks.tsv")
[ -z "$LINE" ] && { echo "no chunk for idx=$SLURM_ARRAY_TASK_ID"; exit 1; }
CHUNK_DIR=$(echo "$LINE" | cut -f3)
NAME=$(basename "$CHUNK_DIR")
OUT_DIR="$R/outputs_affcache/$NAME"
export KERNELS=1   # full GPU
echo "[p2] task=$SLURM_ARRAY_TASK_ID chunk=$CHUNK_DIR -> $OUT_DIR start=$(date)"
bash "$BOLTZFT/pipeline/build_affinity_cache.sh" "$TARGET" "$CHUNK_DIR" "$OUT_DIR"
echo "[p2] done task=$SLURM_ARRAY_TASK_ID $(date)"

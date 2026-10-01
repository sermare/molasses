#!/bin/bash
# CPU array: extract ligand centroids (common frame) from each chunk's folded poses.
#SBATCH --account=co_nilah
#SBATCH --partition=savio3_htc
#SBATCH --qos=savio_lowprio
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=01:00:00
#SBATCH --job-name=bft_cen
#SBATCH --output=/global/scratch/users/sergiomar10/boltzaff/slurm/logs/cen_%A_%a.out
#SBATCH --error=/global/scratch/users/sergiomar10/boltzaff/slurm/logs/cen_%A_%a.err
set -eo pipefail
source /clusterfs/nilah/sergio/miniconda3/etc/profile.d/conda.sh; conda activate boltzba
source /global/scratch/users/sergiomar10/boltzaff/env.sh 588689
LINE=$(awk -v i="$SLURM_ARRAY_TASK_ID" '$1==i{print; exit}' "$R/chunks.tsv")
[ -z "$LINE" ] && { echo "no chunk"; exit 0; }
OUT_DIR=$(echo "$LINE" | cut -f4); NAME=$(basename "$OUT_DIR")
PRED="$OUT_DIR/boltz_results_$NAME/predictions"
[ -d "$PRED" ] || { echo "no predictions in $PRED (chunk not folded yet)"; exit 0; }
python "$ROOT/pipeline_local/pose_centroids.py" --pred-dir "$PRED" \
  --ref-ca "$R/pose_density/ref_ca.npy" --out-csv "$R/pose_density/centroids_$NAME.csv"

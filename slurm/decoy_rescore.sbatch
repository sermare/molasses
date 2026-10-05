#!/bin/bash
# Decoy-protein rescore: standard Boltz-2 co-fold + affinity for one 50-compound chunk of one arm (real / shuffled / other).
# Array id = line number in results/runs/588689/decoy/tasks.tsv (column 1 is 0-based). Resume-safe: boltz skips finished predictions.
#SBATCH --account=co_nilah
#SBATCH --partition=savio3_gpu
#SBATCH --qos=savio_lowprio
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --exclude=n0143.savio3,n0144.savio3,n0215.savio3,n0176.savio3
#SBATCH --time=03:00:00
#SBATCH --job-name=bft_decoy
#SBATCH --output=/global/scratch/users/sergiomar10/boltzaff/slurm/logs/decoy_%A_%a.out
#SBATCH --error=/global/scratch/users/sergiomar10/boltzaff/slurm/logs/decoy_%A_%a.err
set -eo pipefail
source /clusterfs/nilah/sergio/miniconda3/etc/profile.d/conda.sh; conda activate boltzft
source /global/scratch/users/sergiomar10/boltzaff/env.sh 588689
TASKS=/global/scratch/users/sergiomar10/boltzaff/results/runs/588689/decoy/tasks.tsv
LINE=$(awk -v i="$SLURM_ARRAY_TASK_ID" -F'\t' '$1==i-1{print; exit}' "$TASKS")
[ -z "$LINE" ] && { echo "no task for idx=$SLURM_ARRAY_TASK_ID"; exit 1; }
ARM=$(echo "$LINE" | cut -f2); IN=$(echo "$LINE" | cut -f3); OUT=$(echo "$LINE" | cut -f4)
echo "[decoy] task=$SLURM_ARRAY_TASK_ID arm=$ARM in=$IN start=$(date)"
boltz predict "$IN" --out_dir "$OUT" --cache "$BOLTZ_CACHE" --model boltz2 \
  --devices 1 --accelerator gpu --num_workers 2 --preprocessing-threads 1 --diffusion_samples_affinity 5
echo "[decoy] done task=$SLURM_ARRAY_TASK_ID $(date)"

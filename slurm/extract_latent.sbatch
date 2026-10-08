#!/bin/bash
# Extract the affinity head's internal features for the latent-space analysis. Array task 1..3 = arm noft, ft300s0, ft300s1 (see pipeline_local/extract_latent.py).
# Submit: sbatch --job-name=lat_<id> --export=ALL,TARGET=434954-2097 --array=1-3 slurm/extract_latent.sbatch
#SBATCH --account=co_nilah
#SBATCH --partition=savio3_gpu
#SBATCH --qos=savio_lowprio
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --exclude=n0143.savio3,n0144.savio3,n0215.savio3,n0176.savio3,n0214.savio3
#SBATCH --time=03:00:00
#SBATCH --output=/global/scratch/users/sergiomar10/boltzaff/slurm/logs/lat_%A_%a.out
set -eo pipefail
TARGET="${TARGET:?set TARGET}"
source /clusterfs/nilah/sergio/miniconda3/etc/profile.d/conda.sh; conda activate boltzft
source /global/scratch/users/sergiomar10/boltzaff/env.sh "$TARGET"
ARMS=(noft ft300s0 ft300s1); ARM=${ARMS[$((SLURM_ARRAY_TASK_ID-1))]}
python -u /global/scratch/users/sergiomar10/boltzaff/pipeline_local/extract_latent.py --target "$TARGET" --arm "$ARM" ${LIMIT:+--limit $LIMIT}

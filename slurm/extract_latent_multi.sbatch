#!/bin/bash
# Short GPU jobs of the latent extraction: array task k handles block k of the target's sample (about 200 compounds), all four models in one pass
# (pipeline_local/extract_latent_multi.py --shard k --nshards N). A short time limit lets the scheduler backfill these quickly. Use slurm/submit_latent_shards.sh <target>.
#SBATCH --account=co_nilah
#SBATCH --partition=savio3_gpu
#SBATCH --qos=savio_lowprio
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --exclude=n0143.savio3,n0144.savio3,n0215.savio3,n0176.savio3,n0214.savio3
#SBATCH --time=00:20:00
#SBATCH --output=/global/scratch/users/sergiomar10/boltzaff/slurm/logs/latm_%A_%a.out
set -eo pipefail
TARGET="${TARGET:?set TARGET}"
source /clusterfs/nilah/sergio/miniconda3/etc/profile.d/conda.sh; conda activate boltzft
source /global/scratch/users/sergiomar10/boltzaff/env.sh "$TARGET"
python -u /global/scratch/users/sergiomar10/boltzaff/pipeline_local/extract_latent_multi.py --target "$TARGET" --shard "$SLURM_ARRAY_TASK_ID" --nshards "$NSHARDS" --threads 8

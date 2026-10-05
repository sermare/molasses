#!/bin/bash
# Score the No-FT (base) arm once over all eval chunks with the pretrained affinity ckpt.
#SBATCH --account=co_nilah
#SBATCH --partition=savio3_gpu
#SBATCH --qos=savio_lowprio
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --exclude=n0143.savio3,n0144.savio3,n0215.savio3,n0176.savio3
#SBATCH --time=12:00:00   # one arm scores 50 chunks x ~7 min (~6 h); 2 h timed out (scoring is resume-friendly: it skips chunks already written)
#SBATCH --job-name=bft_base
#SBATCH --output=/global/scratch/users/sergiomar10/boltzaff/slurm/logs/base_%j.out
#SBATCH --error=/global/scratch/users/sergiomar10/boltzaff/slurm/logs/base_%j.err
set -eo pipefail
TARGET="${TARGET:?set TARGET}"
source /clusterfs/nilah/sergio/miniconda3/etc/profile.d/conda.sh
conda activate boltzft
source /global/scratch/users/sergiomar10/boltzaff/env.sh "$TARGET"
bash /global/scratch/users/sergiomar10/boltzaff/pipeline_local/score_arm.sh base "$TARGET"

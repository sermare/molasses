#!/bin/bash
# Multi-seed head-FT: array task -> (N, seed); trains then scores that arm.
# Grid = {40,100,300} x seeds{0..4} = 15 tasks. Submit: --array=1-15
#SBATCH --account=co_nilah
#SBATCH --partition=savio3_gpu
#SBATCH --qos=savio_lowprio
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --exclude=n0143.savio3,n0144.savio3,n0215.savio3,n0176.savio3
#SBATCH --time=12:00:00   # one arm scores 50 chunks x ~7 min (~6 h); 2 h timed out (scoring is resume-friendly: it skips chunks already written)
#SBATCH --job-name=bft_ft
#SBATCH --output=/global/scratch/users/sergiomar10/boltzaff/slurm/logs/ft_%A_%a.out
#SBATCH --error=/global/scratch/users/sergiomar10/boltzaff/slurm/logs/ft_%A_%a.err
set -eo pipefail
TARGET="${TARGET:?set TARGET}"
source /clusterfs/nilah/sergio/miniconda3/etc/profile.d/conda.sh
conda activate boltzft
source /global/scratch/users/sergiomar10/boltzaff/env.sh "$TARGET"
PL=/global/scratch/users/sergiomar10/boltzaff/pipeline_local

NS=(40 40 40 40 40 100 100 100 100 100 300 300 300 300 300)
SEEDS=(0 1 2 3 4 0 1 2 3 4 0 1 2 3 4)
i=$((SLURM_ARRAY_TASK_ID-1))
N=${NS[$i]}; SEED=${SEEDS[$i]}
echo "[ft] task=$SLURM_ARRAY_TASK_ID -> N=$N seed=$SEED $(date)"

bash "$PL/train_headft_seed.sh" "$TARGET" "$N" "$SEED"
bash "$PL/score_arm.sh" headft "$TARGET" "$N" "$SEED"
echo "[ft] complete N=$N seed=$SEED $(date)"

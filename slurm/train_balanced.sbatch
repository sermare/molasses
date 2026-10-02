#!/bin/bash
# Training data/composition experiment: array task -> (condition, seed).
# balanced_N300 only (conds.txt has one line) x 5 seeds = 5 tasks; 12 h limit because scoring 50 chunks takes ~7 h.
# Submit AFTER Pass-2 cache is complete: sbatch --export=ALL,TARGET=588689 --array=1-70%12 ...
#SBATCH --account=co_nilah
#SBATCH --partition=savio3_gpu
#SBATCH --qos=savio_lowprio
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --exclude=n0143.savio3,n0144.savio3,n0215.savio3
#SBATCH --time=12:00:00
#SBATCH --job-name=bft_balanced
#SBATCH --output=/global/scratch/users/sergiomar10/boltzaff/slurm/logs/var_%A_%a.out
#SBATCH --error=/global/scratch/users/sergiomar10/boltzaff/slurm/logs/var_%A_%a.err
set -eo pipefail
TARGET="${TARGET:?set TARGET}"
source /clusterfs/nilah/sergio/miniconda3/etc/profile.d/conda.sh; conda activate boltzft
source /global/scratch/users/sergiomar10/boltzaff/env.sh "$TARGET"
mapfile -t CONDS < "$R/ft_inputs_variants/conds.txt"
i=$((SLURM_ARRAY_TASK_ID-1)); c=$((i/5)); s=$((i%5))
bash /global/scratch/users/sergiomar10/boltzaff/pipeline_local/train_variant.sh "$TARGET" "${CONDS[$c]}" "$s"

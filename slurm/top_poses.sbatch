#!/bin/bash
# Fast pose generation for the top-ranked compounds (for visualization).
#SBATCH --account=co_nilah
#SBATCH --partition=savio3_gpu
#SBATCH --qos=savio_lowprio
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --exclude=n0143.savio3,n0144.savio3
#SBATCH --time=01:30:00
#SBATCH --job-name=bft_pose
#SBATCH --output=/global/scratch/users/sergiomar10/boltzaff/slurm/logs/pose_%j.out
#SBATCH --error=/global/scratch/users/sergiomar10/boltzaff/slurm/logs/pose_%j.err
set -eo pipefail
source /clusterfs/nilah/sergio/miniconda3/etc/profile.d/conda.sh; conda activate boltzft
source /global/scratch/users/sergiomar10/boltzaff/env.sh 588689
IN="$R/top_poses/inputs"; OUT="$R/top_poses/out"
echo "[pose] $(ls "$IN" | wc -l) compounds -> $OUT $(date)"
boltz predict "$IN" --out_dir "$OUT" --cache "$BOLTZ_CACHE" --model boltz2 \
  --devices 1 --accelerator gpu --num_workers 2 --preprocessing-threads 1 \
  --skip_affinity_prediction --override
echo "[pose] done $(date); cifs=$(find -L "$OUT" -name '*_model_0.cif' | wc -l)"

#!/bin/bash
# CPU, no-Boltz supervised baselines (LightGBM-ECFP + nearest-active Tanimoto). Runs in
# parallel with the GPU pipeline; needs only SMILES + labels + the ranking.
#SBATCH --account=co_nilah
#SBATCH --partition=savio3_htc
#SBATCH --qos=savio_lowprio
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=04:00:00
#SBATCH --job-name=bft_cpu
#SBATCH --output=/global/scratch/users/sergiomar10/boltzaff/slurm/logs/cpu_%j.out
#SBATCH --error=/global/scratch/users/sergiomar10/boltzaff/slurm/logs/cpu_%j.err
set -eo pipefail
TARGET="${TARGET:-588689}"
source /clusterfs/nilah/sergio/miniconda3/etc/profile.d/conda.sh; conda activate boltzba
cd /global/scratch/users/sergiomar10/boltzaff
python pipeline_local/cpu_baselines.py --target "$TARGET"

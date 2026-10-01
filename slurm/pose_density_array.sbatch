#!/bin/bash
#SBATCH --account=co_nilah
#SBATCH --partition=savio3_htc
#SBATCH --qos=savio_lowprio
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --job-name=bft_pdens
#SBATCH --output=/global/scratch/users/sergiomar10/boltzaff/slurm/logs/pdens_%A_%a.out
TS=(588689 504329 1053173-743445 493248-485317 434954-2097 540297-493091 463203-2650 624273-588549)
T=${TS[$((SLURM_ARRAY_TASK_ID-1))]}
source /clusterfs/nilah/sergio/miniconda3/etc/profile.d/conda.sh; conda activate boltzba
echo "==== $T $(date) ===="
python /global/scratch/users/sergiomar10/boltzaff/pipeline_local/pose_density_residue.py --target "$T" --sample 1500

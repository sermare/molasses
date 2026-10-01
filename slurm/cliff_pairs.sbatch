#!/bin/bash
#SBATCH --account=co_nilah
#SBATCH --partition=savio3_htc
#SBATCH --qos=savio_lowprio
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --job-name=bft_cliffs
#SBATCH --output=/global/scratch/users/sergiomar10/boltzaff/slurm/logs/cliffs_%j.out
source /clusterfs/nilah/sergio/miniconda3/etc/profile.d/conda.sh; conda activate boltzba
python /global/scratch/users/sergiomar10/boltzaff/pipeline_local/cliff_pairs.py --target "${TARGET:-588689}" --sim 0.6 --procs 4

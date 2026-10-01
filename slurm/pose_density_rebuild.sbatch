#!/bin/bash
#SBATCH --account=co_nilah
#SBATCH --partition=savio3_htc
#SBATCH --qos=savio_lowprio
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --job-name=bft_pdrebuild
#SBATCH --output=/global/scratch/users/sergiomar10/boltzaff/slurm/logs/pdrebuild_%j.out
source /clusterfs/nilah/sergio/miniconda3/etc/profile.d/conda.sh; conda activate boltzba
cd /global/scratch/users/sergiomar10/boltzaff/notebooks
python build_pose_density_all_nb.py && jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=400 13_pose_density_all.ipynb 2>&1 | tail -3
echo "REBUILT $(date)"

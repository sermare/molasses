#!/bin/bash
# Runs after the decoy array ends (any outcome): harvest the affinity outputs, then rebuild + execute notebook 19.
#SBATCH --account=co_nilah
#SBATCH --partition=savio3_htc
#SBATCH --qos=savio_lowprio
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --job-name=bft_decoyfin
#SBATCH --output=/global/scratch/users/sergiomar10/boltzaff/slurm/logs/decoyfin_%j.out
source /clusterfs/nilah/sergio/miniconda3/etc/profile.d/conda.sh; conda activate boltzba
cd /global/scratch/users/sergiomar10/boltzaff
python pipeline_local/decoy_harvest.py
cd notebooks && python build_decoy_nb.py && jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=900 19_decoy_protein.ipynb 2>&1 | tail -2
echo "DECOY_FINALIZE_DONE $(date)"

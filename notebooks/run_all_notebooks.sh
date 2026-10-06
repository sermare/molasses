#!/bin/bash
# Re-execute every notebook in place (they fill themselves from whatever targets are fully scored). Order: dependencies first.
cd /global/scratch/users/sergiomar10/boltzaff/notebooks
source /clusterfs/nilah/sergio/miniconda3/etc/profile.d/conda.sh; conda activate boltzba
export WORKDIR=/global/scratch/users/sergiomar10/boltzaff/results BOLTZFT=/global/scratch/users/sergiomar10/boltzaff/BoltzFT
for nb in 01_replication 03_ft_vs_noft_reranking 06_training_strategies 02_uncertainty 00_overview 04_structural_cliffs 05_pose_density_and_decoy 07_structure_confidence; do
  echo "$(date +%T) start $nb"
  jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=boltzba --ExecutePreprocessor.timeout=7200 $nb.ipynb > ../results/analysis/nb_$nb.log 2>&1 && echo "$(date +%T) ok $nb" || echo "$(date +%T) FAILED $nb"
done
echo ALLDONE

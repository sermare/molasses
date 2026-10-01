#!/bin/bash
set -uo pipefail
ROOT=/global/scratch/users/sergiomar10/boltzaff
source /clusterfs/nilah/sergio/miniconda3/etc/profile.d/conda.sh; conda activate boltzba
for T in 588689 504329 1053173-743445 493248-485317 434954-2097 540297-493091 463203-2650 624273-588549; do
  n=$(find -L "$ROOT/results/runs/$T/outputs_chunks_full" -maxdepth 6 -name '*_model_0.cif' 2>/dev/null | head -1 | wc -l)
  [ "$n" -eq 0 ] && { echo "[$T] no CIFs yet, skip"; continue; }
  echo "==== $T $(date) ===="
  python "$ROOT/pipeline_local/pose_density_residue.py" --target "$T" --sample 2500 2>&1 | tail -10
done
# refresh the self-filling notebook with whatever completed
cd "$ROOT/notebooks" && python build_pose_density_all_nb.py && \
  jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=300 13_pose_density_all.ipynb 2>&1 | tail -2
echo "POSE DENSITY ALL DONE $(date)"

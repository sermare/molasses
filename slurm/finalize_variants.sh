#!/bin/bash
# Wait for the 70 variant arms, run eval_variants, execute notebook 10, THEN resume the rollout.
set -uo pipefail
ROOT=/global/scratch/users/sergiomar10/boltzaff
source /clusterfs/nilah/sergio/miniconda3/etc/profile.d/conda.sh
resume(){ echo "[finvar $(date '+%F %T')] resuming rollout"; sbatch --export=ALL "$ROOT/slurm/rollout.sbatch" 2>&1 | tail -1; }
trap resume EXIT
VS=$ROOT/results/runs/588689/headft_variants_scores
mapfile -t CONDS < $ROOT/results/runs/588689/ft_inputs_variants/conds.txt
ncomplete(){ local n=0; for c in "${CONDS[@]}"; do for s in 0 1 2 3 4; do [ "$(ls $VS/${c}_seed${s}/chunk_*.csv 2>/dev/null|wc -l)" -eq 50 ]&&n=$((n+1)); done; done; echo $n; }
for i in $(seq 1 60); do   # up to ~5h
  n=$(ncomplete); echo "[finvar $(date '+%F %T')] variant arms complete: $n/70"
  [ "$n" -ge 70 ] && { echo "[finvar] all 70 complete"; break; }
  # if variant scoring stalled (no bft_var in queue) and incomplete, resubmit
  squeue -u $USER -h -n bft_var -o %i | grep -q . || { echo "[finvar] bft_var drained w/ $n/70 -> resubmit"; sbatch --export=ALL,TARGET=588689 --array=1-70%40 $ROOT/slurm/train_variants.sbatch >/dev/null; sleep 60; }
  sleep 300
done
conda activate boltzft; source $ROOT/env.sh 588689
python $ROOT/pipeline_local/eval_variants.py --target 588689 --out-dir $ROOT/results/analysis/588689 2>&1 | tail -20
conda activate boltzba
cd $ROOT/notebooks && python build_ft_strategies_nb.py && jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=400 10_ft_strategies.ipynb 2>&1 | tail -2
echo "[finvar] notebook 10 finalized; rollout resumes on exit"

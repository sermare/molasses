#!/bin/bash
# Wait for 588689's 16 main arms, finalize (eval + notebook 08), THEN resume the 7-target rollout.
# Durable; runs the rollout-resume in a trap so it happens even on timeout.
set -uo pipefail
ROOT=/global/scratch/users/sergiomar10/boltzaff
source /clusterfs/nilah/sergio/miniconda3/etc/profile.d/conda.sh
resume_rollout(){ echo "[fin $(date '+%F %T')] resuming rollout"; sbatch --export=ALL "$ROOT/slurm/rollout.sbatch" 2>&1 | tail -1; }
trap resume_rollout EXIT
AFF=$ROOT/results/runs/588689/headft_affcache
complete(){ [ "$(ls $AFF/base/scores/chunk_*.csv 2>/dev/null|wc -l)" -eq 50 ]||return 1
  for N in 40 100 300; do for S in 0 1 2 3 4; do [ "$(ls $AFF/lightning_top${N}_seed${S}/scores/chunk_*.csv 2>/dev/null|wc -l)" -eq 50 ]||return 1; done; done; }
for i in $(seq 1 40); do   # up to ~3.3h
  if complete; then echo "[fin] 588689 all 16 arms complete"; break; fi
  n=0; for N in 40 100 300; do for S in 0 1 2 3 4; do c=$(ls $AFF/lightning_top${N}_seed${S}/scores/chunk_*.csv 2>/dev/null|wc -l); [ "$c" -eq 50 ]&&n=$((n+1)); done; done
  echo "[fin $(date '+%F %T')] waiting; complete FT arms=$n/15"; sleep 240
done
conda activate boltzft; source $ROOT/env.sh 588689
python $ROOT/pipeline_local/eval_headft_seeds.py --targets 588689 --out-dir $ROOT/results/analysis/588689 2>&1 | tail -14
conda activate boltzba
cd $ROOT/notebooks && jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=300 08_replication_final.ipynb 2>&1 | tail -2
echo "[fin] notebook 08 finalized; rollout resumes on exit"

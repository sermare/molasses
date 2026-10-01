#!/bin/bash
# Prioritize 588689 finalization: wait for all 16 main arms, re-run notebook 08, then ALWAYS
# release the held rollout Pass-1 + orchestrator (even on timeout/error). Durable + self-contained.
set -uo pipefail
ROOT=/global/scratch/users/sergiomar10/boltzaff
P1_JOBS="39310755 39310761 39310766 39310776 39310787 39310797 39310822"
ROLL="39310744"
release(){ echo "[prio $(date '+%F %T')] releasing holds"; for j in $P1_JOBS $ROLL; do scontrol release "$j" 2>/dev/null; done; }
trap release EXIT
source /clusterfs/nilah/sergio/miniconda3/etc/profile.d/conda.sh
AFF=$ROOT/results/runs/588689/headft_affcache
complete(){ local n=$(ls $AFF/base/scores/chunk_*.csv 2>/dev/null|wc -l); [ "$n" -eq 50 ]||return 1
  for N in 40 100 300; do for S in 0 1 2 3 4; do [ "$(ls $AFF/lightning_top${N}_seed${S}/scores/chunk_*.csv 2>/dev/null|wc -l)" -eq 50 ]||return 1; done; done; }
for i in $(seq 1 48); do   # up to ~4h (48*300s)
  if complete; then echo "[prio] 588689 all 16 arms complete"; break; fi
  echo "[prio $(date '+%F %T')] waiting; complete arms=$(ls $AFF/base/scores/chunk_*.csv 2>/dev/null|wc -l|xargs -I{} echo base={})"; sleep 300
done
conda activate boltzft; source $ROOT/env.sh 588689
python $ROOT/pipeline_local/eval_headft_seeds.py --targets 588689 --out-dir $ROOT/results/analysis/588689 2>&1 | tail -15
conda activate boltzba
cd $ROOT/notebooks && jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=300 08_replication_final.ipynb 2>&1 | tail -3
echo "[prio] notebook 08 refreshed with final numbers"

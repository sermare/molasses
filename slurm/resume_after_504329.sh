#!/bin/bash
# Wait for 504329's pipeline to finish (table2.csv), then resume the full 7-target rollout.
set -uo pipefail
ROOT=/global/scratch/users/sergiomar10/boltzaff
resume(){ echo "[resume $(date '+%F %T')] resuming rollout"; sbatch --export=ALL "$ROOT/slurm/rollout.sbatch" 2>&1|tail -1; }
trap resume EXIT
for i in $(seq 1 96); do   # up to ~8h
  [ -f "$ROOT/results/analysis/504329/table2.csv" ] && { echo "[resume] 504329 table2 present"; break; }
  # if 504329 driver died without finishing, relaunch it (per-target name)
  squeue -u $USER -h -n drv_504329 -o %i | grep -q . || { echo "[resume] drv_504329 gone, relaunching"; sbatch --job-name=drv_504329 --export=ALL,TARGET=504329 "$ROOT/slurm/driver.sbatch" >/dev/null; sleep 60; }
  echo "[resume $(date '+%F %T')] waiting for 504329 pipeline"; sleep 300
done

#!/bin/bash
# Relink tail folds, submit Pass-1 resweeps for both remaining targets, and restart their drivers (normal priority, wider array). Safe to re-run.
cd /global/scratch/users/sergiomar10/boltzaff
source /clusterfs/nilah/sergio/miniconda3/etc/profile.d/conda.sh; conda activate boltzft
for t in 463203-2650 624273-588549; do echo "=== $t $(date +%T)"; python slurm/resweep_pass1.py $t --submit --conc 45 2>&1 | grep -E "relinked|incomplete|Submitted|returncode" | cut -c1-250; done
squeue -u $USER -h -n drv_463203-2650 -o %i | grep -q . || sbatch --job-name=drv_463203-2650 --export=ALL,TARGET=463203-2650,DRIVER_NICE=0,ARRAY_CONC=45 slurm/driver.sbatch
squeue -u $USER -h -n drv_624273-588549 -o %i | grep -q . || sbatch --job-name=drv_624273-588549 --export=ALL,TARGET=624273-588549,DRIVER_NICE=0,ARRAY_CONC=45 slurm/driver.sbatch
echo "RESTART-DONE $(date +%T)"

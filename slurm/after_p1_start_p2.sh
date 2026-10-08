#!/bin/bash
# Hand-off from Pass-1 to Pass-2 for one target, without letting Pass-2 compete with the last Pass-1 jobs.
#   slurm/after_p1_start_p2.sh <run dir name> <assay id>        e.g. 463203-2650 2650
# 1. wait until no p1m_<id> job is PENDING (every remaining Pass-1 job already has a GPU)
# 2. consolidate (parallel), build Pass-2 jobs for every folded compound that has no result yet (--redo re-plans the ones that were cancelled), submit them at lower priority
# 3. wait until no p1m_<id> job is left at all, consolidate again and add Pass-2 jobs for the compounds folded last
set -u
T="$1"; TAG="$2"; ROOT=/global/scratch/users/sergiomar10/boltzaff; R=$ROOT/results/runs/$T; cd "$ROOT"
source /clusterfs/nilah/sergio/miniconda3/etc/profile.d/conda.sh; conda activate boltzft
log(){ echo "[$(date +%T)] $*"; }
cons(){ python -u slurm/consolidate_parallel.py --outputs-root $R/outputs_chunks_full --inputs-dir $R/inputs_full --out-dir $R/consolidated_full | tail -3; }
while squeue -u $USER -h -t PD -n p1m_$TAG | grep -q .; do sleep 60; done
log "no Pass-1 job pending any more: consolidating and starting Pass-2"
python slurm/clean_empty_tails.py $T | tail -2; cons
python -u slurm/make_p2_minis.py $T --size 60 --nice 50 --conc 60 --redo --submit
while squeue -u $USER -h -n p1m_$TAG | grep -q .; do sleep 60; done
log "all Pass-1 jobs finished: final consolidation and Pass-2 for the last compounds"
python slurm/clean_empty_tails.py $T | tail -2; cons
python -u slurm/make_p2_minis.py $T --size 60 --nice 50 --conc 60 --submit
log "hand-off done"

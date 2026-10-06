#!/bin/bash
# Wait for 2097's Pass-2 repairs to leave the queue, confirm every eval id is cached, then submit base + head-FT scoring via the watchdog (checkpoints already exist, so only scoring runs).
cd /global/scratch/users/sergiomar10/boltzaff
source /clusterfs/nilah/sergio/miniconda3/etc/profile.d/conda.sh; conda activate boltzba
while true; do
  if [ -z "$(squeue -u $USER -h -o %j | grep -x p2_2097)" ]; then
    n=$(python3 - <<'PY'
import os,glob
R="results/runs/434954-2097"; fin=set()
for p in glob.glob(f"{R}/outputs_affcache/chunk_*/boltz_results_*/predictions/*/affinity_embeddings_*.npz"): fin.add(os.path.basename(p)[20:-4])
print(sum(x not in fin for x in open(f"{R}/ft_inputs_full/eval_ids.txt").read().split()))
PY
)
    echo "$(date +%T) missing eval ids: $n"
    if [ "$n" = "0" ]; then python pipeline_local/watchdog.py --submit; break; fi
    echo "still missing $n with no repair queued -- needs manual repair"; break
  fi
  sleep 300
done

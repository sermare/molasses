#!/bin/bash
# Build each target's UMAP grid as soon as all of its extraction shards exist (merge shards, then pipeline_local/umap_grid.py). Polls every 20 s; exits when every target has its figure.
ROOT=/global/scratch/users/sergiomar10/boltzaff; cd "$ROOT"; PY=${UMAP_PY:?set UMAP_PY to the python of the umap environment}
source /clusterfs/nilah/sergio/miniconda3/etc/profile.d/conda.sh; conda activate boltzba
TARGETS="434954-2097 588689 504329 1053173-743445 493248-485317 540297-493091"
while true; do
  left=0
  for t in $TARGETS; do
    [ -f results/analysis/$t/latent/umap_grid.png ] && continue
    left=$((left+1))
    ok=$(cd pipeline_local && python -c "import bft_latent as L; print(int(L.merge_shards('$t')))" 2>/dev/null)
    if [ "$ok" = "1" ]; then echo "[$(date +%T)] $t: shards complete, building the UMAP grid"; $PY -u pipeline_local/umap_grid.py $t 2>&1 | grep -v Warning | tail -3; fi
  done
  [ $left -eq 0 ] && { echo "all figures done"; break; }
  sleep 20
done

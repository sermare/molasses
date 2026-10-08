#!/bin/bash
# slurm/submit_latent_shards.sh <run dir name> [compounds per shard, default 200]  -> writes shards/manifest.json and submits the array
T="$1"; PER="${2:-200}"; ROOT=/global/scratch/users/sergiomar10/boltzaff; TAG="${T##*-}"; L="$ROOT/results/analysis/$T/latent"
N=$(( ( $(tail -n +2 "$L/sample.csv" | wc -l) + PER - 1 ) / PER ))      # rows of the sample (a quoted SMILES never contains a newline)
mkdir -p "$L/shards"; echo "{\"nshards\": $N}" > "$L/shards/manifest.json"
sbatch --nice=0 --job-name=latm_$TAG --export=ALL,TARGET=$T,NSHARDS=$N --array=0-$((N-1)) "$ROOT/slurm/extract_latent_multi.sbatch"

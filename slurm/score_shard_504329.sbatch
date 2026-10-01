#!/bin/bash
# Shard worker: scores eval chunks (highest index first) for all 16 arms of 504329. Each (arm,chunk) is claimed with an atomic mkdir lock,
# written to a temp file and renamed, so many workers can run at once. Existing csvs are skipped (resume-friendly).
#SBATCH --account=co_nilah
#SBATCH --partition=savio3_gpu
#SBATCH --qos=savio_lowprio
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --exclude=n0143.savio3,n0144.savio3,n0215.savio3
#SBATCH --time=06:00:00
#SBATCH --job-name=ft_504329
#SBATCH --output=/global/scratch/users/sergiomar10/boltzaff/slurm/logs/shard_%A_%a.out
set -eo pipefail
TARGET=504329
source /clusterfs/nilah/sergio/miniconda3/etc/profile.d/conda.sh; conda activate boltzft
source /global/scratch/users/sergiomar10/boltzaff/env.sh "$TARGET"
R="$WORKDIR/runs/$TARGET"; CACHE="$R/outputs_affcache"; INFER="$BOLTZFT/pipeline/affinity_cached_infer.py"
LOCKS="$R/headft_affcache/locks"; mkdir -p "$LOCKS"
ARMS=(base); for N in 40 100 300; do for S in 0 1 2 3 4; do ARMS+=("top${N}_seed${S}"); done; done
for c in $(seq 49 -1 0); do
  name=$(printf "chunk_%03d" $c); split="$R/ft_inputs_full/eval_chunks/$name.txt"; [ -f "$split" ] || continue
  for a in "${ARMS[@]}"; do
    if [ "$a" = base ]; then CK="$BOLTZ_CACHE/boltz2_aff.ckpt"; D="$R/headft_affcache/base/scores"
    else CK="$R/headft_full/$a/checkpoints/last.ckpt"; D="$R/headft_affcache/lightning_$a/scores"; fi
    out="$D/$name.csv"; [ -f "$out" ] && continue
    mkdir -p "$D"; mkdir "$LOCKS/${a}_$name" 2>/dev/null || continue
    [ -f "$out" ] && continue
    echo "[shard $SLURM_ARRAY_TASK_ID] $a $name $(date +%T)"
    python "$INFER" --pretrained "$CK" --cache-dir "$CACHE" --split-file "$split" --out-csv "$out.tmp" --no-kernels && mv "$out.tmp" "$out" || rm -f "$out.tmp"
  done
done
echo "[shard] done $(date)"

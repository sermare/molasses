#!/bin/bash
# Score ONE eval chunk of one head-FT arm (or the base model): lets the last few chunks of a slow arm run in parallel.
# The main arm job skips any chunk whose csv already exists, so this is safe alongside it. Output goes to a temp file and is renamed when complete.
# Submit: sbatch --export=ALL,TARGET=434954-2097,N=300,SEED=1,CHUNK=049 --job-name=chunk_2097 slurm/score_chunk.sbatch    (N=0 -> base model)
#SBATCH --account=co_nilah
#SBATCH --partition=savio3_gpu
#SBATCH --qos=savio_lowprio
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --exclude=n0143.savio3,n0144.savio3,n0215.savio3,n0176.savio3
#SBATCH --time=02:00:00
#SBATCH --output=/global/scratch/users/sergiomar10/boltzaff/slurm/logs/chunk_%j.out
set -eo pipefail
source /clusterfs/nilah/sergio/miniconda3/etc/profile.d/conda.sh; conda activate boltzft
source /global/scratch/users/sergiomar10/boltzaff/env.sh "$TARGET"
R="$WORKDIR/runs/$TARGET"
if [ "$N" = "0" ]; then CK="$BOLTZ_CACHE/boltz2_aff.ckpt"; D="$R/headft_affcache/base/scores"
else CK="$R/headft_full/top${N}_seed${SEED}/checkpoints/last.ckpt"; D="$R/headft_affcache/lightning_top${N}_seed${SEED}/scores"; fi
out="$D/chunk_$CHUNK.csv"; [ -f "$out" ] && { echo "already done"; exit 0; }
tmp="$D/.tmp_chunk_${CHUNK}_$SLURM_JOB_ID.csv"
python "$BOLTZFT/pipeline/affinity_cached_infer.py" --pretrained "$CK" --cache-dir "$R/outputs_affcache" \
  --split-file "$R/ft_inputs_full/eval_chunks/chunk_$CHUNK.txt" --out-csv "$tmp" --no-kernels
[ -f "$out" ] && rm -f "$tmp" || mv "$tmp" "$out"
echo "done chunk $CHUNK"

#!/bin/bash
# Re-score single base-model eval chunks in parallel: array task i -> chunk_<i>. Skips a chunk whose csv already has one row per compound in the chunk list.
#SBATCH --account=co_nilah
#SBATCH --partition=savio3_gpu
#SBATCH --qos=savio_lowprio
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --exclude=n0143.savio3,n0144.savio3,n0215.savio3,n0176.savio3,n0175.savio3,n0214.savio3
#SBATCH --time=00:25:00
#SBATCH --job-name=basechunk
#SBATCH --output=/global/scratch/users/sergiomar10/boltzaff/slurm/logs/basechunk_%A_%a.out
set -eo pipefail
TARGET="${TARGET:?set TARGET}"
source /clusterfs/nilah/sergio/miniconda3/etc/profile.d/conda.sh; conda activate boltzft
source /global/scratch/users/sergiomar10/boltzaff/env.sh "$TARGET"
R="$WORKDIR/runs/$TARGET"; name=$(printf "chunk_%03d" "$SLURM_ARRAY_TASK_ID"); split="$R/ft_inputs_full/eval_chunks/$name.txt"; out="$R/headft_affcache/base/scores/$name.csv"
want=$(grep -c . "$split"); have=$(( $(wc -l < "$out" 2>/dev/null || echo 1) - 1 )); [ -f "$out" ] && [ "$have" -ge "$want" ] && { echo "$name complete ($have rows)"; exit 0; }
rm -f "$out"
python "$BOLTZFT/pipeline/affinity_cached_infer.py" --pretrained "$BOLTZ_CACHE/boltz2_aff.ckpt" --cache-dir "$R/outputs_affcache" --split-file "$split" --out-csv "$out" --no-kernels
echo "[basechunk] $name done $(date)"

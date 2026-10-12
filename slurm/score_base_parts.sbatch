#!/bin/bash
# Re-score a quarter of one base eval chunk (array task N <-> row N of ft_inputs_full/base_redo/index.tsv); when all four parts of the chunk exist, merge them into the chunk csv.
#SBATCH --account=co_nilah
#SBATCH --partition=savio3_gpu
#SBATCH --qos=savio_lowprio
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --exclude=n0143.savio3,n0144.savio3,n0215.savio3,n0176.savio3,n0175.savio3,n0214.savio3
#SBATCH --time=00:15:00
#SBATCH --job-name=basepart
#SBATCH --output=/global/scratch/users/sergiomar10/boltzaff/slurm/logs/basepart_%A_%a.out
set -eo pipefail
TARGET="${TARGET:?set TARGET}"
source /clusterfs/nilah/sergio/miniconda3/etc/profile.d/conda.sh; conda activate boltzft
source /global/scratch/users/sergiomar10/boltzaff/env.sh "$TARGET"
R="$WORKDIR/runs/$TARGET"; D="$R/ft_inputs_full/base_redo"; LINE=$(awk -F'\t' -v i="$SLURM_ARRAY_TASK_ID" '$1==i{print;exit}' "$D/index.tsv"); C=$(echo "$LINE" | cut -f2); K=$(echo "$LINE" | cut -f3)
name=$(printf "chunk_%03d" "$C"); split="$D/splits/${name}_p$K.txt"; part="$D/out/${name}_p$K.csv"
want=$(grep -c . "$split"); [ -f "$part" ] && [ $(( $(wc -l < "$part") - 1 )) -ge "$want" ] || { rm -f "$part"; python "$BOLTZFT/pipeline/affinity_cached_infer.py" --pretrained "$BOLTZ_CACHE/boltz2_aff.ckpt" --cache-dir "$R/outputs_affcache" --split-file "$split" --out-csv "$part" --no-kernels; }
if [ -f "$D/out/${name}_p0.csv" ] && [ -f "$D/out/${name}_p1.csv" ] && [ -f "$D/out/${name}_p2.csv" ] && [ -f "$D/out/${name}_p3.csv" ]; then
  tmp="$R/headft_affcache/base/scores/.${name}.tmp$$"; head -1 "$D/out/${name}_p0.csv" > "$tmp"; for k in 0 1 2 3; do tail -n +2 "$D/out/${name}_p$k.csv" >> "$tmp"; done
  full=$(grep -c . "$R/ft_inputs_full/eval_chunks/$name.txt"); rows=$(( $(wc -l < "$tmp") - 1 ))
  if [ "$rows" -eq "$full" ]; then mv "$tmp" "$R/headft_affcache/base/scores/$name.csv"; echo "merged $name ($rows rows)"; else rm -f "$tmp"; echo "$name parts incomplete ($rows of $full)"; fi
fi

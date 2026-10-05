#!/bin/bash
# Pass-2 repair for any target: recompute the affinity cache for the compounds listed in <run>/p2_tail/inputs/mini_NNN (small chunks, one GPU each),
# then copy each result into its original chunk folder. Used when whole-chunk Pass-2 tasks were cancelled or left partial chunks.
# Submit: sbatch --job-name=p2_<assay id> --export=ALL,TARGET=<run dir name> --array=1-<N> slurm/pass2_tail.sbatch   (N = number of mini_ dirs)
#SBATCH --account=co_nilah
#SBATCH --partition=savio3_gpu
#SBATCH --qos=savio_lowprio
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --exclude=n0143.savio3,n0144.savio3,n0215.savio3,n0176.savio3
#SBATCH --time=03:00:00
#SBATCH --output=/global/scratch/users/sergiomar10/boltzaff/slurm/logs/p2tail_%A_%a.out
set -eo pipefail
TARGET="${TARGET:?set TARGET}"
source /clusterfs/nilah/sergio/miniconda3/etc/profile.d/conda.sh; conda activate boltzft
source /global/scratch/users/sergiomar10/boltzaff/env.sh "$TARGET"
M=$(printf "%03d" $((SLURM_ARRAY_TASK_ID-1)))
MINI=$R/p2_tail; export KERNELS=1
bash "$BOLTZFT/pipeline/build_affinity_cache.sh" "$TARGET" "$MINI/inputs/mini_$M" "$MINI/outputs/mini_$M"
python3 - "$MINI" "$M" "$R" <<'PY'
import sys, shutil, csv, os, glob
mini, m, run = sys.argv[1:4]
ids = {os.path.basename(f)[:-5] for f in glob.glob(f"{mini}/inputs/mini_{m}/*.yaml")}
chunk = {r[1]: r[0] for r in csv.reader(open(f"{mini}/missing.tsv"), delimiter="\t")}
src = glob.glob(f"{mini}/outputs/mini_{m}/boltz_results_*/predictions")[0]
for i in ids:
    s = f"{src}/{i}"
    if not os.path.exists(f"{s}/affinity_embeddings_{i}.npz"): print("MISSING", i); sys.exit(1)
    k = chunk[i]; d = f"{run}/outputs_affcache/chunk_{k}/boltz_results_chunk_{k}/predictions/{i}"
    shutil.rmtree(d, ignore_errors=True); shutil.copytree(s, d)
print("moved", len(ids))
PY

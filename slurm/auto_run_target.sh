#!/bin/bash
# Hands-off finish for one target whose Pass-1 is consolidated and ft_inputs_full is built:
# (Variant for a run that was started on a trimmed evaluation set: eval_ids.txt / eval_chunks hold only compounds with a cache, the full definition is eval_ids_full.txt.
#  base scoring and the 15 arms are already submitted; when every compound is cached the trimmed compounds are added as extra eval chunks so the arms score them too.)
#   1. while any train/eval compound lacks a Pass-2 result: when no p2_<tag> job is queued, plan the missing ones into small 20-compound minis and submit them (15 min limit, any GPU)
#   2. when nothing is missing: delete the memoized cache_index.json, submit base scoring and the 15 head-FT arms (train + score)
#   3. every loop: resubmit any arm that stopped without a live job (watchdog.py), nothing is ever held or paused
#   4. when all 16 arms are scored: run the evaluation (table2.csv) and exit
# Usage: nohup bash slurm/auto_run_target.sh 463203-2650 >> results/runs/463203-2650/auto_run.log 2>&1 &
T="${1:?target run dir name}"; TAG="${T##*-}"; ROOT=/global/scratch/users/sergiomar10/boltzaff; R="$ROOT/results/runs/$T"; cd "$ROOT"
PY=/clusterfs/nilah/sergio/miniconda3/envs/boltzba/bin/python; log(){ echo "[$(date '+%F %T')] $*"; }
source /clusterfs/nilah/sergio/miniconda3/etc/profile.d/conda.sh; conda activate boltzba; source "$ROOT/env.sh" "$T"; cd "$ROOT"
export R T; SUBMITTED=1; TOPPED=0; [ -f "$R/.topped_up" ] && TOPPED=1; LAST_MISSING=-1; STUCK=0
while true; do
  OUT=$($PY - <<'PY'
import os, glob
R = os.environ["R"]
need = set(open(f"{R}/ft_inputs_full/eval_ids_full.txt").read().split()) | set(open(f"{R}/ft_inputs_full/train_ids_top300.txt").read().split())
have = {os.path.basename(p)[len("affinity_embeddings_"):-4] for p in glob.glob(f"{R}/outputs_affcache/chunk_*/boltz_results_*/predictions/*/affinity_embeddings_*.npz")}
print(len(need - have))
PY
)
  MISSING=$(echo "$OUT" | tail -1); LIVE=$(squeue -u "$USER" -h -o %j | grep -c "^p2_$TAG$")
  log "missing Pass-2 results: $MISSING; p2_$TAG jobs queued: $LIVE"
  if [ "$MISSING" != "0" ]; then
    if [ "$LIVE" = "0" ]; then
      if [ "$MISSING" = "$LAST_MISSING" ]; then STUCK=$((STUCK+1)); else STUCK=0; fi; LAST_MISSING=$MISSING
      log "planning the $MISSING missing compounds into small jobs (round without progress: $STUCK)"
      PLAN=$(MISS_SIZE=20 $PY - <<'PY'
import os, glob, shutil, subprocess
from pathlib import Path
R = Path(os.environ["R"]); T = R / "p2_tail"; INP = R / "inputs_chunks_full"
need = set(open(R / "ft_inputs_full/eval_ids_full.txt").read().split()) | set(open(R / "ft_inputs_full/train_ids_top300.txt").read().split())
have = {os.path.basename(p)[len("affinity_embeddings_"):-4] for p in glob.glob(str(R / "outputs_affcache/chunk_*/boltz_results_*/predictions/*/affinity_embeddings_*.npz"))}
chunk_of = {}
for ch in sorted(c for c in os.listdir(INP) if len(c) == 9 and c.startswith("chunk_")):
    for y in os.listdir(INP / ch):
        if y.endswith(".yaml"): chunk_of[y[:-5]] = ch[len("chunk_"):]
todo = sorted((i for i in need - have if i in chunk_of), key=lambda i: (chunk_of[i], i)); size = int(os.environ["MISS_SIZE"])
idx = len([d for d in os.listdir(T / "inputs") if d.startswith("mini_")]); n0 = idx; rows = []
for s in range(0, len(todo), size):
    d = T / "inputs" / f"mini_{idx:03d}"; shutil.rmtree(str(d), ignore_errors=True); d.mkdir(parents=True); shutil.rmtree(str(T / "outputs" / f"mini_{idx:03d}"), ignore_errors=True)
    for i in todo[s:s + size]:
        src = R / "inputs_full" / f"{i}.yaml"
        try: os.link(str(src), str(d / f"{i}.yaml"))
        except OSError: shutil.copy(str(src), str(d))
        rows.append(f"{chunk_of[i]}\t{i}")
    idx += 1
open(str(T / "missing.tsv"), "a").write("\n".join(rows) + ("\n" if rows else ""))
print(n0, idx - n0, len(todo))
PY
)
      read N0 NM NT <<< "$PLAN"; log "planned $NT compounds into $NM minis starting at mini_$N0"
      OFF=$N0; LEFT=$NM
      while [ "$LEFT" -gt 0 ]; do
        CNT=$(( LEFT > 1000 ? 1000 : LEFT ))
        sbatch --time=00:15:00 --job-name="p2_$TAG" --export=ALL,TARGET="$T",IDX_OFFSET=$OFF --array=1-$CNT%200 "$ROOT/slurm/pass2_tail.sbatch" | sed "s/^/[$(date +%T)] /"
        OFF=$((OFF+CNT)); LEFT=$((LEFT-CNT))
      done
    fi
  else
    if [ "$TOPPED" = "0" ]; then
      $PY - <<'PY'
import os
R = os.environ["R"]; F = f"{R}/ft_inputs_full"; full = open(f"{F}/eval_ids_full.txt").read().split(); cur = set(open(f"{F}/eval_ids.txt").read().split()); extra = [i for i in full if i not in cur]
k0 = len([c for c in os.listdir(f"{F}/eval_chunks") if c.startswith("chunk_")])
for j in range(0, len(extra), 1000): open(f"{F}/eval_chunks/chunk_{k0 + j // 1000:03d}.txt", "w").write("\n".join(extra[j:j + 1000]) + "\n")
open(f"{F}/eval_ids.txt", "w").write("\n".join(full) + "\n"); print("top-up: added", len(extra), "compounds as", (len(extra) + 999) // 1000, "extra eval chunks")
PY
      rm -f "$R/outputs_affcache/cache_index.json"; touch "$R/.topped_up"; TOPPED=1; log "evaluation set restored to the full definition; cache index cleared"
    fi
    if [ "$SUBMITTED" = "0" ]; then
      rm -f "$R/outputs_affcache/cache_index.json"; log "Pass-2 complete: cache index cleared, submitting base scoring and the 15 head-FT arms"
      sbatch --nice=0 --job-name="base_$TAG" --export=ALL,TARGET="$T" "$ROOT/slurm/score_base.sbatch" | sed "s/^/[$(date +%T)] /"
      sbatch --nice=0 --job-name="ft_$TAG" --export=ALL,TARGET="$T" --array=1-15 "$ROOT/slurm/train_score_seed.sbatch" | sed "s/^/[$(date +%T)] /"
      SUBMITTED=1
    else
      $PY "$ROOT/pipeline_local/watchdog.py" --submit 2>&1 | grep -i "$TAG" | sed "s/^/[$(date +%T)] /"
    fi
    if [ -s "$ROOT/results/analysis/$T/per_seed.csv" ] || [ -f "$ROOT/results/analysis/$T/table2.csv" ]; then log "evaluation exists, done"; break; fi
    DONE_ARMS=$($PY -c "import sys; sys.path.insert(0,'$ROOT/pipeline_local'); import bft_common as bc; print(len(bc.scored_arms('$T')))" 2>/dev/null)
    log "scored arms: $DONE_ARMS/16"
    if [ "$DONE_ARMS" = "16" ]; then
      $PY "$ROOT/pipeline_local/eval_headft_seeds.py" --targets "$T" --out-dir "$WORKDIR/analysis/$T" && log "DONE evaluation -> $WORKDIR/analysis/$T/table2.csv" || log "evaluation errored (check scoring completeness)"; break
    fi
  fi
  sleep 600
done

#!/bin/bash
# Take one target from "Pass-1 (nearly) done" to Pass-2 submitted, without the driver: relink tail folds, consolidate, build ft inputs,
# drop yamls that never folded, submit Pass-2. Usage: finish_target.sh <run dir name> <assay id>   (log: results/runs/<t>/finish.log)
set -u
T="$1"; TAG="$2"; ROOT=/global/scratch/users/sergiomar10/boltzaff; LOG="$ROOT/results/runs/$T/finish.log"; : > "$LOG"; log(){ echo "[$(date +%T)] $*" >> "$LOG"; }
source /clusterfs/nilah/sergio/miniconda3/etc/profile.d/conda.sh; conda activate boltzft; cd "$ROOT"; source env.sh "$T"; R="$ROOT/results/runs/$T"
python - "$R" >> "$LOG" 2>&1 <<'PY'
import os, sys
from pathlib import Path
R=Path(sys.argv[1])/"outputs_chunks_full"; n=0
for d in [d for d in os.listdir(R) if "_tail_" in d]:
    ck=d.split("_tail_")[0]; main=R/ck/f"boltz_results_{ck}"/"predictions"; main.mkdir(parents=True,exist_ok=True); have=set(os.listdir(main))
    for r in os.listdir(R/d):
        if not r.startswith("boltz_results_"): continue
        P=R/d/r/"predictions"
        for i in os.listdir(P):
            if os.path.exists(P/i/f"embeddings_{i}.npz"):
                dst=main/i
                if i in have and os.path.exists(dst/f"embeddings_{i}.npz"): continue
                if os.path.lexists(dst):
                    if os.path.islink(dst): os.unlink(dst)
                    else: continue
                os.symlink(os.path.realpath(P/i),dst); n+=1; have.add(i)
print("relinked",n)
PY
log "consolidating"
python "$BOLTZFT/pipeline/merge_consolidate.py" --outputs-root "$R/outputs_chunks_full" --inputs-dir "$R/inputs_full" --out-dir "$R/consolidated_full" >> "$LOG" 2>&1 && log "consolidated" || { log "CONSOLIDATE FAILED"; exit 1; }
python "$BOLTZFT/pipeline/build_ft_inputs_full.py" --target "$T" --results-csv "$RESULTS_CSV" --consolidated-dir "$R/consolidated_full" --out-dir "$R/ft_inputs_full" >> "$LOG" 2>&1
[ -s "$R/ft_inputs_full/eval_ids.txt" ] || { log "build_ft_inputs FAILED"; exit 1; }
log "ft inputs built: eval=$(wc -l < "$R/ft_inputs_full/eval_ids.txt")"
python - "$R" >> "$LOG" 2>&1 <<'PY'
import os, sys
from pathlib import Path
R=Path(sys.argv[1]); S=R/"consolidated_full/structures"; rm=0
for ck in os.listdir(R/"inputs_chunks_full"):
    if len(ck)!=9: continue
    for y in os.listdir(R/"inputs_chunks_full"/ck):
        if not os.path.exists(S/(y[:-5]+".npz")):
            p=R/"inputs_chunks_full"/ck/y
            if p.is_symlink(): os.unlink(p); rm+=1
print("removed yamls without a structure (never folded):",rm)
PY
J=$(sbatch --nice=0 --job-name="p2_$TAG" --export=ALL,TARGET="$T" --array=1-143%60 slurm/pass2_affcache.sbatch | awk '{print $4}'); log "Pass-2 submitted: $J"
echo DONE >> "$LOG"

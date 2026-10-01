#!/bin/bash
# Stopgap monitor: fill any missing Pass-1 chunks, then consolidate. (Dies at session end.)
source /clusterfs/nilah/sergio/miniconda3/etc/profile.d/conda.sh; conda activate boltzft
source /global/scratch/users/sergiomar10/boltzaff/env.sh 588689
missing() { python - <<'PY'
import subprocess,re
def sh(c): return subprocess.run(c,shell=True,capture_output=True,text=True).stdout
live=set()
for l in sh("squeue -u sergiomar10 -h -o '%i %j'").splitlines():
    p=l.split()
    if len(p)<2 or not p[1].startswith("bft_p1"): continue
    m=re.search(r'_\[?([0-9,\-]+)',p[0])
    if not m: continue
    for part in m.group(1).split(','):
        if '-' in part: a,b=part.split('-'); live.update(range(int(a),int(b)+1))
        elif part.isdigit(): live.add(int(part))
import os
R=os.environ["R"]
done=set()
for i in range(1,144):
    d=f"{R}/outputs_chunks_full/chunk_{i-1:03d}"
    n=len(subprocess.run(f"find -L {d} -name 'embeddings_*.npz' 2>/dev/null",shell=True,capture_output=True,text=True).stdout.split())
    if n>=345: done.add(i)   # ~complete
miss=sorted(set(range(1,144))-live-done)
print(",".join(map(str,miss)))
PY
}
while true; do
  if squeue -u sergiomar10 -h -n bft_p1,bft_p1_cal -o "%i" 2>/dev/null | grep -q .; then sleep 300; continue; fi
  M=$(missing)
  if [ -n "$M" ]; then
    echo "[mon] queue empty but chunks missing: $M -> resubmitting $(date)"
    sbatch --export=ALL,TARGET=588689 --array="$M"%40 --exclude=n0143.savio3,n0144.savio3 "$ROOT/slurm/pass1_embed.sbatch"
    sleep 300; continue
  fi
  break
done
echo "[mon] all 143 chunks present; consolidating $(date)"
python "$BOLTZFT/pipeline/merge_consolidate.py" --outputs-root "$R/outputs_chunks_full" --inputs-dir "$R/inputs_full" --out-dir "$R/consolidated_full"
python "$BOLTZFT/pipeline/build_ft_inputs_full.py" --target "$TARGET" --results-csv "$RESULTS_CSV" --consolidated-dir "$R/consolidated_full" --out-dir "$R/ft_inputs_full"
echo "[mon] DONE structures=$(find -L "$R/consolidated_full/structures" -name '*.npz'|wc -l) eval_ids=$(wc -l < "$R/ft_inputs_full/eval_ids.txt")"

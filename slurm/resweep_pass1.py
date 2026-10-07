#!/usr/bin/env python3
"""Relink finished Pass-1 tail folds into their chunks, list the chunks that are still incomplete and not queued, and (with --submit) submit one resweep
array for them now instead of waiting for the driver (which resubmits only after the previous array has fully ended, at nice 100 and <= 30 tasks).
    python slurm/resweep_pass1.py <run dir name> [--submit] [--conc 60]       e.g. 463203-2650
Uses the driver's own job name (p1_<assay id>), so the driver sees the new array as live and does not submit a duplicate."""
import argparse, os, re, subprocess, sys, glob
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument("target"); ap.add_argument("--submit", action="store_true"); ap.add_argument("--conc", type=int, default=60); a = ap.parse_args()
ROOT = Path("/global/scratch/users/sergiomar10/boltzaff"); R = ROOT / "results/runs" / a.target; TAG = a.target.split("-")[-1]; OUT = R / "outputs_chunks_full"
n = 0
for d in [d for d in os.listdir(OUT) if "_tail_" in d]:
    ck = d.split("_tail_")[0]; main = OUT / ck / f"boltz_results_{ck}" / "predictions"; main.mkdir(parents=True, exist_ok=True); have = set(os.listdir(main))
    for r in os.listdir(OUT / d):
        if not r.startswith("boltz_results_"): continue
        P = OUT / d / r / "predictions"
        if not P.is_dir(): continue
        for i in os.listdir(P):
            if os.path.exists(P / i / f"embeddings_{i}.npz"):
                dst = main / i
                if i in have and os.path.exists(dst / f"embeddings_{i}.npz"): continue
                if os.path.lexists(dst):
                    if os.path.islink(dst): os.unlink(dst)
                    else: continue
                os.symlink(os.path.realpath(P / i), dst); n += 1; have.add(i)
print("relinked", n)
N = sum(1 for _ in open(R / "chunks.tsv")); live = set()
for l in subprocess.run("squeue -u $USER -h -o '%i %j'", shell=True, capture_output=True, text=True).stdout.splitlines():
    p = l.split()
    if len(p) < 2 or p[1] != f"p1_{TAG}": continue
    m = re.search(r"_\[?([0-9,\-]+)", p[0])
    if m:
        for part in m.group(1).split(","):
            if "-" in part: x, y = part.split("-"); live.update(range(int(x), int(y) + 1))
            elif part.isdigit(): live.add(int(part))
miss = []; short = 0
for i in range(1, N + 1):
    c = f"chunk_{i-1:03d}"; exp = len(glob.glob(f"{R}/inputs_chunks_full/{c}/*.yaml")); thr = min(345, exp) if exp else 345
    pd_ = OUT / c / f"boltz_results_{c}" / "predictions"; k = len(os.listdir(pd_)) if pd_.is_dir() else 0
    if k < thr and i not in live: miss.append(i); short += thr - k
print(f"chunks incomplete and not queued: {len(miss)} (about {short} compounds short of the driver's threshold); live tasks: {len(live)}")
arr = ",".join(map(str, miss)); print("array:", arr)
if a.submit and miss:
    print(subprocess.run(["sbatch", "--nice=0", f"--job-name=p1_{TAG}", f"--export=ALL,TARGET={a.target}", f"--array={arr}%{a.conc}",
                          "--exclude=n0143.savio3,n0144.savio3,n0215.savio3,n0176.savio3", str(ROOT / "slurm/pass1_embed.sbatch")], capture_output=True, text=True))

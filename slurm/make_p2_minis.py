#!/usr/bin/env python3
"""Build small Pass-2 jobs (default 60 compounds) for every compound that is already folded (has a pre_affinity file in consolidated_full/structures)
and has no finished Pass-2 result yet, and optionally submit them as an array of slurm/pass2_tail.sbatch.
    python slurm/make_p2_minis.py <run dir name> [--size 60] [--submit] [--nice 50] [--conc 60]       e.g. 463203-2650
Can be run repeatedly while Pass-1 is still finishing: compounds already placed in a mini are skipped, new minis get the next free indices, only
the new index range is submitted. Run merge_consolidate.py first so newly folded compounds have their pre_affinity files.
Pass-2 only needs a compound's own pose, so it does not have to wait for the rest of its chunk or for the rest of the library."""
import argparse, os, shutil, subprocess
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument("target"); ap.add_argument("--size", type=int, default=60); ap.add_argument("--submit", action="store_true")
ap.add_argument("--nice", type=int, default=50); ap.add_argument("--conc", type=int, default=60); ap.add_argument("--redo", action="store_true", help="also re-plan compounds that sit in an old mini but still have no result"); a = ap.parse_args()
ROOT = Path(__file__).resolve().parents[1]; R = ROOT / "results/runs" / a.target; TAG = a.target.split("-")[-1]; T = R / "p2_tail"; INP = R / "inputs_chunks_full"
S = R / "consolidated_full/structures"; AFF = R / "outputs_affcache"
(T / "inputs").mkdir(parents=True, exist_ok=True); tsv = T / "missing.tsv"
planned = {}
if tsv.exists():
    for line in tsv.read_text().splitlines():
        if line.strip(): k, i = line.split("\t"); planned[i] = k
have_pre = {f[len("pre_affinity_"):-4] for f in os.listdir(S) if f.startswith("pre_affinity_")}
chunk_of = {}
for ch in sorted(c for c in os.listdir(INP) if len(c) == 9 and c.startswith("chunk_")):
    for y in os.listdir(INP / ch):
        if y.endswith(".yaml"): chunk_of[y[:-5]] = ch[len("chunk_"):]
def done(i, k):
    p = AFF / f"chunk_{k}/boltz_results_chunk_{k}/predictions/{i}/affinity_embeddings_{i}.npz"; return p.exists()
todo = [i for i in sorted(have_pre) if i in chunk_of and not done(i, chunk_of[i]) and (a.redo or i not in planned)]
by_chunk = {}
for i in todo: by_chunk.setdefault(chunk_of[i], []).append(i)
n0 = len([d for d in os.listdir(T / "inputs") if d.startswith("mini_")]); idx = n0; rows = []
for k in sorted(by_chunk):
    ids = by_chunk[k]
    for s in range(0, len(ids), a.size):
        d = T / "inputs" / f"mini_{idx:03d}"; shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True); shutil.rmtree(T / "outputs" / f"mini_{idx:03d}", ignore_errors=True)
        for i in ids[s:s + a.size]: shutil.copy(R / "inputs_full" / f"{i}.yaml", d); rows.append(f"{k}\t{i}")
        idx += 1
with open(tsv, "a") as f: f.write("\n".join(rows) + ("\n" if rows else ""))
print(f"{len(have_pre):,} folded compounds with pre_affinity; {len(todo):,} need Pass-2 and are now in minis {n0 + 1}..{idx} ({idx - n0} jobs of up to {a.size})")
if a.submit and idx > n0:
    off = n0 if idx > 1000 else 0                      # the cluster's array indices stop at 1000: count from 1 and let the job script add the offset
    cmd = ["sbatch", f"--nice={a.nice}", f"--job-name=p2m_{TAG}", f"--export=ALL,TARGET={a.target},IDX_OFFSET={off}", f"--array={n0 + 1 - off}-{idx - off}%{a.conc}", str(ROOT / "slurm/pass2_tail.sbatch")]
    print(subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT).stdout.strip())

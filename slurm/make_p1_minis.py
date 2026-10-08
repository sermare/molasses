#!/usr/bin/env python3
"""Split a target's still-unfolded Pass-1 compounds into small chunk-pure jobs (default 40 compounds, about 17 GPU-minutes each).
    python slurm/make_p1_minis.py <run dir name> [--size 40] [--submit]        e.g. 463203-2650
1. relinks finished tail folds into their chunks, 2. lists the compounds without an embeddings file, per chunk, 3. writes
results/runs/<t>/p1_mini/index.tsv (idx, target, chunk, list file) and one list per job, 4. with --submit submits slurm/pass1_mini.sbatch as an array.
Small jobs backfill easily on a busy low-priority queue and lose almost nothing when pre-empted. Never run two of these at once for one target."""
import argparse, os, subprocess
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument("target"); ap.add_argument("--size", type=int, default=40); ap.add_argument("--submit", action="store_true"); ap.add_argument("--conc", type=int, default=0); ap.add_argument("--no-relink", action="store_true"); a = ap.parse_args()
from concurrent.futures import ThreadPoolExecutor
ROOT = Path(__file__).resolve().parents[1]; R = ROOT / "results/runs" / a.target; OUT = R / "outputs_chunks_full"; INP = R / "inputs_chunks_full"; TAG = a.target.split("-")[-1]
n = 0
for d in ([] if a.no_relink else [d for d in os.listdir(OUT) if "_tail_" in d]):                      # relink finished tail folds (same rule as resweep_pass1.py)
    ck = d.split("_tail_")[0]; main = OUT / ck / f"boltz_results_{ck}" / "predictions"; main.mkdir(parents=True, exist_ok=True); have = set(os.listdir(main))
    for r in os.listdir(OUT / d):
        P = OUT / d / r / "predictions"
        if not r.startswith("boltz_results_") or not P.is_dir(): continue
        for i in os.listdir(P):
            if os.path.exists(P / i / f"embeddings_{i}.npz"):
                dst = main / i
                if i in have and os.path.exists(dst / f"embeddings_{i}.npz"): continue
                if os.path.lexists(dst):
                    if os.path.islink(dst): os.unlink(dst)
                    else: continue
                os.symlink(os.path.realpath(P / i), dst); n += 1; have.add(i)
print("relinked", n)
M = R / "p1_mini"; (M / "lists").mkdir(parents=True, exist_ok=True)
for f in (M / "lists").glob("*.txt"): f.unlink()
rows = []; total = 0; idx = 0
chunks = sorted(c for c in os.listdir(INP) if len(c) == 9 and c.startswith("chunk_"))
def missing(ch):       # slow shared filesystem: one listing per chunk, checks run in parallel threads
    pred = OUT / ch / f"boltz_results_{ch}" / "predictions"; ids = sorted(y[:-5] for y in os.listdir(INP / ch) if y.endswith(".yaml"))
    done = set(os.listdir(pred)) if pred.is_dir() else set()
    return ch, [i for i in ids if i not in done or not (pred / i / f"embeddings_{i}.npz").exists()]
with ThreadPoolExecutor(max_workers=32) as ex: per_chunk = list(ex.map(missing, chunks))
for ch, ids in per_chunk:
    total += len(ids)
    for k in range(0, len(ids), a.size):
        idx += 1; lf = M / "lists" / f"mini_{idx:04d}.txt"; lf.write_text("\n".join(ids[k:k + a.size]) + "\n"); rows.append(f"{idx}\t{a.target}\t{ch}\t{lf}")
(M / "index.tsv").write_text("\n".join(rows) + "\n")
print(f"{total} compounds still unfolded in {len([r for r in rows])} jobs of up to {a.size}")
if a.submit and rows:
    cmd = ["sbatch", "--nice=0", f"--job-name=p1m_{TAG}", f"--export=ALL,TARGET={a.target}", f"--array=1-{len(rows)}" + (f"%{a.conc}" if a.conc else ""), str(ROOT / "slurm/pass1_mini.sbatch")]
    print(subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT).stdout.strip())

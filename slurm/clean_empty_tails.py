#!/usr/bin/env python3
"""Move Pass-1 tail folders that hold nothing usable (no processed/manifest.json and no folded compound) out of outputs_chunks_full/, into
outputs_chunks_full/_empty_tails/ (not matched by merge_consolidate's chunk_* glob). Cancelled or pre-empted jobs leave these behind and they crash the
consolidation. Folders with any finished embeddings are never touched, because chunk predictions link into them.
    python slurm/clean_empty_tails.py <run dir name> [--dry-run]"""
import argparse, os, shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument("target"); ap.add_argument("--dry-run", action="store_true"); a = ap.parse_args()
OUT = Path(__file__).resolve().parents[1] / "results/runs" / a.target / "outputs_chunks_full"; DEST = OUT / "_empty_tails"
def check(d):
    res = [r for r in os.listdir(OUT / d) if r.startswith("boltz_results_")]
    if not res: return d, True
    for r in res:
        P = OUT / d / r
        if (P / "processed/manifest.json").exists(): continue
        pred = P / "predictions"
        has = pred.is_dir() and any((pred / i / f"embeddings_{i}.npz").exists() for i in os.listdir(pred))
        if has: return d, False
        return d, True                       # no manifest and nothing folded
    return d, False
tails = [d for d in os.listdir(OUT) if "_tail_" in d and d != "_empty_tails"]
with ThreadPoolExecutor(max_workers=32) as ex: bad = [d for d, b in ex.map(check, tails) if b]
print(f"{len(tails)} tail folders, {len(bad)} empty (no manifest, nothing folded)")
if not a.dry_run:
    DEST.mkdir(exist_ok=True)
    for d in bad: shutil.move(str(OUT / d), str(DEST / d))
    print("moved to", DEST)

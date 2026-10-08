#!/usr/bin/env python3
"""Same result as BoltzFT/pipeline/merge_consolidate.py (same arguments, same layout, same rule that later folders override earlier ones), but the
directory listings and symlink creation run in parallel threads. On this shared filesystem the original needs about two hours per target, this about
10 minutes. Safe to re-run; links that already point to the right file are left alone.
    python slurm/consolidate_parallel.py --outputs-root R/outputs_chunks_full --inputs-dir R/inputs_full --out-dir R/consolidated_full"""
import argparse, json, os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument("--outputs-root", type=Path, required=True); ap.add_argument("--inputs-dir", type=Path, required=True)
ap.add_argument("--out-dir", type=Path, required=True); ap.add_argument("--threads", type=int, default=32); a = ap.parse_args()
dirs = sorted(a.outputs_root.glob("chunk_*/boltz_results_*"))
if not dirs: raise SystemExit(f"no boltz_results_* under {a.outputs_root}/chunk_*/")
OUT = {k: a.out_dir / k for k in ("embeddings", "structures", "records", "mols", "msa")}
for d in OUT.values(): d.mkdir(parents=True, exist_ok=True)
def scan(rd):
    P, Q = rd / "predictions", rd / "processed"
    ent = []                                                    # (kind, dst name, src)
    for p in P.glob("*/embeddings_*.npz"): ent.append(("embeddings", p.name, p))
    for p in (Q / "structures").glob("*.npz"): ent.append(("structures", p.name, p))
    for p in P.glob("*/pre_affinity_*.npz"): ent.append(("structures", p.name, p))
    for p in (Q / "records").glob("*.json"): ent.append(("records", p.name, p))
    for p in (Q / "mols").glob("*.pkl"): ent.append(("mols", p.name, p))
    for p in (Q / "msa").glob("*.npz"): ent.append(("msa", p.name, p))
    man = json.loads((Q / "manifest.json").read_text()); recs = man["records"] if isinstance(man, dict) else man
    return ent, recs
with ThreadPoolExecutor(max_workers=a.threads) as ex: scanned = list(ex.map(scan, dirs))      # map keeps the sorted order
want = {}; records = []; n = dict(embeddings=0, structures=0, msa=0)
for ent, recs in scanned:                                        # later folders override earlier ones, as in merge_consolidate.py
    for kind, name, src in ent:
        want[(kind, name)] = src
        if kind in n and (kind != "structures" or not name.startswith("pre_affinity_")): n[kind] += 1
    records.extend(recs)
stats = dict(created=0, updated=0, unchanged=0)
def make(item):
    (kind, name), src = item; dst = OUT[kind] / name; real = os.path.realpath(src)
    if os.path.islink(dst):
        if os.readlink(dst) == real: return "unchanged"
        os.unlink(dst); os.symlink(real, dst); return "updated"
    if os.path.exists(dst): os.unlink(dst)
    os.symlink(real, dst); return "created"
with ThreadPoolExecutor(max_workers=a.threads) as ex:
    for r in ex.map(make, list(want.items()), chunksize=64): stats[r] += 1
seen, uniq = set(), []
for r in records:
    rid = str(r.get("id"))
    if rid not in seen: seen.add(rid); uniq.append(r)
(a.out_dir / "manifest.json").write_text(json.dumps({"records": uniq}))
y = a.out_dir / "yamls"
if y.is_symlink() or y.exists(): y.unlink()
os.symlink(a.inputs_dir.resolve(), y)
print(f"merged {len(dirs)} result folders -> {a.out_dir}\\nembeddings={n['embeddings']} structures={n['structures']} msa={n['msa']} manifest_records={len(uniq)}\\nlinks: {stats}")

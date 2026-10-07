#!/usr/bin/env python3
"""GPU-hours used by this project, from Slurm accounting (sacct): by stage (job-name prefix), by target (assay-id suffix), by final state and by GPU type (node).
    python pipeline_local/gpu_hours.py [--since 2026-09-20]  -> prints tables, writes results/analysis/gpu_hours_*.csv
GPU-hours = ElapsedRaw x gpus allocated / 3600 for every job step-less record (-X); includes failed, pre-empted and cancelled jobs (they used the GPU too)."""
import argparse, subprocess, io, re, os
import pandas as pd
ap = argparse.ArgumentParser(); ap.add_argument("--since", default="2026-09-20"); a = ap.parse_args()
out = subprocess.run(["sacct", "-P", "-n", "-X", "-S", a.since, "-u", os.environ["USER"], "-o", "JobIDRaw,JobName,State,ElapsedRaw,AllocTRES,NodeList,Start"], capture_output=True, text=True).stdout
d = pd.read_csv(io.StringIO(out), sep="|", header=None, names=["id", "name", "state", "sec", "tres", "node", "start"])
d["gpus"] = d.tres.fillna("").str.extract(r"gres/gpu=(\d+)")[0].astype(float).fillna(0)
d = d[(d.gpus > 0) & d.sec.notna()].copy(); d["gh"] = d.sec.astype(float) * d.gpus / 3600
d["state"] = d.state.str.split().str[0]
nodes = subprocess.run("sinfo -N -h -p savio3_gpu -o '%N %G'", shell=True, capture_output=True, text=True).stdout.split("\n")
gtype = {l.split()[0]: (l.split()[1].split(":")[1] if ":" in l.split()[1] else l.split()[1]) for l in nodes if l.strip()}
d["gpu_type"] = d.node.map(lambda n: gtype.get(str(n).split(",")[0].split("[")[0], "unknown"))
TAGS = {"588689": "588689", "504329": "504329", "743445": "743445", "485317": "485317", "2097": "2097", "493091": "493091", "2650": "2650", "588549": "588549"}
def stage(n):
    n = str(n)
    for p, s in (("p1", "Pass-1 folding"), ("p2", "Pass-2 affinity cache"), ("base", "base scoring"), ("ft", "head-FT train+score"), ("chunk", "single-chunk scoring"), ("bft_p1", "Pass-1 folding"), ("bft_p2", "Pass-2 affinity cache")):
        if n == p or n.startswith(p + "_"): return s
    return "other (" + n + ")" if n not in ("OOD_VSCode",) else "interactive session"
def target(n):
    m = re.search(r"_(\d+)$", str(n)); return m.group(1) if m and m.group(1) in TAGS else "untagged"
d["stage"] = d.name.map(stage); d["target"] = d.name.map(target)
print(f"jobs with a GPU: {len(d):,}; total {d.gh.sum():,.0f} GPU-hours since {a.since}\n")
for col in ("stage", "state", "gpu_type"):
    t = d.groupby(col).gh.agg(["sum", "count"]).sort_values("sum", ascending=False).round(0); t.columns = ["GPU-hours", "jobs"]; print(t.to_string(), "\n"); t.to_csv(f"results/analysis/gpu_hours_{col}.csv")
t = d.pivot_table(index="target", columns="stage", values="gh", aggfunc="sum", fill_value=0).round(0); t["total"] = t.sum(axis=1); print(t.to_string()); t.to_csv("results/analysis/gpu_hours_by_target.csv")
w = d.groupby("state").gh.sum(); lost = w.drop(["COMPLETED"], errors="ignore").sum(); print(f"\nGPU-hours in jobs that did not end COMPLETED: {lost:,.0f} ({100*lost/d.gh.sum():.0f}%) -- includes pre-empted, failed, cancelled and timed-out jobs")
print("\nGPU types on savio3_gpu (sinfo):", sorted(set(gtype.values())))

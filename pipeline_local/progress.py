#!/usr/bin/env python3
"""Per-target progress of the BoltzFT scale-out: Pass-1 folding, Pass-2 affinity cache, training/scoring arms, evaluation, and what is queued.

    python pipeline_local/progress.py              # print the table
    python pipeline_local/progress.py --save       # also write results/analysis/progress.csv and progress.md
    python pipeline_local/progress.py --readme     # also rewrite the block between <!-- PROGRESS:START --> and <!-- PROGRESS:END --> in README.md
                                                   # (adds the block before '## Directory layout' if the markers are missing)
Counts come from the files on disk (unique folded compounds, cached affinity folders, scored chunks), the queue from squeue. Takes about a minute
because the filesystem is slow; directory listings run in parallel."""
import argparse, os, re, subprocess, sys, datetime
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, "/global/scratch/users/sergiomar10/boltzaff/pipeline_local")
import pandas as pd
import bft_common as bc

ROOT = bc.ROOT; RUNS = bc.RUNS; AN = bc.AN


def list_dirs(path):
    try: return os.listdir(path)
    except OSError: return []


def pass1_ids(t):
    """Unique compound ids with a Pass-1 prediction folder, over every chunk folder (tail-fold folders included, so no double counting)."""
    base = RUNS / t / "outputs_chunks_full"; ids = set()
    for c in list_dirs(base):
        if not c.startswith("chunk_"): continue
        for r in list_dirs(base / c):
            if r.startswith("boltz_results_"): ids.update(list_dirs(base / c / r / "predictions"))
    return ids


def pass2_count(t):
    """Number of compounds with a FINISHED Pass-2 result (affinity_embeddings_<id>.npz present), not just a prediction folder: a folder exists
    before its files are written, so counting folders overstated progress (2097 showed 100% at 67%)."""
    base = RUNS / t / "outputs_affcache"; chunks = [c for c in list_dirs(base) if len(c) == 9 and c.startswith("chunk_")]
    def one(c):
        n = 0
        for r in list_dirs(base / c):
            if r.startswith("boltz_results_"):
                P = base / c / r / "predictions"
                n += sum(1 for i in list_dirs(P) if os.path.exists(P / i / f"affinity_embeddings_{i}.npz"))
        return n
    with ThreadPoolExecutor(max_workers=16) as ex: return sum(ex.map(one, chunks))


def queue():
    """{(stage, tag): [running, pending]} from squeue, stage in p1, p2, base, ft."""
    out = subprocess.run(["squeue", "-u", os.environ.get("USER", ""), "-h", "-r", "-o", "%j|%T|%r"], capture_output=True, text=True).stdout
    q = {}
    for line in out.splitlines():
        name, state, reason = line.split("|")
        m = re.match(r"^(p1m?|p2|base|ft)_(.+)$", name)
        if not m: continue
        k = (m.group(1).rstrip("m") if m.group(1) == "p1m" else m.group(1), m.group(2)); q.setdefault(k, [0, 0, 0])
        if state == "RUNNING": q[k][0] += 1
        elif "Held" in reason: q[k][2] += 1          # held by the user (paused on purpose)
        else: q[k][1] += 1
    gpus = sum(1 for l in subprocess.run(["squeue", "-u", os.environ.get("USER", ""), "-h", "-t", "R", "-o", "%b"], capture_output=True, text=True).stdout.splitlines() if "gpu" in l)
    return q, gpus


def stage_of(r):
    if r["table2"]: return "done"
    if r["arms"] == 16: return "scored, evaluating"
    if r["arms"] > 0: return f"scoring ({r['arms']}/16 arms complete)"
    if r["ft_inputs"] and r["p2"] >= 0.99 * r["library"]: return "Pass-2 done, training/scoring next"
    if r["ft_inputs"]: return "Pass-2 running"
    if r["p1"] >= 0.99 * r["library"]: return "Pass-1 done, Pass-2 next"
    if r["p1"] > 0: return "Pass-1 running"
    return "not started"


def collect():
    with ThreadPoolExecutor(max_workers=16) as ex:
        f1 = {t: ex.submit(pass1_ids, t) for t in bc.TARGETS}; f2 = {t: ex.submit(pass2_count, t) for t in bc.TARGETS}
        p1 = {t: len(f1[t].result()) for t in bc.TARGETS}; p2 = {t: f2[t].result() for t in bc.TARGETS}
    q, gpus = queue(); rows = []
    for t in bc.TARGETS:
        tag = t.split("-")[-1]; lib = len(list_dirs(RUNS / t / "inputs_full"))
        r = dict(target=bc.SHORT[t], library=lib, p1=p1[t], p2=p2[t], ft_inputs=(RUNS / t / "ft_inputs_full/eval_ids.txt").exists(), arms=len(bc.scored_arms(t)), table2=(AN / t / "table2.csv").exists())
        run = sum(q.get((s, tag), [0, 0, 0])[0] for s in ("p1", "p2", "base", "ft")); pend = sum(q.get((s, tag), [0, 0, 0])[1] for s in ("p1", "p2", "base", "ft")); held = sum(q.get((s, tag), [0, 0, 0])[2] for s in ("p1", "p2", "base", "ft"))
        r["running"], r["pending"], r["held"] = run, pend, held; r["stage"] = stage_of(r)
        if r["stage"] != "done" and run == 0:        # say so when nothing is on a GPU
            r["stage"] += ", PAUSED (held)" if held and not pend else (", waiting for GPUs" if pend else ", no tasks queued (driver resubmits)")
        rows.append(r)
    return pd.DataFrame(rows), gpus


def fmt(df):
    pct = lambda n, d: f"{n:,} ({100 * n / d:.0f}%)" if d else f"{n:,}"
    out = pd.DataFrame({"Target": df.target, "Library": df.library.map("{:,}".format), "Pass-1 folded": [pct(a, b) for a, b in zip(df.p1, df.library)],
                        "Pass-2 cached": [pct(a, b) for a, b in zip(df.p2, df.library)], "Scored arms": df.arms.map(lambda k: f"{k}/16"),
                        "Table 2": df.table2.map({True: "yes", False: "no"}), "Stage": df.stage, "Tasks running / pending (held)": [f"{a} / {b}" + (f" ({c} held)" if c else "") for a, b, c in zip(df.running, df.pending, df.held)]})
    return out


def to_md(df):
    cols = list(df.columns); lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows(): lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--save", action="store_true"); ap.add_argument("--readme", action="store_true"); a = ap.parse_args()
    raw, gpus = collect(); tbl = fmt(raw); now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    done = int(raw.table2.sum()); stamp = f"Pipeline progress as of {now}: {done} of {len(raw)} targets finished; {gpus} GPUs running."
    print(stamp); print(tbl.to_string(index=False))
    if a.save:
        raw.to_csv(AN / "progress.csv", index=False); (AN / "progress.md").write_text(stamp + "\n\n" + to_md(tbl) + "\n"); print("saved results/analysis/progress.csv and progress.md")
    if a.readme:
        p = ROOT / "README.md"; s = p.read_text(); block = f"<!-- PROGRESS:START -->\n{stamp}\n\n{to_md(tbl)}\n\nRegenerate with `python pipeline_local/progress.py --readme`.\n<!-- PROGRESS:END -->"
        if "<!-- PROGRESS:START -->" in s: s = re.sub(r"<!-- PROGRESS:START -->.*?<!-- PROGRESS:END -->", lambda m: block, s, flags=re.S)
        else: s = s.replace("## Directory layout", "## Pipeline progress\n\n" + block + "\n\n---\n\n## Directory layout", 1)
        p.write_text(s); print("README.md updated")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Resubmit scoring arms that stopped before finishing (timeout, pre-emption, cancelled driver) so nothing sits idle.

    python pipeline_local/watchdog.py            # report what is incomplete and what would be resubmitted
    python pipeline_local/watchdog.py --submit   # resubmit them (training/scoring skips finished checkpoints and chunks)

For every target that has started scoring (some arm has score files) but is not finished, it finds the arms (base, and the 15 head-FT arms
top{40,100,300}_seed{0..4}) with fewer than all eval chunks that have no running or pending job, and resubmits them with the stock job
templates (12 h limit; scoring resumes from the chunks already written). Run it every few hours, or put it in a loop. It never touches finished arms."""
import argparse, os, subprocess, sys
sys.path.insert(0, "/global/scratch/users/sergiomar10/boltzaff/pipeline_local")
import bft_common as bc

ROOT = bc.ROOT; ARMS = [(n, s) for n in bc.BUDGETS for s in bc.SEEDS]      # array task i+1 = ARMS[i], as in train_score_seed.sbatch


def queued_names():
    out = subprocess.run(["squeue", "-u", os.environ["USER"], "-h", "-o", "%j|%i|%T"], capture_output=True, text=True).stdout
    return [l.split("|") for l in out.splitlines()]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--submit", action="store_true"); a = ap.parse_args(); Q = queued_names(); acted = 0
    for t in bc.TARGETS:
        k = bc.n_eval_chunks(t)
        if not k or (bc.AN / t / "table2.csv").exists(): continue
        tag = t.split("-")[-1]; started = bc.scored_arms(t) or any((bc.RUNS / t / "headft_affcache").glob("*/scores/chunk_*.csv"))
        if not started: continue
        have = {"base": False}; todo = []
        base_done = bc.arm_scored(t, "base"); live = lambda prefix: any(n == f"{prefix}_{tag}" for n, _, _ in Q)
        if not base_done and not live("base"): todo.append(("base", None))
        missing = [i + 1 for i, (n, s) in enumerate(ARMS) if not bc.arm_scored(t, f"top{n}_seed{s}")]
        # a head-FT array task is considered live if the ft_<tag> job has any running/pending entry for that task id
        liveids = set()
        for n, jid, state in Q:
            if n == f"ft_{tag}" and "_" in jid:
                part = jid.split("_", 1)[1].strip("[]").split("%")[0]
                for p in part.split(","):
                    if "-" in p: x, y = p.split("-"); liveids.update(range(int(x), int(y) + 1))
                    elif p.isdigit(): liveids.add(int(p))
        todo_ft = [m for m in missing if m not in liveids]
        if live("ft") and not liveids: todo_ft = []            # array listed without task ids: do not guess
        print(f"{bc.SHORT[t]}: base {'done' if base_done else 'incomplete'}; head-FT arms incomplete: {len(missing)}; with no live job: {todo_ft}")
        if a.submit:
            if ("base", None) in todo:
                print(subprocess.run(["sbatch", "--nice=0", f"--job-name=base_{tag}", f"--export=ALL,TARGET={t}", str(ROOT / "slurm/score_base.sbatch")], capture_output=True, text=True).stdout.strip()); acted += 1
            if todo_ft:
                arr = ",".join(map(str, todo_ft))
                print(subprocess.run(["sbatch", "--nice=0", f"--job-name=ft_{tag}", f"--export=ALL,TARGET={t}", f"--array={arr}", str(ROOT / "slurm/train_score_seed.sbatch")], capture_output=True, text=True).stdout.strip()); acted += 1
    print("nothing to resubmit" if not acted and a.submit else ("(dry run: pass --submit to resubmit)" if not a.submit else f"{acted} submission(s)"))


if __name__ == "__main__":
    main()

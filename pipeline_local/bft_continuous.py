#!/usr/bin/env python3
"""Do the Boltz-2 scores (native No-FT and head-FT) track the continuous assay readouts, not just the binary label?
    python pipeline_local/bft_continuous.py [--targets ...]   -> results/analysis/continuous_correlation.csv

Reference: the binary label itself (point-biserial rank correlation). Readouts, from data/mf-pcba_test/<target>.csv: `SD` and `SD Z-score` (primary-screen signal, all compounds) and `DR` (dose-response value, only the few hundred
compounds that were followed up; mostly actives). Spearman rank correlation of each model's score with each readout, on the evaluation compounds (the top 300 by
Boltz rank are excluded from evaluation), over (a) all evaluation compounds, (b) actives only. Head-FT score = mean over the 5 seeds. 95% bootstrap interval
(over compounds) for the head-FT minus No-FT difference. Sign convention is printed: the readout is oriented so that larger = more active."""
import argparse, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/global/scratch/users/sergiomar10/boltzaff/pipeline_local")
import numpy as np, pandas as pd
from scipy.stats import rankdata
import bft_common as bc

READ = ["SD", "DR"]   # SD Z-score is a linear rescale of SD (identical rank correlations), so it is not repeated


def rho(a, b):
    ra, rb = rankdata(a), rankdata(b); ra -= ra.mean(); rb -= rb.mean()
    return float((ra * rb).sum() / np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))


def boot_diff(x, y1, y0, B=300, seed=0):
    rng = np.random.default_rng(seed); n = len(x); d = []
    for _ in range(B):
        i = rng.integers(0, n, n); d.append(rho(x[i], y1[i]) - rho(x[i], y0[i]))
    return np.percentile(d, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--targets", nargs="*"); a = ap.parse_args()
    targets = a.targets or bc.done_targets("scores"); rows = []
    for t in targets:
        S = bc.scores(t).copy(); S["CID"] = S["id"].str.split("_").str[-1].astype(int)
        d = pd.read_csv(bc.ROOT / "data/mf-pcba_test" / f"{t}.csv")[["CID", "target_active_v2"] + READ]
        S = S.merge(d[["CID"] + READ], on="CID", how="left")
        models = {"No-FT": S.noft_p.values}
        for N in bc.BUDGETS: models[f"head-FT N={N}"] = np.mean([S[f"ft{N}_p{s}"].values for s in bc.SEEDS], axis=0)
        y = S.label.values.astype(int); S["binary label"] = y
        for r in ["binary label"] + READ:
            v = S[r].values.astype(float)
            if r == "SD":
                med = {c: np.nanmedian(v[y == c]) for c in (0, 1)}; sign = 1.0 if med[1] >= med[0] else -1.0   # orient: larger = more active
            else: sign = 1.0     # binary label as is; DR exists only for actives (no class contrast), taken as given (larger value = larger DR)
            v = sign * v
            for sub, mask in (("all evaluation compounds", np.isfinite(v)), ("actives only", np.isfinite(v) & (y == 1))):
                if mask.sum() < 30: continue
                for name, sc in models.items():
                    rows.append(dict(target=bc.SHORT[t], readout=r, subset=sub, model=name, n=int(mask.sum()), spearman=round(rho(sc[mask], v[mask]), 3), sign=int(sign)))
                lo, hi = boot_diff(v[mask], models["head-FT N=300"][mask], models["No-FT"][mask])
                rows.append(dict(target=bc.SHORT[t], readout=r, subset=sub, model="N=300 minus No-FT", n=int(mask.sum()),
                                 spearman=round(rho(models["head-FT N=300"][mask], v[mask]) - rho(models["No-FT"][mask], v[mask]), 3), ci_lo=round(lo, 3), ci_hi=round(hi, 3), sign=int(sign)))
        print(bc.SHORT[t], "done", flush=True)
    D = pd.DataFrame(rows); D.to_csv(bc.AN / "continuous_correlation.csv", index=False)
    for (r, sub), g in D.groupby(["readout", "subset"], sort=False):
        print(f"\n== {r} | {sub}"); print(g.pivot(index="target", columns="model", values="spearman").to_string())


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""No-FT vs head-FT on near-identical pairs, evaluated against GROUND TRUTH (not against the models' own output).

Ground truth
  positives : CLIFF pairs (active + near-identical inactive) = true large delta in the label (the label is binary,
              so there is no magnitude; no screen-readout or potency values are used).
  negatives : CONSERVED pairs (both active) = true absence of a large delta (orientation random).
Fair comparison (scale-free): each model's threshold for a 'large' delta is calibrated on the NEGATIVES (its own noise
floor at 90% / 95% specificity); sensitivity = share of true cliffs whose delta clears it in the RIGHT direction.
Also: pairwise accuracy on cliffs, AUROC of |delta| and of the signed delta (cliff vs conserved).
Deltas are computed on two scales (percentile rank and log-odds) so the result does not hinge on one transformation.
CIs resample independent chemical series (connected clusters of pairs), paired across the two models.
Depends on notebooks/14_structural_cliffs.ipynb cell 1 for the pair table (same orientation/series definitions)."""
import warnings; warnings.filterwarnings("ignore")
import sys, nbformat, numpy as np, pandas as pd
from scipy import stats
from scipy.stats import rankdata
B = int(sys.argv[1]) if len(sys.argv) > 1 else 600
nb = nbformat.read("/global/scratch/users/sergiomar10/boltzaff/notebooks/14_structural_cliffs.ipynb", as_version=4)
exec(nb.cells[1].source)                                   # P, Mi, ZS, BIG, big, series ...
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
MODELS = {"No-FT": "base_noft", "head-FT": "ft300"}
for m in MODELS.values():
    P["l_" + m] = lg(Mi[m].reindex(P.id_a).values) - lg(Mi[m].reindex(P.id_b).values)
COLS = {(mn, sc): (("d_" if sc == "percentile" else "l_") + m) for mn, m in MODELS.items() for sc in ("percentile", "log-odds")}

def auc(pos, neg):
    r = rankdata(np.concatenate([pos, neg])); n1, n2 = len(pos), len(neg)
    return (r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n2)

def metrics(df, col):
    c = df.loc[df.kind == "cliff", col].values; k = df.loc[df.kind == "conserved", col].values; o = {}
    for sp in (90, 95):
        tau = np.quantile(np.abs(k), sp / 100); o[f"sens@{sp}"] = np.mean(c > tau); o[f"wrongdir@{sp}"] = np.mean(c < -tau)
    o["auc_abs"] = auc(np.abs(c), np.abs(k)); o["auc_signed"] = auc(c, k)
    o["pairwise_acc"] = np.mean((c > 0) + 0.5 * (c == 0))          # share of true cliffs where the active is ranked higher
    return o

def boot(df, B, seed=0):
    g = list(df.groupby("series").indices.values()); rng = np.random.default_rng(seed); res = []
    for _ in range(B):
        idx = np.concatenate([g[i] for i in rng.integers(0, len(g), len(g))]); d = df.iloc[idx]
        if (d.kind == "cliff").sum() < 5 or (d.kind == "conserved").sum() < 5: continue
        res.append({k: metrics(d, c) for k, c in COLS.items()})
    return res

PARTS = [("all pairs", P), ("without the dominant series", P[~P.big]), ("dominant series only", P[P.big])]
KEY = ["pairwise_acc", "sens@90", "sens@95", "wrongdir@90", "auc_abs", "auc_signed"]
ROWS = []
for nm, df in PARTS:
    C, K = df[df.kind == "cliff"], df[df.kind == "conserved"]
    print(f"\n################ {nm}: {len(C)} true-large-delta pairs (cliffs), {len(K)} true-no-delta pairs (conserved), {df.series.nunique()} series")
    pt = {k: metrics(df, c) for k, c in COLS.items()}; bs = boot(df, B)
    for sc in ("percentile", "log-odds"):
        print(f"  --- scale: {sc}")
        print(f"  {'metric':<16}{'No-FT':>9}{'head-FT':>9}   head-FT minus No-FT [95% CI over series]")
        for key in KEY:
            a = pt[("No-FT", sc)].get(key, np.nan); b = pt[("head-FT", sc)].get(key, np.nan)
            d = [r[("head-FT", sc)].get(key, np.nan) - r[("No-FT", sc)].get(key, np.nan) for r in bs]; d = np.array([x for x in d if np.isfinite(x)])
            ci = f"{b-a:+.3f} [{np.percentile(d,2.5):+.3f}, {np.percentile(d,97.5):+.3f}]" if len(d) > 50 else "n/a"
            print(f"  {key:<16}{a:>9.3f}{b:>9.3f}   {ci}")
            ROWS.append(dict(subset=nm, scale=sc, metric=key, noft=a, ft=b, diff=b-a, lo=np.percentile(d,2.5) if len(d)>50 else np.nan, hi=np.percentile(d,97.5) if len(d)>50 else np.nan,
                             n_cliff=len(C), n_conserved=len(K), n_series=df.series.nunique()))

pd.DataFrame(ROWS).to_csv("/global/scratch/users/sergiomar10/boltzaff/results/analysis/588689/cliffs/ft_vs_noft_calibrated.csv", index=False); print("saved ft_vs_noft_calibrated.csv")

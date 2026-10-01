"""Helpers for 00_overview / 01_replication: per-(arm, seed) metrics computed from bft_common.scores(t), cached per target."""
import importlib.util, sys, warnings
warnings.filterwarnings("ignore", message="Sample size too small")
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, "/global/scratch/users/sergiomar10/boltzaff/pipeline_local")
from bft_common import *

_spec = importlib.util.spec_from_file_location("bft_metrics", ROOT / "BoltzFT/evaluation/metrics.py")
_m = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_m)
MCOLS = {"ap": "auprc", "ef1": "ef_1pct", "bedroc": "bedroc"}
MNAME = {"ap": "AP", "ef1": "EF@1%", "bedroc": "BEDROC(a=20)"}


def metrics_table(t, cache=True):
    """Rows: arm ('base'/'headft'), n_train, seed, ap, ef1, bedroc, n_eval, n_active. None if target not fully scored."""
    S = scores(t)
    if S is None: return None
    cp = AN / t / "repl_metrics.csv"
    if cache and cp.exists():
        d = pd.read_csv(cp)
        if len(d) == 16 and d.n_eval.iloc[0] == len(S): return d
    y = S.label.values.astype(int); rows = []
    def one(arm, N, s, col):
        m = _m.all_metrics(y, S[col].values.astype(float))
        rows.append(dict(target=t, arm=arm, n_train=N, seed=s, ap=m["auprc"], ef1=m["ef_1pct"], bedroc=m["bedroc"], n_eval=m["n_eval"], n_active=m["n_active"]))
    one("base", 0, -1, "noft_p")
    for N in BUDGETS:
        for s in SEEDS: one("headft", N, s, f"ft{N}_p{s}")
    d = pd.DataFrame(rows)
    if cache:
        try: cp.parent.mkdir(parents=True, exist_ok=True); d.to_csv(cp, index=False)
        except Exception as e: print("cache write failed:", e)
    return d


def geo_stats(ratios):
    """Cross-target summary of a list of per-target ratios (targets = unit)."""
    r = np.asarray(ratios, float); k = len(r)
    out = dict(k=k, geo=float(np.exp(np.log(r).mean())) if k else np.nan, lo=np.nan, hi=np.nan, improved=int((r > 1).sum()), sign_p=np.nan, wilcoxon_p=np.nan)
    if k >= 2:
        lr = np.log(r); se = lr.std(ddof=1) / np.sqrt(k); tc = stats.t.ppf(0.975, k - 1)
        out["lo"], out["hi"] = float(np.exp(lr.mean() - tc * se)), float(np.exp(lr.mean() + tc * se))
        out["sign_p"] = float(stats.binomtest(out["improved"], k, 0.5).pvalue)
        try: out["wilcoxon_p"] = float(stats.wilcoxon(lr).pvalue)
        except Exception: pass
    return out


def nofT_table():
    """One row per target: paper No-FT metrics, the dataset's shipped Boltz-2 score on the paper's eval subset (all 8 targets, available now),
    and our own No-FT re-scoring where the base arm is complete."""
    rows = []
    for t in TARGETS:
        r = dict(target=SHORT[t], paper_AP=PAPER_BASE.loc[t, "auprc"], paper_EF1=PAPER_BASE.loc[t, "ef_1pct"], paper_n_eval=int(PAPER_BASE.loc[t, "n_eval"]), paper_n_active=int(PAPER_BASE.loc[t, "n_active"]))
        sh = shipped_eval(t)
        if sh is not None:
            m = _m.all_metrics(sh.label.values, sh.p.values.astype(float)); r.update(shipped_n_eval=m["n_eval"], shipped_n_active=m["n_active"], shipped_AP=m["auprc"], shipped_EF1=m["ef_1pct"], shipped_BEDROC=m["bedroc"])
        bs = base_scores(t)
        if bs is not None:
            m = _m.all_metrics(bs.label.values, bs.noft_p.values.astype(float)); r.update(ours_n_eval=m["n_eval"], ours_AP=m["auprc"], ours_EF1=m["ef_1pct"], ours_BEDROC=m["bedroc"])
            if sh is not None:
                j = bs.merge(sh, on="id", suffixes=("_ours", "_shipped"))
                r["spearman_ours_vs_shipped"] = float(stats.spearmanr(j.noft_p, j.p).correlation); r["n_common"] = len(j)
        rows.append(r)
    return pd.DataFrame(rows).set_index("target")

"""Helpers for notebooks/06_training_strategies.ipynb (training-set selection and strategy variants), target-generic."""
import sys, glob, re
sys.path.insert(0, "/global/scratch/users/sergiomar10/boltzaff/pipeline_local")
from bft_common import *
from scipy import stats

VAR_STRATS = ["top", "balanced", "actdiv", "random", "diverse", "stratified", "hardneg"]


def ml(t):
    p = RUNS / t / "ft_inputs_full/ml_table.csv"
    return pd.read_csv(p, dtype={"complex_id": str}).set_index("complex_id") if p.exists() else None


def std_sets(t):
    """{N: DataFrame of the standard top-N training compounds (rows of ml_table)} or None."""
    m = ml(t)
    if m is None: return None
    out = {}
    for N in BUDGETS:
        ids = train_ids(t, N)
        if ids is None: return None
        out[N] = m.loc[ids]
    return out


def ap_by_arm(t):
    """Per (N, seed) AP/EF1% on the full eval set from bft_common.scores; cached. None if not fully scored."""
    S = scores(t)
    if S is None: return None
    cp = AN / t / "train_strategies_ap.csv"
    if cp.exists():
        d = pd.read_csv(cp)
        if len(d) == 16 and d.n_eval.iloc[0] == len(S): return d
    y = S.label.values.astype(int); rows = []
    def ef1(s):
        k = max(1, int(round(0.01 * len(y)))); o = np.argsort(-s, kind="stable")[:k]; return y[o].mean() / y.mean()
    def one(N, sd, col):
        s = S[col].values.astype(float); rows.append(dict(N=N, seed=sd, ap=average_precision(y, s), ef1=ef1(s), n_eval=len(y), n_active=int(y.sum())))
    one(0, -1, "noft_p")
    for N in BUDGETS:
        for sd in SEEDS: one(N, sd, f"ft{N}_p{sd}")
    d = pd.DataFrame(rows)
    try: cp.parent.mkdir(parents=True, exist_ok=True); d.to_csv(cp, index=False)
    except Exception as e: print("cache write failed:", e)
    return d


def variants(t):
    """(per_seed, spread) DataFrames for target t, or None if there are no variant arms."""
    a = AN / t
    try:
        per = pd.read_csv(a / "variants_per_seed.csv"); sp = pd.read_csv(a / "variants_spread.csv")
    except Exception:
        return None
    return (per, sp) if len(per) else None


def compare(per, metric="ap"):
    """Each strategy vs top at the same N, paired over seeds; ratio = seed geometric mean of paired ratios, p = paired t-test on logs
    (same statistic as pipeline_local/eval_variants.compare_table)."""
    rows = []
    for n, gN in per.groupby("n_train"):
        if "top" not in set(gN.strategy): continue
        top = gN[gN.strategy == "top"].set_index("seed")
        for st, gS in gN.groupby("strategy"):
            if st == "top": continue
            s = gS.set_index("seed"); sds = sorted(set(top.index) & set(s.index))
            if len(sds) < 2: continue
            tv = top.loc[sds, metric].to_numpy(float); sv = s.loc[sds, metric].to_numpy(float)
            rows.append(dict(n_train=int(n), strategy=st, top_mean=tv.mean(), strat_mean=sv.mean(), ratio=float(np.exp(np.mean(np.log(sv / tv)))),
                             n_seeds=len(sds), p=float(stats.ttest_rel(np.log(sv), np.log(tv)).pvalue)))
    return pd.DataFrame(rows)


def var_train_ids(t, strat, N):
    p = RUNS / t / f"ft_inputs_variants/train_ids_{strat}_N{N}.txt"
    return [l.strip() for l in open(p) if l.strip()] if p.exists() else None


def var_ml(t):
    p = RUNS / t / "ft_inputs_variants/ml_table.csv"
    return pd.read_csv(p, dtype={"complex_id": str}).set_index("complex_id") if p.exists() else None


def var_eval_ids(t):
    p = RUNS / t / "ft_inputs_variants/eval_ids.txt"
    return [l.strip() for l in open(p) if l.strip()] if p.exists() else None


def var_arm_scores(t, cond, seed=0):
    ev = var_eval_ids(t)
    fs = sorted(glob.glob(str(RUNS / t / f"headft_variants_scores/{cond}_seed{seed}/chunk_*.csv")))
    if not fs or ev is None: return None
    f = pd.concat([pd.read_csv(p) for p in fs], ignore_index=True); f["sample_id"] = f.sample_id.astype(str)
    return f.drop_duplicates("sample_id").set_index("sample_id").affinity_probability_binary.reindex(ev)


def val_curves(t):
    """Per-epoch validation BCE / mean prob of the standard head-FT arms (headft_full), or None."""
    rows = []
    for arm in sorted(glob.glob(str(RUNS / t / "headft_full/top*_seed*"))):
        name = arm.split("/")[-1]
        if name.startswith("work"): continue
        N = int(name.split("_")[0][3:]); sd = int(name.split("seed")[1])
        for e in range(5):
            try: d = pd.read_csv(f"{arm}/validation_predictions/val_predictions_epoch_{e}.csv")
            except Exception: continue
            y = d.binary_target.values; p = np.clip(d.affinity_probability_binary.values, 1e-9, 1 - 1e-9)
            rows.append(dict(N=N, seed=sd, epoch=e, val_prob=float(p.mean()), val_bce=float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))), n_pos=int(y.sum()), n=len(y)))
    return pd.DataFrame(rows) if rows else None

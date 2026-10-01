"""Helpers for notebook 03_ft_vs_noft_reranking (library-level No-FT vs head-FT comparison, re-ranking).
Uses bft_common.scores(t) only; ground truth is the binary label."""
import numpy as np, pandas as pd
from bft_common import *

FRACS = [0.005, 0.01, 0.02, 0.05]
ARMS = [("No-FT", None)] + [(f"FT N={N}", N) for N in BUDGETS]


def load(t):
    """(S, msg). S is None when the target is not fully scored; msg says what is missing."""
    try:
        S = scores(t)
    except Exception as e:                                   # one target's problem must not stop the notebook
        return None, f"pending: could not load scores ({type(e).__name__}: {e})"
    if S is None:
        st = status().loc[SHORT[t]]
        miss = []
        if not st.ft_inputs: miss.append("ft_inputs (eval ids) not built")
        miss.append(f"only {st.arms_scored} scoring arms finished")
        return None, "pending: " + "; ".join(miss)
    return S, ""


def order(score):
    return np.argsort(-np.asarray(score), kind="stable")


def kof(frac, N):
    return int(round(frac * N))


def cols(S, N):
    return [f"ft{N}_p{s}" for s in SEEDS if f"ft{N}_p{s}" in S]


def conf_k(score, y, k):
    o = order(score)[:k]; tp = int(y[o].sum()); P = int(y.sum()); neg = len(y) - P
    return dict(flagged=k, TP=tp, FP=k - tp, FN=P - tp, TN=neg - (k - tp), sens=tp / P, spec=1 - (k - tp) / neg, prec=tp / k)


def conf_thr(score, y, thr=0.5):
    m = np.asarray(score) > thr; tp = int((m & (y == 1)).sum()); k = int(m.sum()); P = int(y.sum()); neg = len(y) - P
    return dict(flagged=k, TP=tp, FP=k - tp, FN=P - tp, TN=neg - (k - tp), sens=tp / P, spec=1 - (k - tp) / neg, prec=tp / max(k, 1))


def fp_at_sens(score, y, tp_target):
    """False positives accumulated when the ranking first reaches tp_target actives."""
    o = order(score); cs = np.cumsum(y[o]); k = int(np.searchsorted(cs, tp_target) + 1)
    return k - int(tp_target), k


def score_sets(S):
    """{arm label: ensemble score}, {arm label: list of per-seed scores}."""
    ens = {"No-FT": S.noft_p.values}; per = {"No-FT": [S.noft_p.values]}
    for N in BUDGETS:
        c = cols(S, N); ens[f"FT N={N}"] = S[c].mean(axis=1).values; per[f"FT N={N}"] = [S[x].values for x in c]
    return ens, per


def boot_diff(s_a, s_b, y, frac=0.01, sens_levels=(0.25, 0.5), B=300, seed=0):
    """Paired stratified bootstrap over compounds (actives and inactives resampled separately).
    Returns dict of arrays (length B): dTP@frac (b - a), dFP at matched sensitivity (b - a) for each level."""
    rng = np.random.default_rng(seed); pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    out = {"dTP": [], **{f"dFP@{l}": [] for l in sens_levels}}
    for _ in range(B):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]); yb = y[idx]; k = kof(frac, len(idx))
        res = []
        for s in (s_a, s_b):
            sb = s[idx]; o = order(sb); cs = np.cumsum(yb[o]); res.append((cs, o))
        out["dTP"].append(res[1][0][k - 1] - res[0][0][k - 1])
        for l in sens_levels:
            tp = int(np.ceil(l * yb.sum())); out[f"dFP@{l}"].append((np.searchsorted(res[1][0], tp) + 1 - tp) - (np.searchsorted(res[0][0], tp) + 1 - tp))
    return {k: np.array(v) for k, v in out.items()}


def ci(a, lo=2.5, hi=97.5):
    return float(np.percentile(a, lo)), float(np.percentile(a, hi))


def sim_table(t):
    """id -> (smiles, label) for every compound of the raw dataset (train + eval), same id scheme as bft_common.scores."""
    d = pd.read_csv(DATA / f"{t}.csv"); d["id"] = f"{t}_" + d.CID.astype(str)
    ev = eval_ids(t)
    if not d.id.isin(ev).any(): d["id"] = f"{SHORT[t]}_" + d.CID.astype(str)
    return d.set_index("id")[["neut-smiles", "target_active_v2"]]


def fps(smiles):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*"); g = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    return [g.GetFingerprint(m) if (m := Chem.MolFromSmiles(s)) is not None else None for s in smiles]


def max_sim(smiles, ref_fps):
    from rdkit import DataStructs
    return np.array([max(DataStructs.BulkTanimotoSimilarity(f, ref_fps)) if f is not None else np.nan for f in fps(smiles)])


def oriented_percentiles(S, col):
    """Percentile (0-100, 100 = best) of a dataset score within the library, oriented so that actives have higher AUROC > 0.5."""
    from scipy.stats import rankdata
    v = S[col]; ok = v.notna()
    if ok.sum() < 100 or S.label[ok].nunique() < 2: return None
    r = pd.Series(np.nan, index=S.index); r[ok] = 100 * (rankdata(v[ok]) - 1) / (ok.sum() - 1)
    from sklearn.metrics import roc_auc_score
    a = roc_auc_score(S.label[ok], r[ok])
    return (r if a >= 0.5 else 100 - r), a

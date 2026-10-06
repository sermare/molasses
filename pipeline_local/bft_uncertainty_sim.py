"""Helpers for the 'uncertainty vs Tanimoto similarity' part of notebooks/02_uncertainty.ipynb.

tc_train(t): for every evaluation compound, the highest ECFP4 Tanimoto (2048 bits, radius 2, same as the rest of the project) to
  tc_all = any of the 300 training compounds (the N=300 set; N=40 and N=100 are nested inside it),
  tc_act = the training ACTIVES only (NaN if a target had no training actives).
Cached in results/analysis/<t>/tc_train.csv (about a minute per target to build).
partial_spearman(u, tc, s): rank correlation between u and tc after removing the linear effect of the score s on both (all in rank space).
boot_partial(...): bootstrap interval over compounds (ranks are computed once on the full sample, then resampled)."""
import numpy as np, pandas as pd
from scipy.stats import rankdata
import bft_common as bc, bft_rerank as br


def tc_train(t):
    f = bc.AN / t / "tc_train.csv"
    if f.exists(): return pd.read_csv(f, index_col=0)
    from rdkit import DataStructs
    S = bc.scores(t)
    if S is None: return None
    tt = br.sim_table(t); ids = bc.train_ids(t, 300)
    smi = tt["neut-smiles"].reindex(ids).values; act = np.array([bool(tt.target_active_v2.get(i, False)) for i in ids])
    rf = br.fps(smi); ok = np.array([x is not None for x in rf]); ref = [x for x in rf if x is not None]; ref_act = act[ok]
    out = np.full((len(S), 2), np.nan)
    for i, fp in enumerate(br.fps(S.smiles.values)):
        if fp is None: continue
        sims = np.array(DataStructs.BulkTanimotoSimilarity(fp, ref)); out[i, 0] = sims.max()
        if ref_act.any(): out[i, 1] = sims[ref_act].max()
    D = pd.DataFrame(out, index=S.id.values, columns=["tc_all", "tc_act"]); D.to_csv(f); return D


def _pearson_ranks(a, b): return float(np.corrcoef(a, b)[0, 1])


def _partial(ru, rt, rs):
    a, b, c = _pearson_ranks(ru, rt), _pearson_ranks(ru, rs), _pearson_ranks(rt, rs)
    return (a - b * c) / np.sqrt(max((1 - b * b) * (1 - c * c), 1e-12))


def partial_spearman(u, tc, s):
    m = np.isfinite(u) & np.isfinite(tc) & np.isfinite(s)
    return _partial(rankdata(u[m]), rankdata(tc[m]), rankdata(s[m]))


def boot_partial(u, tc, s, B=200, seed=0):
    m = np.isfinite(u) & np.isfinite(tc) & np.isfinite(s); ru, rt, rs = rankdata(u[m]), rankdata(tc[m]), rankdata(s[m]); n = m.sum()
    rng = np.random.default_rng(seed); v = []
    for _ in range(B):
        i = rng.integers(0, n, n); v.append(_partial(ru[i], rt[i], rs[i]))
    return _partial(ru, rt, rs), np.percentile(v, [2.5, 97.5])

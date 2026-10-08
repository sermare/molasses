"""Helpers for notebooks/08_latent_space_and_baselines.ipynb: the compound sample used for latent-space extraction, loading of extracted features, and the analyses.
Sampling per target (results/analysis/<t>/latent/sample.csv): the 300 training compounds, ALL evaluation actives, a random set of evaluation inactives (default 2,500),
and the top-1% picks of No-FT and of the 5-seed head-FT ensemble (N=300), so that true and false positives are all present.
`w` makes the random part an unbiased estimate of the whole evaluation set: actives weight 1, randomly drawn inactives weight n_inactives / n_drawn, extra inactives that
are in only because they are top picks weight 0 (use them for qualitative views, never for estimating AP)."""
import numpy as np, pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))


def sample_path(t):
    import bft_common as bc
    return bc.AN / t / "latent" / "sample.csv"


def make_sample(t, n_random=2500, seed=0, force=False):
    import bft_common as bc
    p = sample_path(t)
    if p.exists() and not force: return pd.read_csv(p)
    S = bc.scores(t)
    if S is None: raise RuntimeError(f"{t} is not fully scored")
    ml = pd.read_csv(bc.RUNS / t / "ft_inputs_full/ml_table.csv").set_index("complex_id")
    tr_ids = bc.train_ids(t, 300)
    tr = pd.DataFrame({"id": tr_ids, "label": ml.is_binder.reindex(tr_ids).astype(int).values, "smiles": ml.smiles.reindex(tr_ids).values})
    tr["is_train"] = True; tr["noft_p"] = ml.affinity_probability_binary.reindex(tr_ids).values; tr["ft_p"] = np.nan
    ev = S[["id", "label", "smiles", "noft_p"]].copy(); ev["ft_p"] = bc.ft_mean(S, 300).values; ev["is_train"] = False
    k = int(np.ceil(0.01 * len(ev))); ev["pick_noft"] = False; ev["pick_ft"] = False
    ev.loc[ev.noft_p.nlargest(k).index, "pick_noft"] = True; ev.loc[ev.ft_p.nlargest(k).index, "pick_ft"] = True
    act = ev[ev.label == 1]; ina = ev[ev.label == 0]; rnd = ina.sample(n=min(n_random, len(ina)), random_state=seed)
    ev["random_inactive"] = ev.index.isin(rnd.index)
    keep = ev[(ev.label == 1) | ev.random_inactive | ev.pick_noft | ev.pick_ft].copy()
    keep["w"] = np.where(keep.label == 1, 1.0, np.where(keep.random_inactive, len(ina) / len(rnd), 0.0))
    tr["pick_noft"] = tr["pick_ft"] = False; tr["random_inactive"] = False; tr["w"] = 0.0
    D = pd.concat([tr, keep], ignore_index=True); D["group"] = np.where(D.is_train, "train", np.where(D.label == 1, "eval active", "eval inactive"))
    p.parent.mkdir(parents=True, exist_ok=True); D.to_csv(p, index=False); (p.parent / "sample_ids.txt").write_text("\n".join(D.id) + "\n")
    return D


def load_latent(t, arm):
    """arrays of one arm: ids, g (n, 2, 384) post-MLP features, g_raw (n, 2, 128), value (n, 2), logit (n, 2); None if the arm is not extracted yet."""
    import bft_common as bc
    f = bc.AN / t / "latent" / f"{arm}.npz"
    if not f.exists(): return None
    z = np.load(f, allow_pickle=True); return {k: z[k] for k in z.files}


# =================================================================================================================== analyses on the extracted features
from scipy.stats import rankdata, spearmanr


def aligned(t, arms=("noft", "ft300s0", "ft300s1"), kind="g", module="mean"):
    """(sample DataFrame, {arm: (n, d) array aligned to the sample rows}); arms that are not extracted yet are skipped. module: 'mean' of the two ensemble modules, or 0 / 1."""
    D = make_sample(t); out = {}
    for arm in arms:
        L = load_latent(t, arm)
        if L is None: continue
        pos = pd.Series(np.arange(len(L["ids"])), index=[str(i) for i in L["ids"]]); X = L[kind]
        X = X.mean(1) if module == "mean" else X[:, int(module)]
        out[arm] = X[pos.reindex(D.id).values]
    return D, out


def wavg_precision(y, s, w):
    """Average precision with sample weights (random inactives carry the weight that makes the sample an unbiased estimate of the whole evaluation set)."""
    from sklearn.metrics import average_precision_score
    return average_precision_score(y, s, sample_weight=w)


def wauc(y, s, w):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y, s, sample_weight=w)


def pca_coords(Xn, Xf, k=10):
    """PCA basis fitted on the No-FT features of the sample; both arms are projected on it (so a shift between the two panels is a real shift of the features)."""
    from sklearn.decomposition import PCA
    p = PCA(n_components=k, random_state=0).fit(Xn); return p.transform(Xn), p.transform(Xf), p.explained_variance_ratio_


def probe_table(t, arms=("noft", "ft300s0")):
    """Linear probes (logistic regression, features standardised, C by 5-fold CV inside the training compounds) on frozen features.
    trained_on_300: trained on the 300 training compounds, evaluated on the evaluation sample (weighted, estimates the whole evaluation set).
    oracle_cv: 5-fold cross-validated on the evaluation sample itself: an upper bound of what is linearly readable from these features.
    head_score: the head's own probability on the same compounds, for comparison."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    D, X = aligned(t, arms); rows = []
    tr = D.is_train.values; ev = (~D.is_train.values) & (D.w.values > 0); y = D.label.values
    for arm, F in X.items():
        sc = StandardScaler().fit(F[tr]); Ftr = sc.transform(F[tr]); Fev = sc.transform(F[ev])
        best, bestap = 0.1, -1
        for C in (0.001, 0.01, 0.1, 1):
            cv = StratifiedKFold(5, shuffle=True, random_state=0)
            ap = cross_val_score(LogisticRegression(C=C, max_iter=2000, class_weight="balanced"), Ftr, y[tr], cv=cv, scoring="average_precision").mean()
            if ap > bestap: best, bestap = C, ap
        clf = LogisticRegression(C=best, max_iter=2000, class_weight="balanced").fit(Ftr, y[tr]); s = clf.decision_function(Fev)
        rows.append(dict(target=bc_short(t), arm=arm, probe="trained on the 300 training compounds", ap=wavg_precision(y[ev], s, D.w.values[ev]), auroc=wauc(y[ev], s, D.w.values[ev]), C=best))
        # oracle: cross-validated on the evaluation sample
        idx = np.where(ev)[0]; sk = StratifiedKFold(5, shuffle=True, random_state=0); sc_all = np.zeros(len(idx))
        for a, b in sk.split(idx, y[idx]):
            ss = StandardScaler().fit(F[idx[a]]); m = LogisticRegression(C=0.01, max_iter=2000).fit(ss.transform(F[idx[a]]), y[idx[a]], sample_weight=D.w.values[idx[a]])
            sc_all[b] = m.decision_function(ss.transform(F[idx[b]]))
        rows.append(dict(target=bc_short(t), arm=arm, probe="cross-validated on the evaluation sample (upper bound)", ap=wavg_precision(y[idx], sc_all, D.w.values[idx]), auroc=wauc(y[idx], sc_all, D.w.values[idx]), C=0.01))
    for arm, col in (("noft", "noft_p"), ("ft300s0", "ft_p")):
        if arm in X: rows.append(dict(target=bc_short(t), arm=arm, probe="the head's own probability", ap=wavg_precision(y[ev], D[col].values[ev], D.w.values[ev]), auroc=wauc(y[ev], D[col].values[ev], D.w.values[ev]), C=np.nan))
    return pd.DataFrame(rows)


def bc_short(t):
    import bft_common as bc
    return bc.SHORT[t]


def displacement_table(t, arm="ft300s0", ref="noft"):
    """Direction and size of the feature shift d = f_FT - f_NoFT per compound (384-dim post-MLP features, mean of the two modules).
    coherence = |mean d| / mean |d|: 1 if every compound moves the same way, near 0 if the shifts point in unrelated directions.
    cos_to_activity_direction: cosine between a compound's shift and the mean shift of the TRAINING actives (the 'activity direction'); its AUROC for the label is on eval compounds.
    rank_change: percentile rank under head-FT minus under No-FT, over the whole evaluation set (negative = demoted); partial_rho controls for the No-FT rank."""
    import bft_common as bc
    D, X = aligned(t, (ref, arm)); 
    if arm not in X or ref not in X: return None
    d = X[arm] - X[ref]; nrm = np.linalg.norm(d, axis=1); g = D.group.values; ev = (~D.is_train.values) & (D.w.values > 0); y = D.label.values
    S = bc.scores(t); p0 = pd.Series(S.noft_p.values, index=S.id).rank(pct=True); p1 = pd.Series(bc.ft_mean(S, 300).values, index=S.id).rank(pct=True)
    out = dict(target=bc.SHORT[t], coherence=float(np.linalg.norm(d.mean(0)) / nrm.mean()), norm_train_active=float(nrm[(g == "train") & (y == 1)].mean()), norm_train_inactive=float(nrm[(g == "train") & (y == 0)].mean()),
               norm_eval_active=float(nrm[g == "eval active"].mean()), norm_eval_inactive=float(nrm[g == "eval inactive"].mean()))
    rng = np.random.default_rng(0)
    def mean_cos(a, b, n=4000):
        i = rng.integers(0, len(a), n); j = rng.integers(0, len(b), n); u = a[i] / np.linalg.norm(a[i], axis=1, keepdims=True); v = b[j] / np.linalg.norm(b[j], axis=1, keepdims=True); return float((u * v).sum(1).mean())
    out["cos_random_pairs_eval_inactive"] = mean_cos(d[g == "eval inactive"], d[g == "eval inactive"]); out["cos_random_pairs_eval_active"] = mean_cos(d[g == "eval active"], d[g == "eval active"])
    out["cos_active_vs_inactive"] = mean_cos(d[g == "eval active"], d[g == "eval inactive"])
    sv = np.linalg.svd(d - 0, compute_uv=False); out["share_of_shift_in_first_direction"] = float(sv[0] ** 2 / (sv ** 2).sum())
    a_dir = d[(g == "train") & (y == 1)].mean(0); a_dir /= np.linalg.norm(a_dir); cosd = (d @ a_dir) / np.maximum(nrm, 1e-9)
    out["auroc_cos_to_activity_direction"] = wauc(y[ev], cosd[ev], D.w.values[ev]); out["auroc_shift_norm"] = wauc(y[ev], nrm[ev], D.w.values[ev])
    rc = (p1.reindex(D.id).values - p0.reindex(D.id).values); r0 = p0.reindex(D.id).values
    m = ev & np.isfinite(rc); out["rho_norm_vs_rank_change"] = float(spearmanr(nrm[m], rc[m])[0])
    rn = rankdata(nrm[m]); rr = rankdata(rc[m]); r00 = rankdata(r0[m]); 
    def resid(a, b): A = np.c_[np.ones(len(b)), b]; return a - A @ np.linalg.lstsq(A, a, rcond=None)[0]
    out["partial_rho_norm_vs_rank_change_given_noft_rank"] = float(np.corrcoef(resid(rn, r00), resid(rr, r00))[0, 1])
    return out, d, D


def neighbour_table(t, arm_pairs=(("noft", "No-FT latent"), ("ft300s0", "head-FT latent"))):
    """Nearest-active baselines: rank evaluation-sample compounds by their highest cosine similarity (in the given feature space) to a TRAINING ACTIVE, and compare with the Morgan
    Tanimoto nearest-active baseline and with the head's own probability. Weighted AP / AUROC on the sample (estimates of the whole evaluation set)."""
    import bft_common as bc
    D, X = aligned(t, [a for a, _ in arm_pairs]); tc = pd.read_csv(bc.AN / t / "tc_train.csv", index_col=0)
    ev = (~D.is_train.values) & (D.w.values > 0); y = D.label.values; w = D.w.values; rows = []
    tr_act = D.is_train.values & (y == 1)
    for arm, name in arm_pairs:
        if arm not in X: continue
        F = X[arm] / np.linalg.norm(X[arm], axis=1, keepdims=True); sim = (F[ev] @ F[tr_act].T).max(1)
        rows.append(dict(target=bc.SHORT[t], ranking=f"nearest training active, {name} (cosine)", ap=wavg_precision(y[ev], sim, w[ev]), auroc=wauc(y[ev], sim, w[ev])))
    s = tc.tc_act.reindex(D.id[ev]).values; ok = np.isfinite(s)
    rows.append(dict(target=bc.SHORT[t], ranking="nearest training active, Morgan Tanimoto", ap=wavg_precision(y[ev][ok], s[ok], w[ev][ok]), auroc=wauc(y[ev][ok], s[ok], w[ev][ok])))
    for col, name in (("noft_p", "No-FT probability"), ("ft_p", "head-FT probability")):
        rows.append(dict(target=bc.SHORT[t], ranking=name, ap=wavg_precision(y[ev], D[col].values[ev], w[ev]), auroc=wauc(y[ev], D[col].values[ev], w[ev])))
    rows.append(dict(target=bc.SHORT[t], ranking="random", ap=float(np.average(y[ev], weights=w[ev])), auroc=0.5))
    return pd.DataFrame(rows)


def applicability_table(t, arm="ft300s0"):
    """Top-1% picks of head-FT (all are in the sample): share that are active by tercile of (a) latent similarity (head-FT features) to the nearest training active and (b) Tanimoto to the nearest training active."""
    import bft_common as bc
    D, X = aligned(t, (arm,)); 
    if arm not in X: return None
    tc = pd.read_csv(bc.AN / t / "tc_train.csv", index_col=0); y = D.label.values; picks = D.pick_ft.values & (~D.is_train.values); tr_act = D.is_train.values & (y == 1)
    F = X[arm] / np.linalg.norm(X[arm], axis=1, keepdims=True); lat = (F[picks] @ F[tr_act].T).max(1); tan = tc.tc_act.reindex(D.id[picks]).values; yy = y[picks]; rows = []
    for name, v in (("latent cosine to nearest training active", lat), ("Morgan Tanimoto to nearest training active", tan)):
        ok = np.isfinite(v); q = np.nanquantile(v[ok], [1 / 3, 2 / 3])
        for lab, m in (("lowest third", ok & (v <= q[0])), ("middle third", ok & (v > q[0]) & (v <= q[1])), ("highest third", ok & (v > q[1]))):
            rows.append(dict(target=bc.SHORT[t], measure=name, tercile=lab, picks=int(m.sum()), active_share=float(yy[m].mean()), value_range=f"{v[m].min():.2f}-{v[m].max():.2f}"))
    return pd.DataFrame(rows)


def chemistry_table(t, arm="ft300s0", k=5):
    """Spearman correlation between interpretable descriptors and (a) the first k principal components of the head-FT features, (b) the size of the shift f_FT - f_NoFT."""
    import bft_signals as sg, bft_common as bc
    D, X = aligned(t, ("noft", arm))
    if arm not in X: return None
    P = sg.physchem(t).set_index("id").reindex(D.id); ev = (~D.is_train.values) & (D.w.values > 0)
    pf = pca_coords(X[arm][ev], X[arm][ev], k)[0]                       # PCA of the head-FT features themselves
    shift = np.linalg.norm(X[arm] - X["noft"], axis=1)[ev]; rows = []
    for p in sg.PROPS:
        v = P[p].values[ev]; ok = np.isfinite(v)
        row = dict(target=bc.SHORT[t], property=p, shift_norm=float(spearmanr(v[ok], shift[ok])[0]))
        for j in range(k): row[f"PC{j + 1} (head-FT)"] = float(spearmanr(v[ok], pf[ok, j])[0])
        rows.append(row)
    return pd.DataFrame(rows)

#!/usr/bin/env python3
"""What explains the disagreement between the five head-FT seeds (N=300) on a compound?  python pipeline_local/variance_drivers.py <target>

Outcome y = rank (0-1) of the across-seed variance of the head-FT LOGIT, over the whole evaluation set (~50k compounds). The logit is used because variance of a probability is
dominated by the sigmoid (a probability near 0 cannot move much). Models: LightGBM, 3-fold cross-validated R^2 (and AUC for 'top quartile of variance').
Feature groups: score (mean head-FT logit, No-FT logit, their difference), chemistry (10 RDKit properties + Brenk / NIH flags; formal charge is constant in these libraries and dropped),
Tanimoto to the training set, other scores (Boltz-2 dataset, Boltzina, docking, GNINA, Vina), ECFP4 bits.
With --scaffold the cross-validation is grouped by generic Murcko scaffold (variance_drivers_scaffold.csv).
Writes results/analysis/<target>/variance_drivers.csv (long table: analysis, feature, R2, AUC), variance_pdp.csv (variance rank by decile of each property / of the score),
variance_ecfp_bits.csv (most useful ECFP bits for what the score does not explain)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
from scipy.stats import rankdata
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bft_common as bc, bft_signals as sg, bft_latent as lat

CHEM = [p for p in sg.PROPS if p != "charge"]; FLAGS = ["brenk", "nih"]
GROUPS = {"size (MW, heavy atoms, rings, aromatic rings, rotatable bonds)": ["MW", "heavy", "rings", "aromatic_rings", "rotatable"], "polarity (TPSA, HBD, HBA)": ["TPSA", "HBD", "HBA"],
          "lipophilicity (logP)": ["logP"], "saturation (fsp3)": ["fsp3"], "substructure flags (Brenk, NIH)": FLAGS}
SCORE = ["FT logit", "No-FT logit", "shift (No-FT - FT)"]
OTHER = ["score_boltz2", "score_boltzina", "docking_score_boltzina", "score_gnina", "score_vina"]
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))


GROUPS_ARRAY = None      # set by run(scaffold=True): generic Murcko scaffold id per compound, so no scaffold is split between training and test folds


def cv(X, y, jobs=2, seed=0):
    """3-fold CV (random, or grouped by scaffold when GROUPS_ARRAY is set): returns (R2, AUC of top quartile, out-of-fold prediction)."""
    pr = np.zeros(len(y)); splits = GroupKFold(3).split(X, y, GROUPS_ARRAY) if GROUPS_ARRAY is not None else KFold(3, shuffle=True, random_state=seed).split(X)
    for a, b in splits:
        pr[b] = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=31, min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=jobs).fit(X[a], y[a]).predict(X[b])
    return 1 - ((y - pr) ** 2).sum() / ((y - y.mean()) ** 2).sum(), roc_auc_score(y > np.quantile(y, .75), pr), pr


def run(t, scaffold=False):
    global GROUPS_ARRAY
    S = bc.scores(t)
    if S is None or S[[f"ft300_p{s}" for s in range(5)]].isna().any().any(): return None
    S = S.set_index("id"); P = lg(S[[f"ft300_p{s}" for s in range(5)]].values); y = rankdata(P.var(1)) / len(S); n = len(S)
    GROUPS_ARRAY = pd.factorize(pd.read_csv(bc.AN / t / "signals_scaffolds.csv").set_index("id").reindex(S.index).generic.fillna("none"))[0] if scaffold else None
    ph = sg.physchem(t).set_index("id").reindex(S.index); fl = sg.filter_flags(t).set_index("id").reindex(S.index)
    tc = pd.read_csv(bc.AN / t / "tc_train.csv").set_index("Unnamed: 0").reindex(S.index)
    F = pd.DataFrame({"FT logit": P.mean(1), "No-FT logit": lg(S.noft_p.values)}, index=S.index); F["shift (No-FT - FT)"] = F["No-FT logit"] - F["FT logit"]
    for c in CHEM: F[c] = ph[c].values.astype(float)
    for c in FLAGS: F[c] = fl[c].values.astype(float)
    F["Tanimoto to training (all)"] = tc.tc_all.values; F["Tanimoto to training actives"] = tc.tc_act.values
    for c in OTHER: F[c] = S[c].values.astype(float)
    rows = []
    def add(analysis, feature, X, yy=y):
        r2, auc, pr = cv(np.asarray(X, float).reshape(n, -1), yy); rows.append(dict(target=bc.SHORT[t], analysis=analysis, feature=feature, R2=r2, AUC=auc)); print(bc.SHORT[t], analysis, feature, round(r2, 3), flush=True); return pr
    chem_cols = CHEM + FLAGS
    # A. one feature (or group) alone
    p_score = add("alone", "score (all three)", F[SCORE])
    for c in SCORE + chem_cols + ["Tanimoto to training (all)", "Tanimoto to training actives"] + OTHER: add("alone", c, F[[c]])
    for g, cols in GROUPS.items(): add("alone", g, F[cols])
    add("alone", "chemistry (all 12)", F[chem_cols]); add("alone", "other scores (all 5)", F[OTHER]); add("alone", "Tanimoto to training (both)", F[["Tanimoto to training (all)", "Tanimoto to training actives"]])
    # B. score + chemistry, leave one out / leave one group out
    base = SCORE + chem_cols; add("score + chemistry: nothing removed", "-", F[base])
    for c in chem_cols: add("score + chemistry: leave one out", c, F[[x for x in base if x != c]])
    for g, cols in GROUPS.items(): add("score + chemistry: leave one group out", g, F[[x for x in base if x not in cols]])
    add("score + chemistry: leave one group out", "score (all three)", F[chem_cols])
    for c in SCORE: add("score + chemistry: leave one out", c, F[[x for x in base if x != c]])
    # C. score plus one chemistry feature (what each adds to the score)
    for c in chem_cols: add("score + one chemistry feature", c, F[SCORE + [c]])
    for g, cols in GROUPS.items(): add("score + one chemistry group", g, F[SCORE + cols])
    # D. what is left after the score: model the residual of a score-only model (out-of-fold)
    res = y - p_score; res = res - res.mean()
    for c in chem_cols: add("on the residual of the score", c, F[[c]], res)
    for g, cols in GROUPS.items(): add("on the residual of the score", g, F[cols], res)
    add("on the residual of the score", "chemistry (all 12)", F[chem_cols], res); add("on the residual of the score", "Tanimoto to training (both)", F[["Tanimoto to training (all)", "Tanimoto to training actives"]], res)
    add("on the residual of the score", "other scores (all 5)", F[OTHER], res)
    E = lat.ecfp_matrix(S.smiles.values).astype(np.float32); add("on the residual of the score", "ECFP4 bits (2048)", E, res)
    add("on the residual of the score", "chemistry + ECFP", np.hstack([F[chem_cols].values, E]), res)
    add("alone", "ECFP4 bits (2048)", E); add("score + one chemistry group", "ECFP4 bits (2048)", np.hstack([F[SCORE].values, E]))
    out = bc.AN / t
    if scaffold: pd.DataFrame(rows).to_csv(out / "variance_drivers_scaffold.csv", index=False); return True
    pd.DataFrame(rows).to_csv(out / "variance_drivers.csv", index=False)
    # direction: variance rank (raw and after the score) by decile of each property and of the score
    pdp = []
    for c in SCORE[:2] + chem_cols + ["Tanimoto to training (all)"]:
        v = F[c].values; q = pd.qcut(pd.Series(v).rank(method="first"), 10, labels=False) if len(np.unique(v)) > 2 else pd.Series((v > np.median(v)).astype(int)).values
        if len(np.unique(v)) <= 2: q = (v == np.max(v)).astype(int)
        d = pd.DataFrame({"bin": q, "value": v, "y": y, "resid": res}).groupby("bin").agg(value=("value", "median"), variance_rank=("y", "median"), residual=("resid", "median"), n=("y", "size")).reset_index(); d.insert(0, "feature", c); d.insert(0, "target", bc.SHORT[t]); pdp.append(d)
    sd = pd.DataFrame({"bin": pd.qcut(rankdata(F["FT logit"]), 10, labels=False), "sd_logit": P.std(1), "sd_prob": (1 / (1 + np.exp(-P))).std(1)}).groupby("bin").median().reset_index()
    sd.insert(0, "target", bc.SHORT[t]); sd.to_csv(out / "variance_score_deciles.csv", index=False); pd.concat(pdp).to_csv(out / "variance_pdp.csv", index=False)
    # ECFP bits that help most beyond the score
    m = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=31, min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=2, importance_type="gain").fit(E, res)
    g = m.feature_importances_ / m.feature_importances_.sum(); top = np.argsort(-g)[:30]
    pd.DataFrame({"target": bc.SHORT[t], "bit": top, "gain_share": g[top], "prevalence": E[:, top].mean(0), "residual_with_bit": [np.median(res[E[:, b] > 0]) if (E[:, b] > 0).any() else np.nan for b in top],
                  "residual_without_bit": [np.median(res[E[:, b] == 0]) for b in top], "variance_rank_with_bit": [np.median(y[E[:, b] > 0]) if (E[:, b] > 0).any() else np.nan for b in top]}).to_csv(out / "variance_ecfp_bits.csv", index=False)
    return True


def ecfp_alone(t):
    """add the two ECFP rows to an existing variance_drivers.csv without recomputing everything (older runs of run() lacked them)"""
    f = bc.AN / t / "variance_drivers.csv"; D = pd.read_csv(f)
    if ((D.analysis == "alone") & (D.feature == "ECFP4 bits (2048)")).any(): return
    S = bc.scores(t).set_index("id"); P = lg(S[[f"ft300_p{s}" for s in range(5)]].values); y = rankdata(P.var(1)) / len(S); n = len(S)
    F = np.column_stack([P.mean(1), lg(S.noft_p.values), lg(S.noft_p.values) - P.mean(1)]); E = lat.ecfp_matrix(S.smiles.values).astype(np.float32); rows = []
    for an, X in (("alone", E), ("score + one chemistry group", np.hstack([F, E]))):
        r2, auc, _ = cv(X, y); rows.append(dict(target=bc.SHORT[t], analysis=an, feature="ECFP4 bits (2048)", R2=r2, AUC=auc)); print(bc.SHORT[t], an, round(r2, 3), flush=True)
    pd.concat([D, pd.DataFrame(rows)]).to_csv(f, index=False)


if __name__ == "__main__":
    t = sys.argv[1]
    if len(sys.argv) > 2 and sys.argv[2] == "--ecfp-alone": ecfp_alone(t)
    elif len(sys.argv) > 2 and sys.argv[2] == "--scaffold": r = run(t, scaffold=True); print("done" if r else "no data", t)
    else: r = run(t); print("done" if r else "no data", t)

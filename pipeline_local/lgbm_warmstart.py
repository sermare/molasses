#!/usr/bin/env python3
"""LightGBM on ECFP, from scratch against two warm-started versions that start from Boltz-2's own score (the analogue of fine-tuning).

    python pipeline_local/lgbm_warmstart.py --target 434954-2097          -> results/analysis/<target>/lgbm_warmstart.csv

Same data as cpu_baselines.py and the head-FT runs: training compounds = top-N of the Boltz-2 ranking (N = 40, 100, 300, nested), evaluation = every
compound ranked below 300, same metrics (BoltzFT/evaluation/metrics.py). Arms (5 seeds each; the seeds only change LightGBM's row/column subsampling,
the training set is fixed, as for head-FT):
  Boltz2_noFT(score)        the shipped Boltz-2 score on its own (reference; ratio 1 by definition)
  LGBM_ECFP_scratch         what the paper's baseline does: ECFP only, no Boltz information
  LGBM_ECFP_residual        boosting starts from logit(Boltz-2 score) (init_score) and learns a correction from ECFP; the number of trees is chosen by 3-fold CV
                            inside the labelled set, and 0 trees (= the Boltz-2 score unchanged) is one of the candidates
  LGBM_ECFP_plus_score      ECFP plus logit(Boltz-2 score) as one more feature, from scratch
  NearestActive_Tanimoto    rank by the highest ECFP4 Tanimoto to a training active (deterministic)
Caveat built into the design: all training compounds come from the top of the Boltz-2 ranking, so their scores span a narrow range."""
import argparse, itertools, sys
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cpu_baselines as cb
from cpu_baselines import all_metrics
from rdkit import DataStructs

KEYS = ["auroc", "auprc", "ef_1pct", "bedroc", "n_eval", "n_active"]
GRID_RES = [(0, 0, 0.0)] + list(itertools.product([25, 50, 100, 300], [7, 15], [0.02, 0.05, 0.1]))    # (trees, leaves, learning rate); first = score unchanged
COMMON = dict(subsample=0.8, subsample_freq=1, colsample_bytree=0.8, min_child_samples=5, min_data_in_bin=1, verbose=-1, n_jobs=2)


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6); return np.log(p / (1 - p))


def fit_resid(X, y, s, cfg, seed):
    if cfg[0] == 0: return None
    clf = lgb.LGBMClassifier(n_estimators=cfg[0], num_leaves=cfg[1], learning_rate=cfg[2], random_state=seed, bagging_seed=seed, feature_fraction_seed=seed, **COMMON)
    clf.fit(X, y, init_score=s); return clf


def predict_resid(clf, X, s):
    return s if clf is None else clf.predict(X, raw_score=True) + s


def cv_resid(X, y, s):
    if min(np.bincount(y, minlength=2)) < 3: return GRID_RES[0], float("nan")          # too few labels to cross-validate: keep the Boltz-2 score
    skf = StratifiedKFold(3, shuffle=True, random_state=0); best, best_ap = GRID_RES[0], -1
    for cfg in GRID_RES:
        aps = []
        for a, b in skf.split(X, y):
            if len(set(y[b])) < 2: continue
            aps.append(average_precision_score(y[b], predict_resid(fit_resid(X[a], y[a], s[a], cfg, 0), X[b], s[b])))
        m = float(np.mean(aps)) if aps else -1
        if m > best_ap + 1e-9: best_ap, best = m, cfg                                  # strict improvement: ties keep the simpler model (fewer trees)
    return best, best_ap


def row(method, N, seed, m, **extra):
    return dict(method=method, n_train=N, seed=seed, **{k: m[k] for k in KEYS}, **extra)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--target", required=True); ap.add_argument("--sizes", type=int, nargs="+", default=[40, 100, 300])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4]); a = ap.parse_args()
    df = cb.load(a.target); print(f"[{a.target}] {len(df)} compounds; ECFP ...", flush=True)
    fps, bits = cb.fingerprints(df["neut-smiles"].tolist()); y_all = df["Active_v2"].to_numpy(int); score = df["affinity_probability_binary"].to_numpy(float); lg = logit(score)
    ev = df["rank"].to_numpy() > 300; yE = y_all[ev]; XE = bits[ev].astype(np.float32); sE = lg[ev]; fpsE = [fps[i] for i in np.where(ev)[0]]
    assert set(df.iloc[:300].CID) & set(df.loc[ev, "CID"]) == set(), "train/eval overlap"
    rows = [row("Boltz2_noFT(score)", 0, -1, all_metrics(yE, score[ev]))]
    for N in a.sizes:
        y = y_all[:N]; X = bits[:N].astype(np.float32); s = lg[:N]
        if min(np.bincount(y, minlength=2)) < 1: print(f"N={N}: no positive or no negative, skipped", flush=True); continue
        # nearest active (Tanimoto)
        act = [fps[i] for i in range(N) if y[i] == 1 and fps[i] is not None]
        if act: rows.append(row("NearestActive_Tanimoto", N, -1, all_metrics(yE, np.array([max(DataStructs.BulkTanimotoSimilarity(f, act)) if f is not None else 0.0 for f in fpsE]))))
        # from scratch (ECFP only) and ECFP + score feature: grid CV as in cpu_baselines
        for name, Xtr, Xev in (("LGBM_ECFP_scratch", X, XE), ("LGBM_ECFP_plus_score", np.hstack([X, s[:, None]]), np.hstack([XE, sE[:, None]]))):
            cfg = cb.cv_select(Xtr, y); preds = []
            for sd in a.seeds:
                clf = lgb.LGBMClassifier(**cfg, random_state=sd, bagging_seed=sd, feature_fraction_seed=sd, **COMMON).fit(Xtr, y)
                p = clf.predict_proba(Xev)[:, 1]; preds.append(p); rows.append(row(name, N, sd, all_metrics(yE, p), cfg=str(cfg)))
            rows.append(row(name + "_avg5", N, -1, all_metrics(yE, np.mean(preds, 0)), cfg=str(cfg)))
        # residual boosting from the Boltz-2 score
        cfg, cvap = cv_resid(X, y, s); preds = []
        for sd in a.seeds:
            clf = fit_resid(X, y, s, cfg, sd); p = predict_resid(clf, XE, sE); preds.append(p); rows.append(row("LGBM_ECFP_residual", N, sd, all_metrics(yE, p), cfg=str(cfg), cv_ap=cvap))
        rows.append(row("LGBM_ECFP_residual_avg5", N, -1, all_metrics(yE, np.mean(preds, 0)), cfg=str(cfg), cv_ap=cvap))
        print(f"N={N}: residual cfg (trees, leaves, lr) = {cfg}, CV AP {cvap:.3f}", flush=True)
    out = Path(cb.ROOT / "results/analysis" / a.target); out.mkdir(parents=True, exist_ok=True); D = pd.DataFrame(rows); D.insert(0, "target", a.target)
    D.to_csv(out / "lgbm_warmstart.csv", index=False)
    g = D[~D.method.str.endswith("avg5")].groupby(["method", "n_train"]).auprc.mean().unstack().round(3); print(g.to_string())


if __name__ == "__main__":
    main()

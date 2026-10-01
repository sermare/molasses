#!/usr/bin/env python3
"""CPU, no-Boltz supervised baselines for one MF-PCBA target, on the common eval set.

Models (learn from molecular descriptors only; independent of the Boltz pipeline):
  - LightGBM on 2048-bit ECFP (Morgan r2), CV-tuned, 5 seeds (+5-model average)
  - Nearest-active Tanimoto (rank by max ECFP similarity to a training active)
Training compounds = top-N of the Boltz-2 ranking (N=40/100/300, nested), same as head-FT.
Eval = the common set (rank > 300). Writes cpu_baselines_<target>.csv.
"""
import argparse, gzip, sys, itertools
from pathlib import Path
import numpy as np, pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import StratifiedKFold, cross_val_score
import lightgbm as lgb

ROOT = Path("/global/scratch/users/sergiomar10/boltzaff")
sys.path.insert(0, str(ROOT / "BoltzFT/evaluation"))
from metrics import all_metrics

MFG = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
GRID = list(itertools.product([100, 300, 600], [15, 31, 63], [0.01, 0.05, 0.1]))  # 27 combos


def load(target):
    df = pd.read_csv(ROOT / f"data/{target}_results.csv")
    df["CID"] = df["CID"].astype(int)
    ids = [int(x.strip().removeprefix(target + "_"))
           for x in gzip.open(ROOT / f"BoltzFT/data/splits/{target}.txt.gz", "rt")]
    df = df.set_index("CID").loc[ids].reset_index()
    df["rank"] = np.arange(1, len(df) + 1)
    return df


def fingerprints(smiles):
    fps, bits = [], np.zeros((len(smiles), 2048), dtype=np.uint8)
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(str(s))
        fp = MFG.GetFingerprint(m) if m else None
        fps.append(fp)
        if fp is not None:
            arr = np.zeros(2048, dtype=np.int8); DataStructs.ConvertToNumpyArray(fp, arr); bits[i] = arr
    return fps, bits


def cv_select(Xtr, ytr):
    if min(np.bincount(ytr)) < 3:            # can't stratify 3-fold
        return dict(n_estimators=200, num_leaves=31, learning_rate=0.05)
    skf = StratifiedKFold(3, shuffle=True, random_state=0)
    best, best_ap = None, -1
    for ne, nl, lr in GRID:
        clf = lgb.LGBMClassifier(n_estimators=ne, num_leaves=nl, learning_rate=lr,
                                 subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                 min_child_samples=5, min_data_in_bin=1, verbose=-1, n_jobs=2)
        ap = cross_val_score(clf, Xtr, ytr, cv=skf, scoring="average_precision").mean()
        if ap > best_ap:
            best_ap, best = ap, dict(n_estimators=ne, num_leaves=nl, learning_rate=lr)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="588689")
    ap.add_argument("--sizes", type=int, nargs="+", default=[40, 100, 300])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = ap.parse_args()

    df = load(args.target)
    print(f"[{args.target}] {len(df)} compounds; featurizing ECFP...", flush=True)
    fps, bits = fingerprints(df["neut-smiles"].tolist())
    scafs = [MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(str(s))) if Chem.MolFromSmiles(str(s)) else None
             for s in df["neut-smiles"]]
    y_all = df["Active_v2"].to_numpy(dtype=int)
    eval_mask = df["rank"].to_numpy() > 300
    eidx = np.where(eval_mask)[0]
    yE = y_all[eval_mask]; XE = bits[eval_mask]
    fpsE = [fps[i] for i in eidx]
    scafE = [scafs[i] for i in eidx]
    print(f"eval set: {eval_mask.sum()} compounds, {int(yE.sum())} actives", flush=True)

    # CONTROL (item 3): the training pool (top 300) must be disjoint from the eval set.
    assert set(df.iloc[:300].CID) & set(df.loc[eval_mask, "CID"]) == set(), "LEAK: train/eval overlap"
    print(f"[disjointness] train(top300) n=300  eval(>300) n={eval_mask.sum()}  overlap=0  OK", flush=True)

    rows = []
    eval_scores = {"Boltz2_noFT(score)": df.loc[eval_mask, "affinity_probability_binary"].to_numpy(float)}
    nn_pred, lgb_pred = {}, {}
    # reference: the off-the-shelf Boltz-2 ranking on the same eval set
    m = all_metrics(yE, df.loc[eval_mask, "affinity_probability_binary"].to_numpy(float))
    rows.append(dict(method="Boltz2_noFT(score)", n_train=0, seed=-1, **{k: m[k] for k in
                ["auroc", "auprc", "ef_1pct", "bedroc", "n_eval", "n_active"]}))

    for N in args.sizes:
        tr = df.iloc[:N]
        Xtr = bits[:N]; ytr = y_all[:N]
        if min(np.bincount(ytr, minlength=2)) < 1:
            print(f"N={N}: no positive/negative in train, skipping LGBM"); continue
        cfg = cv_select(Xtr, ytr)
        print(f"N={N}: best cfg {cfg}", flush=True)
        preds = []
        for sd in args.seeds:
            clf = lgb.LGBMClassifier(**cfg, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                     min_child_samples=5, min_data_in_bin=1,
                                     random_state=sd, bagging_seed=sd, feature_fraction_seed=sd,
                                     verbose=-1, n_jobs=2)
            clf.fit(Xtr, ytr)
            p = clf.predict_proba(XE)[:, 1]; preds.append(p)
            mm = all_metrics(yE, p)
            rows.append(dict(method="LightGBM_ECFP", n_train=N, seed=sd,
                        **{k: mm[k] for k in ["auroc", "auprc", "ef_1pct", "bedroc", "n_eval", "n_active"]}))
        avg = np.mean(preds, axis=0); lgb_pred[N] = avg
        mm = all_metrics(yE, avg)   # paper-style 5-model average
        rows.append(dict(method="LightGBM_ECFP_avg5", n_train=N, seed=-1,
                    **{k: mm[k] for k in ["auroc", "auprc", "ef_1pct", "bedroc", "n_eval", "n_active"]}))

        # nearest-active Tanimoto (deterministic; ranks eval by max sim to a training active)
        act_fps = [fps[i] for i in range(N) if ytr[i] == 1 and fps[i] is not None]
        if act_fps:
            sim = np.array([max(DataStructs.BulkTanimotoSimilarity(fp, act_fps)) if fp is not None else 0.0
                            for fp in fpsE])
            nn_pred[N] = sim
            mm = all_metrics(yE, sim)
            rows.append(dict(method="NearestActive_Tanimoto", n_train=N, seed=-1,
                        **{k: mm[k] for k in ["auroc", "auprc", "ef_1pct", "bedroc", "n_eval", "n_active"]}))

    out = ROOT / f"results/analysis/cpu_baselines_{args.target}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    res = pd.DataFrame(rows)
    res.to_csv(out, index=False)

    from sklearn.metrics import average_precision_score
    from metrics import bedroc as _bedroc, enrichment_factor as _ef

    # ---- CONTROL: analog bias (held-out actives vs training actives) ----
    analog = []
    for N in args.sizes:
        tafp = [fps[i] for i in range(N) if y_all[i] == 1 and fps[i] is not None]
        tasc = set(scafs[i] for i in range(N) if y_all[i] == 1 and scafs[i])
        if not tafp:
            continue
        for i in eidx:
            if y_all[i] != 1 or fps[i] is None:
                continue
            analog.append(dict(n_train=N, cid=int(df.iloc[i].CID),
                               max_tc=float(max(DataStructs.BulkTanimotoSimilarity(fps[i], tafp))),
                               shares_active_scaffold=int(scafs[i] in tasc)))
    pd.DataFrame(analog).to_csv(ROOT / f"results/analysis/cpu_analog_{args.target}.csv", index=False)

    # ---- CONTROL: stratified RECOVERY from the FULL-eval ranking (paper convention) ----
    # NOT stratum-relative EF. Rank the complete eval set once (ties -> eval-ID/input order via
    # stable sort); recovery = how many of a stratum's actives land in the GLOBAL top-k.
    def recover(score, sub_mask, k):
        order = np.argsort(-score, kind="stable")
        topset = set(order[:k].tolist())
        idx = np.where(sub_mask & (yE == 1))[0]
        return int(sum(i in topset for i in idx))
    k1 = max(1, round(0.01 * len(yE)))
    strat = []
    for N in args.sizes:
        train_scaf = set(scafs[i] for i in range(N) if scafs[i])            # all top-N training compounds
        unseen = np.array([s not in train_scaf for s in scafE])
        lowsim = (nn_pred[N] < 0.3) if N in nn_pred else np.zeros(len(yE), bool)
        methods = {"Boltz2_initial_score": eval_scores["Boltz2_noFT(score)"]}  # score_boltz2 = INITIAL rank, not No-FT
        if N in nn_pred: methods["NearestActive_Tanimoto"] = nn_pred[N]
        if N in lgb_pred: methods["LightGBM_ECFP_avg5"] = lgb_pred[N]
        for stratum, mask in [("full", np.ones(len(yE), bool)),
                              ("unseen_scaffold", unseen), ("lowsim_lt0.3", lowsim)]:
            na = int((mask & (yE == 1)).sum())
            for name, sc in methods.items():
                strat.append(dict(method=name, n_train=N, stratum=stratum, n=int(mask.sum()),
                                  n_active=na, base_rate=float(yE[mask].mean()) if mask.sum() else np.nan,
                                  rec_top100=recover(sc, mask, 100), rec_top1pct=recover(sc, mask, k1)))
    pd.DataFrame(strat).to_csv(ROOT / f"results/analysis/cpu_controls_{args.target}.csv", index=False)

    # ---- bootstrap 95% CIs on full-eval metrics (the paper reports single runs, no CIs) ----
    rng = np.random.default_rng(0); B = 1000
    def boot(score):
        n = len(yE); aps, efs, bes = [], [], []
        for _ in range(B):
            idx = rng.integers(0, n, n); yy = yE[idx]
            if yy.sum() < 1: continue
            ss = score[idx]
            aps.append(average_precision_score(yy, ss)); efs.append(_ef(yy, ss, 0.01)); bes.append(_bedroc(yy, ss))
        q = lambda a: (round(float(np.percentile(a, 2.5)), 4), round(float(np.percentile(a, 97.5)), 4))
        return q(aps), q(efs), q(bes)
    bm = [("Boltz2_initial_score", eval_scores["Boltz2_noFT(score)"], 0)]
    for N in args.sizes:
        if N in nn_pred: bm.append(("NearestActive_Tanimoto", nn_pred[N], N))
        if N in lgb_pred: bm.append(("LightGBM_ECFP_avg5", lgb_pred[N], N))
    ci = []
    for name, sc, N in bm:
        (apl, aph), (efl, efh), (bel, beh) = boot(sc)
        ci.append(dict(method=name, n_train=N, ap_lo=apl, ap_hi=aph, ef1_lo=efl, ef1_hi=efh, bedroc_lo=bel, bedroc_hi=beh))
    pd.DataFrame(ci).to_csv(ROOT / f"results/analysis/cpu_ci_{args.target}.csv", index=False)
    print(f"wrote analog ({len(analog)}), stratified recovery ({len(strat)}), bootstrap CIs ({len(ci)})")

    pd.set_option("display.width", 160)
    print("\n=== summary (per-seed averaged for LightGBM) ===")
    agg = (res[res.seed >= 0].groupby(["method", "n_train"])[["auprc", "ef_1pct", "bedroc"]]
           .agg(["mean", "std"]).round(4)) if (res.seed >= 0).any() else None
    print(res[res.seed < 0][["method", "n_train", "auprc", "ef_1pct", "bedroc"]].round(4).to_string(index=False))
    if agg is not None:
        print("\nLightGBM multi-seed spread:"); print(agg.to_string())
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

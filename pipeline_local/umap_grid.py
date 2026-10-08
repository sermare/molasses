#!/usr/bin/env python3
"""UMAP grid of the affinity head's embeddings: No-FT against head-FT, three replicates each, five colourings.
    <umap env python> pipeline_local/umap_grid.py <target> [--synthetic]       -> results/analysis/<t>/latent/umap_grid.png (+ umap_coords.csv)

Embedding = the 384-dim post-MLP features of the affinity head (mean of the two ensemble modules), for the latent-analysis sample of the target (the 300 training compounds, all evaluation
actives, 2,500 random evaluation inactives and the top-1% picks of both models). UMAP: cosine metric, 30 neighbours, min_dist 0.25.
Columns (6): No-FT replicates 1-3 and head-FT (N=300) seeds 0-2.  No-FT is ONE checkpoint, so its three 'replicates' are three UMAP random seeds on the same embedding (a check that the
layout is stable); the head-FT columns are three independently seeded fine-tunings, each with its own UMAP (random seed = replicate number). Layouts are NOT comparable point-for-point across columns.
Rows: 1 label (training / evaluation, active / inactive); 2 the arm's own affinity probability; 3 variance of the probability across the three head-FT seeds (same colouring in all columns, so
No-FT panels show where the fine-tuned models disagree); 4 Tanimoto distance (1 - max ECFP4 Tanimoto) to the 300 training compounds (self excluded for training compounds);
5 docking site = k-means cluster of the ligand centroid in the target's reference frame (pipeline_local/pose_sites.py).
Under each panel: AUC = cross-validated ROC AUC (5-fold, 15-nearest-neighbour on the two UMAP coordinates, evaluation compounds only, sample-weighted) of separating the classes named in the row label; 0.5 = the layout does not separate them.
--synthetic draws random features (for checking the layout only; the output is stamped SYNTHETIC)."""
import sys, argparse
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
import numpy as np, pandas as pd
import matplotlib as mpl; mpl.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bft_common as bc, bft_latent as lat

ap = argparse.ArgumentParser(); ap.add_argument("target"); ap.add_argument("--synthetic", action="store_true"); ap.add_argument("--neighbors", type=int, default=30); ap.add_argument("--min-dist", type=float, default=0.25); a = ap.parse_args()
t = a.target; bc.style(); BLK = "#000000"; D = lat.make_sample(t); n = len(D); SH = bc.SHORT[t]
ARMS = ["noft", "ft300s0", "ft300s1", "ft300s2"]; feats, probs = {}, {}
if a.synthetic:
    rng = np.random.default_rng(0); y = D.label.values.astype(float)
    for k, arm in enumerate(ARMS): feats[arm] = np.abs(rng.normal(size=(n, 384)) * .05 + (k > 0) * y[:, None] * .02); probs[arm] = np.clip(.1 + .4 * y + rng.normal(size=n) * .1, 0, 1)
else:
    for arm in ARMS:
        L = lat.load_latent(t, arm)
        if L is None: sys.exit(f"{arm} is not extracted yet for {t}")
        pos = pd.Series(np.arange(len(L["ids"])), index=[str(i) for i in L["ids"]]).reindex(D.id).values
        feats[arm] = L["g"].mean(1)[pos]; probs[arm] = (1 / (1 + np.exp(-L["logit"]))).mean(1)[pos]
var_seeds = np.var(np.stack([probs[x] for x in ARMS[1:]]), axis=0)
# Tanimoto distance to the training compounds
E = lat.ecfp_matrix(D.smiles.values).astype(bool); tr = D.is_train.values; inter = E.astype(np.float32) @ E[tr].T.astype(np.float32); cnt = E.sum(1)[:, None] + E[tr].sum(1)[None, :] - inter
tc = np.where(cnt > 0, inter / np.maximum(cnt, 1), 0.0); tri = np.where(tr)[0]; tc[tri, np.arange(len(tri))] = -1; tdist = 1 - tc.max(1)
S = pd.read_csv(bc.AN / t / "latent" / "pose_sites.csv").set_index("id").site.reindex(D.id).values if (bc.AN / t / "latent" / "pose_sites.csv").exists() else np.full(n, np.nan)
import umap
cols = [("No-FT", "noft", 0), ("No-FT", "noft", 1), ("No-FT", "noft", 2), ("head-FT", "ft300s0", 0), ("head-FT", "ft300s1", 1), ("head-FT", "ft300s2", 2)]
coords = {}
for j, (name, arm, rs) in enumerate(cols):
    coords[j] = umap.UMAP(n_neighbors=a.neighbors, min_dist=a.min_dist, metric="cosine", random_state=rs, n_jobs=1).fit_transform(feats[arm]); print("umap", j, flush=True)
def cv_auc(P, z, m, w):
    """5-fold cross-validated AUC of a 15-nearest-neighbour classifier on the 2-D UMAP coordinates, evaluation compounds only (weighted); nan if a class is too small."""
    z = np.asarray(z)[m]; X = P[m]; w = w[m]
    if min(z.sum(), (~z).sum()) < 20: return np.nan
    pr = np.zeros(len(z))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=0).split(X, z): pr[b] = KNeighborsClassifier(15).fit(X[a], z[a]).predict_proba(X[b])[:, 1]
    return roc_auc_score(z, pr, sample_weight=w)
EV = (~D.is_train.values) & (D.w.values > 0); W = D.w.values; q75 = lambda v: v > np.quantile(v[EV], .75)
g = D.group.values; yy = D.label.values; pick_only = (D.w.values == 0) & (~D.is_train.values)
fig, axes = plt.subplots(5, 6, figsize=(24, 19), squeeze=False)
SITE_COL = ["#2a78d6", "#eb6834", "#1baf7a", "#8e44ad", "#c9a227"]
for j, (name, arm, rs) in enumerate(cols):
    P = coords[j]
    sv = S if np.isnan(S).all() else np.where(np.isnan(S), 0, S).astype(int); top = pd.Series(sv[EV]).value_counts().index[0] if not np.isnan(S).all() else None
    tgt = [D.label.values.astype(bool), q75(probs[arm]), q75(var_seeds), q75(tdist), (sv != top) if top is not None else np.zeros(n, bool)]
    AUC = [cv_auc(P, tgt[r], EV, W) for r in range(5)]
    for r in range(5):
        ax = axes[r, j]; ax.set_xticks([]); ax.set_yticks([])
        if r == 0:
            ax.set_title(f"{name}, replicate {j % 3 + 1}\n" + ("(UMAP seed %d on the same embedding)" % rs if name == "No-FT" else "(fine-tuning seed %d)" % rs), fontsize=10)
            m = (g == "eval inactive"); ax.scatter(P[m, 0], P[m, 1], s=3, c="#cccccc", alpha=.5, label="evaluation inactive")
            m = (g == "train") & (yy == 0); ax.scatter(P[m, 0], P[m, 1], s=8, c="#9ecae1", label="training inactive")
            m = (g == "eval active"); ax.scatter(P[m, 0], P[m, 1], s=6, c="#eb6834", alpha=.75, label="evaluation active")
            m = (g == "train") & (yy == 1); ax.scatter(P[m, 0], P[m, 1], s=12, c="#2a78d6", edgecolors=BLK, linewidths=.3, label="training active")
            if j == 0: ax.legend(fontsize=7, markerscale=2, loc="best")
        elif r == 1:
            im = ax.scatter(P[:, 0], P[:, 1], s=3, c=probs[arm], cmap="viridis", vmin=0, vmax=1)
            if j == 5: plt.colorbar(im, ax=axes[1, :].tolist(), fraction=.012, pad=.01, label="probability (the arm's own)")
        elif r == 2:
            im = ax.scatter(P[:, 0], P[:, 1], s=3, c=var_seeds, cmap="magma_r", vmin=0, vmax=np.quantile(var_seeds, .99))
            if j == 5: plt.colorbar(im, ax=axes[2, :].tolist(), fraction=.012, pad=.01, label="variance across 3 seeds")
        elif r == 3:
            im = ax.scatter(P[:, 0], P[:, 1], s=3, c=tdist, cmap="cividis_r", vmin=np.quantile(tdist, .01), vmax=np.quantile(tdist, .99))
            if j == 5: plt.colorbar(im, ax=axes[3, :].tolist(), fraction=.012, pad=.01, label="Tanimoto distance")
        else:
            m = np.isnan(S); ax.scatter(P[m, 0], P[m, 1], s=3, c="#dddddd")
            for k in sorted(set(S[~m].astype(int))): mk = (S == k); ax.scatter(P[mk, 0], P[mk, 1], s=3, c=SITE_COL[(k - 1) % 5], label=f"site {k}")
            if j == 0: ax.legend(fontsize=7, markerscale=2, loc="best")
        if j == 0: ax.set_ylabel(["label\n(active vs inactive)", "affinity probability\n(top quartile vs rest)", "variance across seeds\n(top quartile vs rest)", "Tanimoto distance\n(top quartile vs rest)", "docking site\n(minority vs main site)"][r], fontsize=10)
        ax.grid(False); ax.set_xlabel("AUC = " + ("n/a" if np.isnan(AUC[r]) else f"{AUC[r]:.2f}"), fontsize=9)
stamp = "  [SYNTHETIC FEATURES: layout check only]" if a.synthetic else ""
fig.subplots_adjust(top=0.93, hspace=0.2, wspace=0.08)
fig.suptitle(f"{SH}: UMAP of the affinity head's 384-dim post-MLP features, No-FT (left three) and head-FT N=300 (right three); sample of {n:,} compounds{stamp}", fontsize=13, y=0.97)
out = bc.AN / t / "latent" / ("umap_grid_SYNTHETIC.png" if a.synthetic else "umap_grid.png"); plt.savefig(out, dpi=110, bbox_inches="tight"); print("wrote", out)
if not a.synthetic:
    pd.DataFrame({"id": D.id, **{f"u{j}_{ax_}": coords[j][:, k] for j in range(6) for k, ax_ in enumerate(("x", "y"))}}).to_csv(bc.AN / t / "latent" / "umap_coords.csv", index=False)

#!/usr/bin/env python3
"""One figure per README finding -> notebooks/figures/finding_<n>_<name>.png. Matplotlib only, all text black, no bold, white background.
Re-run after more targets finish; targets without data are skipped. Usage: python make_readme_figures.py"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/global/scratch/users/sergiomar10/boltzaff/pipeline_local")
import numpy as np, pandas as pd, matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
import bft_common as bc, bft_rerank as br
bc.style(); OUT = bc.ROOT / "notebooks/figures"; OUT.mkdir(exist_ok=True)
BLK, OUR, THEIR, NEU = bc.BLK, bc.OUR, bc.THEIR, bc.NEU
DONE = bc.done_targets("scores"); SH = bc.SHORT
print("targets:", [SH[t] for t in DONE])

def save(fig, name):
    fig.tight_layout(); fig.savefig(OUT / name, dpi=130); plt.close(fig); print("wrote", name)

# 1. replication: AP, paper vs ours, No-FT and head-FT N=300
rows = []
for t in DONE:
    d = bc.per_seed(t)
    if d is None: continue
    b = d[d.arm == "base"].ap.iloc[0]; f = d[(d.arm == "headft") & (d.n_train == 300)].ap.values
    rows.append((t, bc.PAPER_BASE.loc[t, "auprc"], b, bc.PAPER_FT.loc[(t, 300), "auprc"], f))
fig, ax = plt.subplots(figsize=(7.5, 4.2)); x = np.arange(len(rows)); w = 0.2
for k, (lab, col, i) in enumerate([("paper No-FT", "#f2b999", 1), ("ours No-FT", "#9ecae1", 2), ("paper head-FT N=300", THEIR, 3), ("ours head-FT N=300 (5 seeds)", OUR, 4)]):
    vals = [r[i] if i < 4 else r[4].mean() for r in rows]; ax.bar(x + (k - 1.5) * w, vals, w, color=col, label=lab)
for j, r in enumerate(rows):
    ax.errorbar(j + 1.5 * w, r[4].mean(), yerr=r[4].std(ddof=1), color=BLK, capsize=3); ax.scatter(np.full(len(r[4]), j + 1.5 * w), r[4], s=8, color=BLK, zorder=3)
    ax.text(j + 1.5 * w, r[4].max() + 0.008, f"x{r[4].mean() / r[2]:.2f}", ha="center", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels([SH[r[0]] for r in rows]); ax.set_ylabel("average precision"); ax.set_ylim(0, 0.33); ax.set_title("Replication: average precision, paper and ours (x = head-FT / No-FT)"); ax.legend(fontsize=8, ncol=2, loc="upper center")
save(fig, "finding_1_replication.png")

# 2. top 1% of the library as confusion matrices (rows: truth, columns: flagged in the top 1% or not)
fig, axes = plt.subplots(len(DONE), 2, figsize=(9.5, 4.6 * len(DONE)), squeeze=False)
for i, t in enumerate(DONE):
    S = bc.scores(t); k = int(np.ceil(0.01 * len(S))); ft = bc.ft_mean(S, 300); na = int(S.label.sum()); nn = len(S) - na
    for j, (nm, sc) in enumerate((("No-FT", S.noft_p), ("head-FT (5-seed ensemble, N=300)", ft))):
        top = S.loc[sc.sort_values(ascending=False).index[:k]]; tp = int(top.label.sum()); fp = k - tp; fn = na - tp; tn = nn - fp
        M = np.array([[tp, fn], [fp, tn]]); R = M / M.sum(axis=1, keepdims=True); ax = axes[i, j]
        ax.imshow(R, cmap="Blues", vmin=0, vmax=1.0)
        for r in range(2):
            for c in range(2): ax.text(c, r, f"{M[r, c]:,}\n({100 * R[r, c]:.1f}% of row)", ha="center", va="center", fontsize=10, color="white" if R[r, c] > 0.55 else BLK)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["flagged\n(top 1%)", "not flagged"]); ax.set_yticks([0, 1]); ax.set_yticklabels([f"active\n(n={na})", f"inactive\n(n={nn:,})"]); ax.grid(False)
        ax.set_title(f"{SH[t]}: {nm}\nrecall {100 * tp / na:.0f}%, precision {100 * tp / k:.0f}%", fontsize=10)
fig.suptitle("Top 1% of the library flagged by No-FT and by head-FT (confusion matrices)"); save(fig, "finding_2_top1pct.png")

# 3. the gain is concentrated in actives that resemble the training actives, with example molecules (3 per bin, closest to the bin's median rank gain)
from rdkit import Chem
from rdkit.Chem import Draw
labs = ["< 0.3", "0.3 - 0.5", ">= 0.5"]; w = 0.35; BIN = {}
for t in DONE:
    S = bc.scores(t); S["ft"] = bc.ft_mean(S, 300); S["rn"] = S.noft_p.rank(ascending=False, method="first"); S["rf"] = S.ft.rank(ascending=False, method="first")
    S["rshift"] = np.log2(S.rn / S.rf); tt = br.sim_table(t); tr = bc.train_ids(t, 300); tra = [i for i in tr if bool(tt.target_active_v2.get(i, False))]
    ref = [f for f in br.fps(tt["neut-smiles"].reindex(tra).values) if f is not None]; A = S[S.label == 1].copy(); A["tc"] = br.max_sim(A.smiles.values, ref)
    A["bin"] = pd.cut(A.tc, [0, 0.3, 0.5, 1.01], labels=labs, right=False); BIN[t] = (A, A.groupby("bin", observed=True).rshift.agg(["median", "size"]), len(ref))
fig = plt.figure(figsize=(16, 4.6 + 2.0 * len(DONE))); gs = fig.add_gridspec(1 + len(DONE), 9, height_ratios=[3.4] + [1.0] * len(DONE)); ax = fig.add_subplot(gs[0, :])
for q, t in enumerate(DONE):
    A, g, nref = BIN[t]; ax.bar(np.arange(3) + (q - 0.5) * w, 2 ** g["median"].values, w, color=[OUR, THEIR][q % 2], label=f"{SH[t]} ({nref} training actives)")
    for i_, (m, n) in enumerate(zip(g["median"].values, g["size"].values)): ax.text(i_ + (q - 0.5) * w, 2 ** m + 0.15, f"n={n}", ha="center", fontsize=8)
ax.axhline(1, color=BLK, lw=0.8); ax.set_xlim(-0.5, 2.5); ax.set_xticks(range(3)); ax.set_xticklabels(labs); ax.set_xlabel("active's highest Tanimoto to a training active"); ax.set_ylabel("median rank gain under head-FT (fold)"); ax.legend(fontsize=8, loc="upper left")
ax.set_title("Head-FT moves up actives that resemble the training actives (below: three example actives per bin, closest to each bin's median rank gain)", fontsize=11)
for r, t in enumerate(DONE):
    A, g, _ = BIN[t]
    for b, lab in enumerate(labs):
        sub = A[A.bin == lab]
        if sub.empty: continue
        med = g.loc[lab, "median"]; pick = sub.iloc[(sub.rshift - med).abs().argsort()[:3]]
        for c, (_, row) in enumerate(pick.iterrows()):
            axm = fig.add_subplot(gs[1 + r, b * 3 + c]); m = Chem.MolFromSmiles(row.smiles)
            if m is not None: axm.imshow(Draw.MolToImage(m, size=(320, 240)))
            axm.axis("off"); axm.set_title(f"{SH[t]}  Tc {row.tc:.2f}, x{2 ** row.rshift:.1f}", fontsize=8)
save(fig, "finding_3_similarity.png")

# 4. near-identical pairs: accuracy by pair type (values from notebook 04, section 3b, 2 targets pooled)
strata = ["active larger\n(n=348)", "same size\n(n=164)", "active smaller\n(n=225)", "property-matched\n(n=58)"]
acc = {"Boltz-2 dataset": [0.71, 0.69, 0.59, 0.66], "No-FT": [0.73, 0.66, 0.59, 0.60], "head-FT N=300": [0.67, 0.62, 0.62, 0.47]}
fig, ax = plt.subplots(figsize=(8.5, 4.2)); w = 0.26
for q, (nm, col) in enumerate(zip(acc, ["#bbbbbb", "#9ecae1", OUR])): ax.bar(np.arange(4) + (q - 1) * w, acc[nm], w, color=col, label=nm)
ax.axhline(0.5, color=BLK, ls="--", lw=0.9); ax.text(3.45, 0.505, "chance", ha="right", fontsize=8); ax.set_ylim(0.3, 0.8); ax.set_xticks(range(4)); ax.set_xticklabels(strata)
ax.set_ylabel("active scored above its inactive twin"); ax.set_title("Near-identical pairs (737, 2 targets): accuracy by size difference\n(source: notebook 04, section 3b)"); ax.legend(fontsize=8, loc="upper right")
save(fig, "finding_4_pairs.png")

# 5. decoy protein (588689): AUROC with bootstrap CI
from sklearn.metrics import roc_auc_score
D = bc.RUNS / "588689/decoy"
if (D / "scores.csv").exists():
    s = pd.read_csv(D / "scores.csv"); sub = pd.read_csv(D / "subset.csv").set_index("id"); w_ = s.pivot(index="id", columns="arm", values="prob").join(sub.target_active_v2.rename("y")).dropna()
    rng = np.random.default_rng(0); res = {}
    for arm in ("real", "other", "shuffled"):
        a = roc_auc_score(w_.y, w_[arm]); bs = [roc_auc_score(w_.y.iloc[i], w_[arm].iloc[i]) for i in (rng.integers(0, len(w_), len(w_)) for _ in range(400))]; res[arm] = (a, *np.percentile(bs, [2.5, 97.5]))
    fig, ax = plt.subplots(figsize=(6, 4.2)); names = {"real": "real protein", "other": "unrelated protein\n(own MSA)", "shuffled": "shuffled sequence\n(no MSA)"}
    for i, arm in enumerate(res): a, lo, hi = res[arm]; ax.bar(i, a, 0.55, color=[OUR, "#9ecae1", "#bbbbbb"][i]); ax.errorbar(i, a, yerr=[[a - lo], [hi - a]], color=BLK, capsize=4); ax.text(i, hi + 0.01, f"{a:.3f}", ha="center", fontsize=9)
    ax.axhline(0.5, color=BLK, ls="--", lw=0.9); ax.set_ylim(0.5, 1.0); ax.set_xticks(range(3)); ax.set_xticklabels([names[a] for a in res]); ax.set_ylabel("AUROC (actives vs inactives)")
    ax.set_title(f"588689: same {len(w_):,} ligands scored against three proteins"); save(fig, "finding_5_decoy.png")

# 6. training strategies (588689): AP ratio to top-N, paired p
V = pd.read_csv(bc.AN / "588689/variants_compare.csv"); V = V[V.metric == "AP"].copy(); V["label"] = V.strategy + " N=" + V.n_train.astype(str)
V = V.sort_values("ratio_vs_top")
fig, ax = plt.subplots(figsize=(8, 4.6)); cols = [("#d62728" if r < 1 else OUR) if p < 0.05 else "#bbbbbb" for r, p in zip(V.ratio_vs_top, V.paired_p)]
ax.barh(V.label, V.ratio_vs_top, color=cols); ax.axvline(1, color=BLK, lw=0.9)
for i, (r, p) in enumerate(zip(V.ratio_vs_top, V.paired_p)): ax.text(r + (0.01 if r >= 1 else -0.01), i, f"x{r:.2f} (p={p:.3f})", va="center", ha="left" if r >= 1 else "right", fontsize=8)
ax.set_xlim(0.5, 1.5); ax.set_xlabel("AP relative to the standard top-N training set (5 seeds, paired)"); ax.set_title("588689: training-set composition. Coloured: p < 0.05 uncorrected;\nnone passes Bonferroni (0.0056 for 9 comparisons)", fontsize=10)
save(fig, "finding_6_strategies.png")

# 7. seeds and ensembling: single-seed AP vs 5-seed ensemble AP at N=300
fig, ax = plt.subplots(figsize=(6.5, 4.2))
for j, t in enumerate(DONE):
    S = bc.scores(t); single = [bc.average_precision(S.label, S[f"ft300_p{s}"]) for s in bc.SEEDS]; ens = bc.average_precision(S.label, bc.ft_mean(S, 300))
    ax.scatter(np.full(5, j) + np.linspace(-0.12, 0.12, 5), single, color=OUR, s=22, label="single seeds" if j == 0 else None); ax.hlines(ens, j - 0.25, j + 0.25, color=BLK, lw=2, label="5-seed ensemble" if j == 0 else None)
    ax.text(j + 0.27, ens, f"+{100 * (ens / np.mean(single) - 1):.0f}% vs mean seed", va="center", fontsize=8)
ax.set_xticks(range(len(DONE))); ax.set_xticklabels([SH[t] for t in DONE]); ax.set_xlim(-0.6, len(DONE) - 0.1); ax.set_ylabel("average precision (N=300)"); ax.legend(fontsize=8, loc="lower right"); ax.set_title("Seed-to-seed spread and the gain from ensembling")
save(fig, "finding_7_seeds.png")

# 8. localisation: Boltz-2 structures of the two finished targets, coloured by pLDDT and by pose density (stitched from the PyMOL renders)
import matplotlib.image as mpimg
fig, axes = plt.subplots(len(DONE), 2, figsize=(14, 5.4 * len(DONE)), squeeze=False)
for i, t in enumerate(DONE):
    for j, (v, ti) in enumerate((("plddt", "coloured by pLDDT (dark blue = very confident, orange = very low)"), ("overview", "coloured by pose density (white = rarely touched, red = most contacted)"))):
        p = bc.AN / t / "render" / f"{v}.png"; ax = axes[i, j]; ax.axis("off")
        if p.exists(): ax.imshow(mpimg.imread(p))
        ax.set_title(f"{SH[t]}: {ti}", fontsize=10)
fig.suptitle("Where poses go and where the model is unsure (ray-traced PyMOL renders)"); save(fig, "finding_8_localization.png")

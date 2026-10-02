#!/usr/bin/env python3
"""One figure per README finding -> notebooks/figures/finding_<n>_<name>.png. Matplotlib only, all text black, no bold, white background.
Axes start at 0 wherever 0 is the natural lower limit. Balanced-training arms (N=300) exist for 588689 only, so they are added as an extra
comparison on that target's held-out set (compounds ranked below 1000), where No-FT, top-N and balanced are scored on the same compounds.
Re-run after more targets finish; targets without data are skipped. Usage: python make_readme_figures.py"""
import sys, glob, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/global/scratch/users/sergiomar10/boltzaff/pipeline_local")
import numpy as np, pandas as pd, matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
import bft_common as bc, bft_rerank as br
bc.style(); OUT = bc.ROOT / "notebooks/figures"; OUT.mkdir(exist_ok=True)
BLK, OUR, THEIR, NEU = bc.BLK, bc.OUR, bc.THEIR, bc.NEU; GREEN = "#1baf7a"
DONE = bc.done_targets("scores"); SH = bc.SHORT
print("targets:", [SH[t] for t in DONE])

def save(fig, name):
    fig.tight_layout(); fig.savefig(OUT / name, dpi=130); plt.close(fig); print("wrote", name)

# ---- balanced-training comparison set (588689 only): held-out compounds ranked below 1000 ----
BAL = None; T0 = "588689"
if T0 in DONE and (bc.RUNS / T0 / "headft_variants_scores/balanced_N300_seed0").exists():
    R0 = bc.RUNS / T0; S0 = bc.scores(T0).set_index("id"); E = [l.strip() for l in open(R0 / "ft_inputs_variants/eval_ids.txt") if l.strip()]
    def arm(cond, s):
        f = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(str(R0 / f"headft_variants_scores/{cond}_seed{s}/chunk_*.csv")))]); f["sample_id"] = f.sample_id.astype(str)
        return f.drop_duplicates("sample_id").set_index("sample_id").affinity_probability_binary.reindex(E).values
    BAL = dict(E=E, S=S0.loc[E], y=S0.loc[E, "label"].values.astype(int), noft=S0.loc[E, "noft_p"].values,
               top=np.array([arm("top_N300", s) for s in bc.SEEDS]), bal=np.array([arm("balanced_N300", s) for s in bc.SEEDS]))
    print("balanced comparison set:", len(E), "compounds,", int(BAL["y"].sum()), "actives")
else: print("no balanced arms available")
AP = average_precision_score

# 1. replication: AP, paper vs ours, No-FT and head-FT N=300 (+ balanced on 588689's held-out set)
rows = []
for t in DONE:
    d = bc.per_seed(t)
    if d is None: continue
    rows.append((t, bc.PAPER_BASE.loc[t, "auprc"], d[d.arm == "base"].ap.iloc[0], bc.PAPER_FT.loc[(t, 300), "auprc"], d[(d.arm == "headft") & (d.n_train == 300)].ap.values))
fig, axes = plt.subplots(1, 2 if BAL else 1, figsize=(13 if BAL else 7.5, 4.4), gridspec_kw=dict(width_ratios=[2, 1.15]) if BAL else None); axes = np.atleast_1d(axes); ax = axes[0]
x = np.arange(len(rows)); w = 0.2
for k, (lab, col, i) in enumerate([("paper No-FT", "#f2b999", 1), ("ours No-FT", "#9ecae1", 2), ("paper head-FT N=300", THEIR, 3), ("ours head-FT N=300 (5 seeds)", OUR, 4)]):
    ax.bar(x + (k - 1.5) * w, [r[i] if i < 4 else r[4].mean() for r in rows], w, color=col, label=lab)
for j, r in enumerate(rows):
    ax.errorbar(j + 1.5 * w, r[4].mean(), yerr=r[4].std(ddof=1), color=BLK, capsize=3); ax.scatter(np.full(len(r[4]), j + 1.5 * w), r[4], s=8, color=BLK, zorder=3)
    ax.text(j + 1.5 * w, r[4].max() + 0.008, f"x{r[4].mean() / r[2]:.2f}", ha="center", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels([SH[r[0]] for r in rows]); ax.set_ylabel("average precision"); ax.set_ylim(0, 0.33); ax.set_title("Main evaluation set (x = head-FT / No-FT)", fontsize=10); ax.legend(fontsize=8, ncol=2, loc="upper center")
if BAL:
    ax = axes[1]; a_top = [AP(BAL["y"], p) for p in BAL["top"]]; a_bal = [AP(BAL["y"], p) for p in BAL["bal"]]; a_no = AP(BAL["y"], BAL["noft"])
    for i, (vals, col) in enumerate((([a_no], "#9ecae1"), (a_top, OUR), (a_bal, GREEN))):
        m = np.mean(vals); ax.bar(i, m, 0.6, color=col)
        if len(vals) > 1: ax.errorbar(i, m, yerr=np.std(vals, ddof=1), color=BLK, capsize=3); ax.scatter(np.full(len(vals), i), vals, s=9, color=BLK, zorder=3)
        ax.text(i, max(vals) + 0.008, f"{m:.3f}" + (f"\nx{m / a_no:.2f}" if i else ""), ha="center", fontsize=9)
    ax.set_xticks(range(3)); ax.set_xticklabels(["No-FT", "head-FT\ntop-N (N=300)", "head-FT\nbalanced (N=300)"]); ax.set_ylim(0, 0.33); ax.set_ylabel("average precision")
    ax.set_title(f"588689 held-out set (rank > 1000, n={len(BAL['y']):,}):\nbalanced training, 5 seeds", fontsize=10)
fig.suptitle("Replication: average precision, paper and ours"); save(fig, "finding_1_replication.png")

# 2. top 1% as confusion matrices (rows truth, columns flagged or not), No-FT against head-FT top-N for each finished target
def conf_ax(ax, y, sc, title):
    k = int(np.ceil(0.01 * len(y))); na = int(y.sum()); nn = len(y) - na; order = np.argsort(-sc, kind="stable")[:k]; tp = int(y[order].sum()); fp = k - tp
    M = np.array([[tp, na - tp], [fp, nn - fp]]); R = M / M.sum(axis=1, keepdims=True); ax.imshow(R, cmap="Blues", vmin=0, vmax=1.0)
    for r in range(2):
        for c in range(2): ax.text(c, r, f"{M[r, c]:,}\n({100 * R[r, c]:.1f}%)", ha="center", va="center", fontsize=9, color="white" if R[r, c] > 0.55 else BLK)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["flagged\n(top 1%)", "not flagged"], fontsize=8); ax.set_yticks([0, 1]); ax.set_yticklabels([f"active\n(n={na})", f"inactive\n(n={nn:,})"], fontsize=8); ax.grid(False)
    ax.set_title(f"{title}\nrecall {100 * tp / na:.0f}%, precision {100 * tp / k:.0f}%", fontsize=9)
nrow = len(DONE); fig, axes = plt.subplots(nrow, 2, figsize=(9.5, 4.3 * nrow), squeeze=False)
for i, t in enumerate(DONE):
    S = bc.scores(t); y = S.label.values.astype(int)
    conf_ax(axes[i, 0], y, S.noft_p.values, f"{SH[t]} (main set): No-FT"); conf_ax(axes[i, 1], y, bc.ft_mean(S, 300).values, f"{SH[t]} (main set): head-FT top-N, 5-seed ensemble")
fig.suptitle("Top 1% of the library flagged (confusion matrices; cells: count and share of the row)"); save(fig, "finding_2_top1pct.png")

# 3. rank gain by similarity to the training actives (bootstrap CI of the median); one series per finished target, then example molecules (top-N, 3 per bin, closest to the bin's median rank gain)
from rdkit import Chem
from rdkit.Chem import Draw
labs = ["< 0.3", "0.3 - 0.5", ">= 0.5"]
def gain_bins(smiles, y, p_ref, p_mod, ref_fps, B=1000):
    rn = pd.Series(p_ref).rank(ascending=False, method="first").values; rf = pd.Series(p_mod).rank(ascending=False, method="first").values; sh = np.log2(rn / rf)
    tc = br.max_sim(smiles, ref_fps); b = pd.cut(pd.Series(tc), [0, 0.3, 0.5, 1.01], labels=labs, right=False)
    d = pd.DataFrame({"tc": tc, "sh": sh, "bin": b})[y == 1]; rng = np.random.default_rng(0); rows = []
    for lab in labs:
        v = d[d.bin == lab].sh.values
        if len(v) == 0: rows.append((lab, np.nan, 0, np.nan, np.nan)); continue
        bs = np.median(v[rng.integers(0, len(v), (B, len(v)))], axis=1); rows.append((lab, np.median(v), len(v), *np.percentile(bs, [2.5, 97.5])))
    return d, pd.DataFrame(rows, columns=["bin", "median", "size", "lo", "hi"]).set_index("bin")
def train_fps(t, ids):
    tt = br.sim_table(t); tra = [i for i in ids if bool(tt.target_active_v2.get(i, False))]; return [f for f in br.fps(tt["neut-smiles"].reindex(tra).values) if f is not None]
SER = []; BIN = {}
for q, t in enumerate(DONE):
    S = bc.scores(t); S['rn'] = S.noft_p.rank(ascending=False, method='first'); S['rf'] = bc.ft_mean(S, 300).rank(ascending=False, method='first'); ref = train_fps(t, bc.train_ids(t, 300)); d, g = gain_bins(S.smiles.values, S.label.values.astype(int), S.noft_p.values, bc.ft_mean(S, 300).values, ref)
    SER.append((f"{SH[t]} head-FT top-N, main set ({len(ref)} training actives)", g, [OUR, THEIR, "#8e44ad", "#2a9d8f"][q % 4])); BIN[t] = (S, d, g)
fig = plt.figure(figsize=(16, 4.8 + 2.0 * len(DONE))); gs = fig.add_gridspec(1 + len(DONE), 9, height_ratios=[3.6] + [1.0] * len(DONE)); ax = fig.add_subplot(gs[0, :]); w = 0.8 / len(SER)
for q, (lab, g, col) in enumerate(SER):
    xs = np.arange(len(g)) + (q - (len(SER) - 1) / 2) * w; m = 2 ** g["median"].values; lo = 2 ** g["lo"].values; hi = 2 ** g["hi"].values
    ax.bar(xs, m, w, color=col, label=lab); ax.errorbar(xs, m, yerr=[m - lo, hi - m], fmt="none", color=BLK, capsize=3, lw=1)
    for xx, mm, h, n in zip(xs, m, hi, g["size"].values): ax.text(xx, h * 1.02 + 0.2, f"n={n}", ha="center", fontsize=7)
ax.axhline(1, color=BLK, lw=0.8); ax.set_xlim(-0.5, 2.5); ax.set_ylim(0, None); ax.set_xticks(range(3)); ax.set_xticklabels(labs); ax.set_xlabel("active's highest Tanimoto to a training active (of that model's training set)"); ax.set_ylabel("median rank gain (fold)\n95% bootstrap CI over actives"); ax.legend(fontsize=8, loc="upper left")
ax.set_title("Head-FT moves up actives that resemble the training actives (below: three example actives per bin, closest to each bin's median rank gain, top-N model; titles: Tanimoto and rank under No-FT -> under head-FT)", fontsize=10)
for r, t in enumerate(DONE):
    S, d, g = BIN[t]; g = g.assign(median=g['median']); A = S[S.label == 1].copy(); A["tc"] = d.tc.values; A["bin"] = d.bin.values; A["rshift"] = d.sh.values
    for b, lab in enumerate(labs):
        sub = A[A.bin == lab]
        if sub.empty: continue
        pick = sub.iloc[(sub.rshift - g.loc[lab, "median"]).abs().argsort()[:3]]
        for c, (_, row) in enumerate(pick.iterrows()):
            axm = fig.add_subplot(gs[1 + r, b * 3 + c]); m = Chem.MolFromSmiles(row.smiles)
            if m is not None: axm.imshow(Draw.MolToImage(m, size=(320, 240)))
            axm.axis("off"); axm.set_title(f"{SH[t]}  Tc {row.tc:.2f}\nrank {int(row.rn):,} -> {int(row.rf):,}", fontsize=8)
save(fig, "finding_3_similarity.png")

# 4. near-identical pairs: accuracy by size difference (notebook 04, section 3b) and, on 588689's held-out set, No-FT vs top-N vs balanced
strata = ["active larger\n(n=691)", "same size\n(n=293)", "active smaller\n(n=561)", "property-matched\n(n=116)"]
acc = {"Boltz-2 dataset": [0.70, 0.66, 0.68, 0.59], "No-FT": [0.71, 0.67, 0.67, 0.61], "head-FT N=300": [0.66, 0.68, 0.66, 0.58]}
fig, axes = plt.subplots(1, 2 if BAL else 1, figsize=(14 if BAL else 8.5, 4.4), gridspec_kw=dict(width_ratios=[2.2, 1]) if BAL else None); axes = np.atleast_1d(axes); ax = axes[0]; w = 0.26
for q, (nm, col) in enumerate(zip(acc, ["#bbbbbb", "#9ecae1", OUR])): ax.bar(np.arange(4) + (q - 1) * w, acc[nm], w, color=col, label=nm)
ax.axhline(0.5, color=BLK, ls="--", lw=0.9); ax.text(3.45, 0.51, "chance", ha="right", fontsize=8); ax.set_ylim(0, 0.85); ax.set_xticks(range(4)); ax.set_xticklabels(strata)
ax.set_ylabel("active scored above its inactive twin"); ax.set_title("1,545 pairs, 4 targets: by size difference (notebook 04, section 3b)", fontsize=10); ax.legend(fontsize=8, loc="upper right")
if BAL:
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    P = pd.read_csv(bc.AN / f"{T0}/cliffs/pairs.csv"); ids = pd.Index(sorted(set(P.id_a) | set(P.id_b))); n = len(ids)
    _, comp = connected_components(coo_matrix((np.ones(len(P)), (ids.get_indexer(P.id_a), ids.get_indexer(P.id_b))), shape=(n, n)), directed=False); ser = dict(zip(ids, comp))
    pos = pd.Series(np.arange(len(BAL["E"])), index=BAL["E"]); C = P[(P.kind == "cliff") & P.id_a.isin(pos.index) & P.id_b.isin(pos.index)].copy(); C["series"] = C.id_a.map(ser)
    sc = {"No-FT": BAL["noft"], "head-FT top-N": BAL["top"].mean(0), "head-FT balanced": BAL["bal"].mean(0)}; rng = np.random.default_rng(0); groups = [g for _, g in C.groupby("series")]
    def accuracy(df, s): a = s[pos[df.id_a].values]; b = s[pos[df.id_b].values]; return np.mean((a > b) + 0.5 * (a == b))
    for i, (nm, col) in enumerate(zip(sc, ["#9ecae1", OUR, GREEN])):
        a = accuracy(C, sc[nm]); bs = [accuracy(pd.concat([groups[j] for j in rng.integers(0, len(groups), len(groups))]), sc[nm]) for _ in range(300)]; lo, hi = np.percentile(bs, [2.5, 97.5])
        axes[1].bar(i, a, 0.6, color=col); axes[1].errorbar(i, a, yerr=[[a - lo], [hi - a]], color=BLK, capsize=4); axes[1].text(i, hi + 0.02, f"{a:.2f}", ha="center", fontsize=9)
    axes[1].axhline(0.5, color=BLK, ls="--", lw=0.9); axes[1].set_ylim(0, 0.85); axes[1].set_xticks(range(3)); axes[1].set_xticklabels(["No-FT", "head-FT\ntop-N", "head-FT\nbalanced"]); axes[1].set_ylabel("active scored above its inactive twin")
    axes[1].set_title(f"588689 pairs inside the held-out set\n({len(C)} cliff pairs, {len(groups)} series; 95% CI over series)", fontsize=10)
save(fig, "finding_4_pairs.png")

# 5. decoy protein (588689): AUROC with bootstrap CI (stock Boltz-2 affinity head only; fine-tuned heads were not run on the decoys)
from sklearn.metrics import roc_auc_score
D = bc.RUNS / "588689/decoy"
if (D / "scores.csv").exists():
    s = pd.read_csv(D / "scores.csv"); sub = pd.read_csv(D / "subset.csv").set_index("id"); w_ = s.pivot(index="id", columns="arm", values="prob").join(sub.target_active_v2.rename("y")).dropna()
    rng = np.random.default_rng(0); res = {}
    for arm_ in ("real", "other", "shuffled"):
        a = roc_auc_score(w_.y, w_[arm_]); bs = [roc_auc_score(w_.y.iloc[i], w_[arm_].iloc[i]) for i in (rng.integers(0, len(w_), len(w_)) for _ in range(400))]; res[arm_] = (a, *np.percentile(bs, [2.5, 97.5]))
    fig, ax = plt.subplots(figsize=(6.4, 4.4)); names = {"real": "real protein", "other": "unrelated protein\n(own MSA)", "shuffled": "shuffled sequence\n(no MSA)"}
    for i, arm_ in enumerate(res): a, lo, hi = res[arm_]; ax.bar(i, a, 0.55, color=[OUR, "#9ecae1", "#bbbbbb"][i]); ax.errorbar(i, a, yerr=[[a - lo], [hi - a]], color=BLK, capsize=4); ax.text(i, hi + 0.015, f"{a:.3f}", ha="center", fontsize=9)
    ax.axhline(0.5, color=BLK, ls="--", lw=0.9); ax.text(2.45, 0.51, "chance", ha="right", fontsize=8); ax.set_ylim(0, 1.05); ax.set_xticks(range(3)); ax.set_xticklabels([names[a] for a in res]); ax.set_ylabel("AUROC (actives vs inactives)")
    ax.set_title(f"588689: same {len(w_):,} ligands scored against three proteins\n(stock Boltz-2 affinity head; fine-tuned heads not run on decoys)", fontsize=10); save(fig, "finding_5_decoy.png")

# 6. training strategies (588689): AP relative to top-N, paired p (x axis from 0)
V = pd.read_csv(bc.AN / "588689/variants_compare.csv"); V = V[V.metric == "AP"].copy(); V["label"] = V.strategy + " N=" + V.n_train.astype(str); V = V.sort_values("ratio_vs_top")
fig, ax = plt.subplots(figsize=(8.5, 4.8)); cols = [(("#d62728" if r < 1 else GREEN) if p < 0.05 else "#bbbbbb") for r, p in zip(V.ratio_vs_top, V.paired_p)]
ax.barh(V.label, V.ratio_vs_top, color=cols); ax.axvline(1, color=BLK, lw=0.9)
for i, (r, p) in enumerate(zip(V.ratio_vs_top, V.paired_p)): ax.text(r + 0.02, i, f"x{r:.2f} (p={p:.3f})", va="center", ha="left", fontsize=8)
ax.set_xlim(0, 1.7); ax.set_xlabel("AP relative to the standard top-N training set (5 seeds, paired; 1 = no change)"); ax.set_title("588689: training-set composition. Coloured: p < 0.05 uncorrected;\nnone passes Bonferroni (0.0056 for 9 comparisons)", fontsize=10)
save(fig, "finding_6_strategies.png")

# 7. seeds and ensembling: single-seed AP vs 5-seed ensemble AP at N=300 (y axis from 0)
fig, ax = plt.subplots(figsize=(6.5, 4.4))
for j, t in enumerate(DONE):
    S = bc.scores(t); single = [bc.average_precision(S.label, S[f"ft300_p{s}"]) for s in bc.SEEDS]; ens = bc.average_precision(S.label, bc.ft_mean(S, 300))
    ax.scatter(np.full(5, j) + np.linspace(-0.12, 0.12, 5), single, color=OUR, s=22, label="single seeds" if j == 0 else None); ax.hlines(ens, j - 0.25, j + 0.25, color=BLK, lw=2, label="5-seed ensemble" if j == 0 else None)
    ax.text(j + 0.27, ens, f"+{100 * (ens / np.mean(single) - 1):.0f}% vs mean seed", va="center", fontsize=8)
ax.set_ylim(0, 0.27); ax.set_xticks(range(len(DONE))); ax.set_xticklabels([SH[t] for t in DONE]); ax.set_xlim(-0.6, len(DONE) - 0.1); ax.set_ylabel("average precision (N=300)"); ax.legend(fontsize=8, loc="lower right"); ax.set_title("Seed-to-seed spread and the gain from ensembling")
save(fig, "finding_7_seeds.png")

# 8. localisation: Boltz-2 structures of the finished targets, coloured by pLDDT and by pose density (stitched from the PyMOL renders)
import matplotlib.image as mpimg
fig, axes = plt.subplots(len(DONE), 2, figsize=(14, 5.4 * len(DONE)), squeeze=False)
for i, t in enumerate(DONE):
    for j, (v, ti) in enumerate((("plddt", "coloured by pLDDT (dark blue = very confident, orange = very low)"), ("overview", "coloured by pose density (white = rarely touched, red = most contacted)"))):
        p = bc.AN / t / "render" / f"{v}.png"; ax = axes[i, j]; ax.axis("off")
        if p.exists(): ax.imshow(mpimg.imread(p))
        ax.set_title(f"{SH[t]}: {ti}", fontsize=10)
fig.suptitle("Where poses go and where the model is unsure (ray-traced PyMOL renders)", y=0.995); fig.tight_layout(rect=(0, 0, 1, 0.975)); save(fig, "finding_8_localization.png")

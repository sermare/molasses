#!/usr/bin/env python3
"""Builds notebooks/08_latent_space.ipynb - ONE self-filling notebook on the latent space of the Boltz-2 affinity head, ordered by priority.
Features come from pipeline_local/extract_latent.py (384-dim post-MLP features 'g' and 128-dim pooled features 'g_raw' of the two ensemble modules, for a per-target sample of
the 300 training compounds, all evaluation actives, 2,500 random evaluation inactives and the top-1% picks), under No-FT and two head-FT seeds (N=300).
Every cell prints 'pending' for anything whose data is not there yet and fills in on re-execution. Matplotlib only, all text black, no bold, axes from 0 where natural.
PCA only (no t-SNE / UMAP), by decision. Baselines and other-signal analyses live in the repo (bft_signals.py, lgbm_warmstart.py) but not in this notebook."""
import nbformat as nbf
from pathlib import Path
NB = Path("/global/scratch/users/sergiomar10/boltzaff/notebooks/08_latent_space.ipynb")
def md(s): return nbf.v4.new_markdown_cell(s)
def code(s): return nbf.v4.new_code_cell(s)

PRE = r'''
import sys, warnings, numpy as np, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt
from IPython.display import display
from scipy import stats
warnings.filterwarnings("ignore")
sys.path.insert(0, "/global/scratch/users/sergiomar10/boltzaff/pipeline_local")
import bft_common as bc, bft_latent as lat
bc.style(); pd.set_option("display.width", 220); pd.set_option("display.max_columns", 40); pd.set_option("display.float_format", lambda x: f"{x:.3f}")
TL = bc.done_targets("scores"); SH = bc.SHORT; BLK, OUR, THEIR, NEU = bc.BLK, bc.OUR, bc.THEIR, bc.NEU; GRN = "#1baf7a"
def each(fn, label="", targets=None):
    """run fn(t) for every finished target; collect DataFrames; report targets that are pending or fail"""
    out, skipped = [], []
    for t in (targets or TL):
        try:
            r = fn(t)
            if r is None: skipped.append(f"{SH[t]} (pending)")
            else: out.append(r)
        except Exception as e: skipped.append(f"{SH[t]} ({type(e).__name__}: {str(e)[:60]})")
    if skipped: print(f"{label}: not available for " + ", ".join(skipped))
    return pd.concat(out, ignore_index=True) if out and isinstance(out[0], pd.DataFrame) else out
def arms_done(t): return [a for a in ("noft", "ft300s0", "ft300s1") if lat.load_latent(t, a) is not None]
def ready(t, need=("noft", "ft300s0")): return all(a in arms_done(t) for a in need)
'''

cells = [
 md("# 08 · The latent space of the affinity head (priority-ordered tracking notebook)\n\n"
    "Self-filling: each cell reports `pending` until its data exists and fills in when the notebook is re-run. Only latent-space questions live here; the supervised baselines and the other-signal analyses were moved out "
    "(code in `pipeline_local/lgbm_warmstart.py` and `bft_signals.py`, results in `results/analysis/<target>/`).\n\n"
    "**What is analysed.** The affinity head pools the ligand-protein pair representation into 128 numbers (`g_raw`), maps them through a two-layer MLP to 384 numbers (`g`, the post-MLP features the paper analyses), "
    "and two small branches turn `g` into the affinity value and the binary logit; two ensemble modules are averaged. Features were extracted from the same cached inputs for a sample of each target: the 300 training compounds, "
    "all evaluation actives, 2,500 random evaluation inactives and the top-1% picks of both models, under No-FT and two head-FT seeds (N=300). So a difference between arms is a real change of the representation, not of the input.\n\n"
    "**Estimates for the whole evaluation set** use sample weights: evaluation actives weight 1, random inactives weight n_inactive / n_drawn, inactives that are in only because they are top picks weight 0.\n\n"
    "**Why this order.** Earlier results say the ranking is largely ligand-driven (the decoy-protein test) and that plain ECFP baselines are strong (nearest-active Tanimoto beat head-FT on 2097), so the first question is "
    "whether the latent space holds anything beyond ligand chemistry. Then what fine-tuning changed, then where it fails, then what it over-weights, then whether anything is shared across targets. PCA and the paper's displacement-rank argument come last because they describe more than they test."),
 code(PRE),
 md("## Question tracker (computed from the files that exist)"),
 code('''def n_ok(check): return sum(1 for t in TL if check(t))
TR = [
 ("P1  Does the latent space hold information that fingerprints do not?", "Part 1", lambda t: ready(t, ("noft",))),
 ("P2  What did fine-tuning change: re-weighting or re-mapping? one shared direction or per-compound?", "Part 2", lambda t: ready(t)),
 ("P2  Do independent seeds shift the features the same way?", "Part 2", lambda t: ready(t, ("noft", "ft300s0", "ft300s1"))),
 ("P3  Where does it generalise and where does it fail (training vs evaluation actives, applicability domain)?", "Part 3", lambda t: ready(t)),
 ("P4  Shortcut check: latent coordinates against size, lipophilicity, charge", "Part 4", lambda t: ready(t)),
 ("P5  Is there a generic activity structure shared across targets (No-FT latent space)?", "Part 5", lambda t: ready(t, ("noft",))),
 ("Low  PCA of the features before and after", "Part 6", lambda t: ready(t)),
 ("Low  The paper's displacement-vs-rank argument with the ceiling-effect caveat", "Part 6", lambda t: ready(t)),
 ("Not started  Balanced-training vs top-N latent comparison (588689; needs extra extraction)", "-", lambda t: False),
 ("Not started  Latent space vs pose contacts (needs pose-derived interaction labels)", "-", lambda t: False),
]
T = pd.DataFrame([dict(question=q, section=s, status=("done" if n_ok(c) == len(TL) else f"{n_ok(c)} of {len(TL)} targets" if n_ok(c) else "pending")) for q, s, c in TR])
display(T.style.hide(axis="index")); print("arms extracted per target:", {SH[t]: arms_done(t) for t in TL})
print("2650 and 588549 are not scored, so they have no head-FT arm: 588549 (the weakest target) cannot be part of the failure analysis yet.")'''),

 md("## Part 1 (P1). Does the latent space hold information that fingerprints do not?\n"
    "The same logistic-regression probe, trained on the same 300 training compounds, on different inputs: ECFP4 bits; the No-FT 128-dim pooled features; the No-FT 384-dim post-MLP features; ECFP plus the post-MLP features; "
    "and the head-FT features. Evaluated on the evaluation sample (weighted AP and AUROC, estimates for the whole evaluation set). References: the head's own probability and a random ranking (AP = active rate). "
    "**Reading:** if the latent features do not beat ECFP, they are mostly a re-encoding of ligand chemistry. If ECFP plus latent beats either alone, the two carry different information. "
    "The probe is deliberately simple (one regularisation strength chosen by 5-fold CV inside the 300 labels), so it is a lower bound on what the features can do."),
 code('''I = each(lat.information_table, "information probes")
if len(I):
    A = I.pivot(index="input", columns="target", values="ap"); order = [i for i in ["reference: random", "ECFP4 bits", "No-FT pooled features (128-dim)", "No-FT post-MLP features (384-dim)", "ECFP4 + No-FT post-MLP", "head-FT pooled features (128-dim)", "head-FT post-MLP features (384-dim)", "reference: No-FT probability", "reference: head-FT probability"] if i in A.index]
    print("Weighted AP on the evaluation sample (trained on the 300 training compounds):"); display(A.loc[order]); print("Weighted AUROC:"); display(I.pivot(index="input", columns="target", values="auroc").loc[order])
    fig, ax = plt.subplots(figsize=(13, 4.6)); w = .8 / len(order); x = np.arange(A.shape[1])
    for i, r_ in enumerate(order): ax.bar(x + (i - len(order) / 2) * w, A.loc[r_].values, w, label=r_)
    ax.set_xticks(x); ax.set_xticklabels(A.columns); ax.set_ylim(0, None); ax.set_ylabel("weighted average precision"); ax.legend(fontsize=6, ncol=2); ax.set_title("Fig 1. Linear probes on ECFP and on the head's features, same 300 training compounds"); plt.tight_layout(); plt.show()
Nn = each(lambda t: lat.neighbour_table(t) if ready(t) else None, "nearest-active rankings")
if len(Nn):
    print("Nearest-active rankings (highest similarity to a training active), weighted AP; latent = cosine in the feature space, Morgan = Tanimoto:"); display(Nn.pivot(index="ranking", columns="target", values="ap"))'''),

 md("## Part 2 (P2). What did fine-tuning change?\n"
    "**2a. Re-weighting or re-mapping.** Compare, within a target, the probe on No-FT features with the probe on head-FT features (Part 1 table, rows 'No-FT ...' and 'head-FT ...'): if a probe on the No-FT features already does as well as the fine-tuned head, fine-tuning mostly re-weights what was already linearly readable; if only the head-FT features support a good probe, the representation was remapped.\n\n"
    "**2b. Direction.** For each compound d = f(head-FT) - f(No-FT). The paper reports only the size |d|. `coherence` = |mean d| / mean |d|: 1 if every compound moves the same way, near 0 if the shifts point in unrelated directions. "
    "`share_of_shift_in_first_direction`: the part of all shift energy in the single strongest direction. The 'activity direction' is the mean shift of the training actives; its cosine to a compound's shift is tested as a label separator.\n\n"
    "**2c. Seeds and layers.** Do two independently seeded fine-tunings move the features the same way (cosine between the two seeds' shifts), and is the change larger in the pooled 128-dim features or after the MLP (relative shift |d| / |f|)? Note that the fine-tuned module includes its pairformer stack, not only the MLP heads."),
 code('''Dd = []
for t in TL:
    r = lat.displacement_table(t) if ready(t) else None
    if r is not None: Dd.append(r)
if Dd:
    tab = pd.DataFrame([r[0] for r in Dd]).set_index("target"); print("2b. Direction of the shift (post-MLP features, seed 0):"); display(tab[["coherence", "cos_random_pairs_eval_inactive", "cos_random_pairs_eval_active", "cos_active_vs_inactive", "share_of_shift_in_first_direction", "auroc_cos_to_activity_direction", "auroc_shift_norm", "norm_train_active", "norm_train_inactive", "norm_eval_active", "norm_eval_inactive"]])
    fig, axes = plt.subplots(1, len(Dd), figsize=(3.4 * len(Dd), 3.4), squeeze=False)
    for ax, (row, d, D) in zip(axes[0], Dd):
        y = D.label.values; g = D.group.values; a_dir = d[(g == "train") & (y == 1)].mean(0); a_dir /= np.linalg.norm(a_dir); c = (d @ a_dir) / np.maximum(np.linalg.norm(d, axis=1), 1e-9); m = (~D.is_train.values) & (D.w.values > 0)
        ax.hist(c[m & (y == 0)], bins=40, range=(-1, 1), color="#bbbbbb", density=True, label="eval inactive"); ax.hist(c[m & (y == 1)], bins=40, range=(-1, 1), color=THEIR, alpha=.6, density=True, label="eval active")
        ax.set_title(row["target"], fontsize=9); ax.set_xlabel("cosine of the shift to the activity direction"); ax.set_ylabel("density")
    axes[0, 0].legend(fontsize=7); fig.suptitle("Fig 2. Is there a shared 'activity direction' in the feature shift?"); plt.tight_layout(rect=[0, 0, 1, .92]); plt.show()
else: print("pending: needs the No-FT and a head-FT arm of at least one target")
Sc = each(lambda t: lat.seed_consistency(t) if ready(t, ("noft", "ft300s0", "ft300s1")) else None, "seed consistency")
if len(Sc): print("2c. Two independent seeds and the two feature levels:"); display(Sc.set_index(["target", "features"]))'''),

 md("## Part 3 (P3). Where does it generalise and where does it fail?\n"
    "(i) **Training actives vs evaluation actives.** How close, in feature space, evaluation actives sit to the nearest training active compared with evaluation inactives, per target; and the number of training actives. Targets with few training actives (743445 has 10) or whose actives differ from the training actives are expected to generalise worst; 588549, the other weak target, is not scored yet. "
    "(ii) **Applicability domain.** The top-1% picks of head-FT split into thirds by their latent similarity, and by their Tanimoto, to the nearest training active: do picks nearer to the training actives turn out active more often, and is the latent measure better than Tanimoto?"),
 code('''rows = []
for t in TL:
    if not ready(t): continue
    D, X = lat.aligned(t, ("noft", "ft300s0")); y = D.label.values; ev = (~D.is_train.values) & (D.w.values > 0); ta = D.is_train.values & (y == 1)
    for arm in ("noft", "ft300s0"):
        F = X[arm] / np.linalg.norm(X[arm], axis=1, keepdims=True); s = (F @ F[ta].T).max(1)
        rows.append(dict(target=SH[t], arm=arm, training_actives=int(ta.sum()), median_sim_eval_active=float(np.median(s[ev & (y == 1)])), median_sim_eval_inactive=float(np.median(s[ev & (y == 0)])), auroc_of_similarity=lat.wauc(y[ev], s[ev], D.w.values[ev])))
if rows: print("(i) Cosine similarity of the post-MLP features to the nearest training active:"); display(pd.DataFrame(rows).set_index(["target", "arm"]))
else: print("pending")
Ap = each(lambda t: lat.applicability_table(t) if ready(t) else None, "applicability domain")
if len(Ap): print("(ii) Share of the top-1% head-FT picks that are active, by tercile of similarity to the nearest training active:"); display(Ap.pivot_table(index=["target", "measure"], columns="tercile", values="active_share")[["lowest third", "middle third", "highest third"]])'''),

 md("## Part 4 (P4). Shortcut check: what does the head over-weight?\nSpearman correlation between interpretable descriptors (MW, logP, heavy atoms, rings, TPSA, charge, ...) and the first principal components of the head-FT features and the size of the shift. A strong, consistent link to size or lipophilicity would say the head leans on a simple property; ligands in these libraries reach 92 to 100 heavy atoms (almost none above 60)."),
 code('''Ch = each(lambda t: lat.chemistry_table(t) if ready(t) else None, "descriptor correlations")
if len(Ch):
    ts = list(Ch.target.unique()); cols = ["shift_norm"] + [f"PC{j} (head-FT)" for j in range(1, 6)]
    fig, axes = plt.subplots(1, len(ts), figsize=(3.6 * len(ts), 4.4), squeeze=False)
    for ax, t_ in zip(axes[0], ts):
        A = Ch[Ch.target == t_].set_index("property")[cols]; im = ax.imshow(A.values, cmap="RdBu_r", vmin=-.8, vmax=.8, aspect="auto"); ax.grid(False)
        ax.set_xticks(range(len(cols))); ax.set_xticklabels(["shift", "PC1", "PC2", "PC3", "PC4", "PC5"], fontsize=7); ax.set_yticks(range(len(A))); ax.set_yticklabels(A.index, fontsize=7); ax.set_title(t_, fontsize=9)
        for i in range(A.shape[0]):
            for j in range(A.shape[1]): ax.text(j, i, f"{A.values[i, j]:.1f}", ha="center", va="center", fontsize=6)
    fig.suptitle("Fig 3. Spearman correlation of descriptors with the head-FT features' principal components and with the size of the shift"); plt.tight_layout(rect=[0, 0, 1, .92]); plt.show()
else: print("pending")'''),

 md("## Part 5 (P5). Is there a generic activity structure shared across targets?\nThe No-FT head is the same for every target, so its features live in one space. A probe is trained on target A's 300 training compounds and applied to target B's evaluation sample. "
    "Entries are weighted AP as a multiple of B's active rate (1 = no better than random); the diagonal is the within-target value. Clearly above 1 off the diagonal would mean the head's features carry a transferable notion of binding; values near 1 mean the structure is target-specific."),
 code('''ok_t = [t for t in TL if ready(t, ("noft",))]
if len(ok_t) >= 2:
    APm, AUCm = lat.transfer_table(ok_t); print("Weighted AP / active rate (rows: trained on, columns: evaluated on):"); display(APm); print("Weighted AUROC:"); display(AUCm)
    fig, ax = plt.subplots(figsize=(5.4, 4.6)); im = ax.imshow(APm.values.astype(float), cmap="Blues", vmin=0); ax.grid(False); ax.set_xticks(range(len(APm))); ax.set_xticklabels(APm.columns); ax.set_yticks(range(len(APm))); ax.set_yticklabels(APm.index)
    for i in range(len(APm)):
        for j in range(len(APm)): ax.text(j, i, f"{APm.values[i, j]:.1f}", ha="center", va="center", fontsize=8)
    ax.set_xlabel("evaluated on"); ax.set_ylabel("trained on"); ax.set_title("Fig 4. Cross-target transfer of a probe on No-FT features"); plt.colorbar(im, ax=ax, label="AP / active rate"); plt.tight_layout(); plt.show()
else: print("pending: needs the No-FT arm of at least two targets")'''),

 md("## Part 6 (lower priority). PCA, and the paper's displacement argument\n"
    "**6a. PCA** of the post-MLP features, fitted on the No-FT features of the evaluation part of the sample and applied to both arms, so a shift between the panels is real. Grey = random evaluation inactives, orange = evaluation actives, light blue = training inactives, dark blue = training actives. Descriptive only.\n\n"
    "**6b. The paper's argument.** The paper reports that the size of the feature shift correlates with the rank change (about -0.6). Part of that is mechanical: compounds that start near the top have the most room to fall. Raw and partial correlation (given the No-FT rank) are shown; 'overestimated compounds get corrected' stays a hypothesis."),
 code('''def pca_fig(targets):
    ok = [t for t in targets if ready(t)]
    if not ok: print("pending: no target has the No-FT and a head-FT arm extracted yet"); return
    fig, axes = plt.subplots(len(ok), 2, figsize=(11, 4.3 * len(ok)), squeeze=False, sharex="row", sharey="row")
    for r, t in enumerate(ok):
        D, X = lat.aligned(t, ("noft", "ft300s0")); ev = (~D.is_train.values) & (D.w.values > 0); _, pn, evr = lat.pca_coords(X["noft"][ev], X["noft"], 10); _, pf2, _ = lat.pca_coords(X["noft"][ev], X["ft300s0"], 10)
        g = D.group.values; y = D.label.values
        for c, (P, name) in enumerate(((pn, "No-FT"), (pf2, "head-FT N=300 (seed 0)"))):
            ax = axes[r, c]; m = (g == "eval inactive") & D.random_inactive.values
            ax.scatter(P[m, 0], P[m, 1], s=4, c="#cccccc", alpha=.5, label="eval inactive (random)"); ax.scatter(P[(g == "train") & (y == 0), 0], P[(g == "train") & (y == 0), 1], s=7, c="#9ecae1", label="training inactive")
            ax.scatter(P[g == "eval active", 0], P[g == "eval active", 1], s=6, c=THEIR, alpha=.7, label="eval active"); ax.scatter(P[(g == "train") & (y == 1), 0], P[(g == "train") & (y == 1), 1], s=10, c=OUR, label="training active")
            ax.set_title(f"{SH[t]}: {name}", fontsize=9); ax.set_xlabel(f"PC1 ({100 * evr[0]:.0f}% of variance in No-FT features)"); ax.set_ylabel(f"PC2 ({100 * evr[1]:.0f}%)")
            if r == 0 and c == 0: ax.legend(fontsize=6, markerscale=2)
    fig.suptitle("Fig 5. PCA of the 384-dim post-MLP features, same basis for both arms"); plt.tight_layout(rect=[0, 0, 1, .98]); plt.show()
pca_fig(TL)
if Dd:
    d2 = pd.DataFrame([{k: r[0][k] for k in ("target", "rho_norm_vs_rank_change", "partial_rho_norm_vs_rank_change_given_noft_rank")} for r in Dd]).set_index("target"); print("6b. Spearman correlation between the size of the shift and the rank change, raw and given the No-FT rank:"); display(d2)'''),

 md("## Conclusions (computed from the results above)\nEvery statement is generated from the tables; sections whose data is missing say so and make no claim."),
 code('''done_t = [SH[t] for t in TL if ready(t)]
print("targets with extracted features (No-FT and a head-FT seed):", done_t if done_t else "none yet: extraction jobs are queued")
if len(I):
    f = I.pivot(index="input", columns="target", values="ap")
    for tgt in f.columns:
        e = f.loc["ECFP4 bits", tgt]; g_ = f.loc["No-FT post-MLP features (384-dim)", tgt]; b = f.loc["ECFP4 + No-FT post-MLP", tgt]
        print(f"P1 {tgt}: probe AP on ECFP {e:.3f}; on No-FT post-MLP features {g_:.3f}; on both {b:.3f}; head's own probability {f.loc['reference: No-FT probability', tgt]:.3f}; random {f.loc['reference: random', tgt]:.3f}")
if Dd: print("P2 coherence of the shift: " + ", ".join(f"{r[0]['target']} {r[0]['coherence']:.2f}" for r in Dd) + " (1 = every compound moves the same way)")
if len(Sc): print("P2 cosine between two seeds' shifts (post-MLP): " + ", ".join(f"{r.target} {r.mean_cosine_between_seeds:.2f}" for r in Sc[Sc.features == 'post-MLP (384)'].itertuples()))
print("Not started: balanced-training vs top-N latent comparison; latent space vs pose contacts.")'''),
]
nb = nbf.v4.new_notebook(); nb.cells = cells
nb.metadata = {"kernelspec": {"name": "boltzba", "display_name": "Python (boltzba)", "language": "python"}, "language_info": {"name": "python"}}
NB.write_text(nbf.writes(nb)); print("wrote", NB)

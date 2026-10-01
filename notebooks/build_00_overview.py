#!/usr/bin/env python3
"""Builds 00_overview.ipynb: pipeline status + headline ours-vs-paper table + index of notebooks. Self-filling, matplotlib only, no HTML."""
import nbformat as nbf
from pathlib import Path
NB = Path("/global/scratch/users/sergiomar10/boltzaff/notebooks/00_overview.ipynb")
md = nbf.v4.new_markdown_cell; code = nbf.v4.new_code_cell
PRE = r'''
import sys; sys.path.insert(0, "/global/scratch/users/sergiomar10/boltzaff/pipeline_local")
from bft_replication import *
from bft_common import DATA, RUNS
import matplotlib.pyplot as plt
from IPython.display import display
style()
'''
cells = [
 md("# 00 Overview: Boltz-2 head fine-tuning reproduction (Furui and Ohue)\n\nSelf-filling: re-run after more targets finish and every table below updates, no edits needed. "
    "Targets with missing data are listed as pending."),
 code(PRE),
 md("## 1. Pipeline status per target\nStage is the furthest stage reached: library (Boltz inputs prepared) -> ft_inputs (fine-tuning inputs and eval set) -> scoring (k of 16 arms scored; 16 = No-FT plus 3 budgets x 5 seeds) -> scored -> table2 (eval_headft_seeds output exists)."),
 code(r'''st = status().copy()
def stage(r):
    k = int(str(r.arms_scored).split("/")[0])
    if r.table2: return "table2 done"
    if k == 16: return "scored (table2 not written yet)"
    if k > 0: return f"scoring ({k}/16 arms)"
    if r.ft_inputs: return "ft_inputs ready, scoring not started"
    if r.library == r.library and r.library > 0: return "library inputs only"
    return "not started"
st["stage"] = st.apply(stage, axis=1); display(st)
DONE = done_targets("scores")
print(f"fully scored targets: {len(DONE)}/8 -> {[SHORT[t] for t in DONE]}")
for t in TARGETS:
    if t not in DONE: print(f"pending: {SHORT[t]} - {st.loc[SHORT[t], 'stage']}")'''),
 md("## 2. Headline: ours vs the paper (scored targets)\nNo-FT AP, and head-FT AP at N=300 as mean and SD over the 5 seeds; ratio = head-FT mean / No-FT. Paper values are the single-seed numbers released in trainer_control.csv. Ratio SD is the seed SD of the head-FT AP divided by No-FT AP."),
 code(r'''rows = []; MT = {}
for t in TARGETS:
    pb = PAPER_BASE.loc[t, "auprc"]; pf = PAPER_FT.loc[(t, 300), "auprc"]
    r = dict(target=SHORT[t], paper_NoFT_AP=pb, paper_FT300_AP=pf, paper_ratio=pf / pb, active_rate_pct=100 * RATE[t])
    try:
        d = metrics_table(t)
    except Exception as e:
        d = None; print(f"{SHORT[t]}: error {e!r}")
    if d is None:
        r.update(status="pending"); print(f"pending: {SHORT[t]} - needs all 16 arms scored")
    else:
        MT[t] = d; b = d[d.arm == "base"].ap.iloc[0]; f = d[(d.arm == "headft") & (d.n_train == 300)].ap
        r.update(ours_NoFT_AP=b, ours_FT300_AP_mean=f.mean(), ours_FT300_AP_sd=f.std(ddof=1), ours_ratio=f.mean() / b, ours_ratio_sd=f.std(ddof=1) / b, status="scored")
    rows.append(r)
H = pd.DataFrame(rows).set_index("target")
display(H.style.format(precision=3, na_rep="-"))
print(f"n targets scored = {len(MT)}")'''),
 code(r'''fig, ax = plt.subplots(figsize=(10, 4.2)); x = np.arange(len(TARGETS)); w = 0.38
ax.bar(x - w/2, [H.loc[SHORT[t], "paper_ratio"] for t in TARGETS], w, color=THEIR, label="paper (single seed)")
ax.bar(x + w/2, [H.loc[SHORT[t]].get("ours_ratio", np.nan) for t in TARGETS], w, yerr=[H.loc[SHORT[t]].get("ours_ratio_sd", np.nan) if t in MT else 0 for t in TARGETS], capsize=3, color=OUR, label="ours (mean +/- SD over 5 seeds)")
for i, t in enumerate(TARGETS):
    if t not in MT: ax.text(i + w/2, 0.05, "pending", rotation=90, ha="center", va="bottom", fontsize=8)
ax.axhline(1, color="black", lw=1); ax.set_xticks(x); ax.set_xticklabels([SHORT[t] for t in TARGETS])
ax.set_xlabel("target (assay id)"); ax.set_ylabel("head-FT N=300 AP / No-FT AP"); ax.set_title("AP ratio at N=300 per target: paper vs ours"); ax.legend(fontsize=8)
plt.tight_layout(); plt.show()'''),
 md("## 3. The data before any training: SD Z-score of actives and inactives\n"
    "Each compound has the primary-screen **SD Z-score** (how many standard deviations the compound's readout lies from the plate/screen mean; it is the screen's raw hit strength, not a potency) and a binary label `target_active_v2`. "
    "The sign of a hit depends on the assay: in most targets actives have a high Z-score, but in one (see the table) actives are at strongly negative Z-scores, so the table works in the hit direction (the raw Z-score is what the density plots show). The label is the only ground truth used for training and evaluation; the Z-score is shown here only to see how clean the actives are and what the fine-tuning selection looks like. "
    "Every target's data is available now, so this section is complete for all 8 targets (n = number of compounds)."),
 code(r'''from scipy.stats import gaussian_kde
from sklearn.metrics import roc_auc_score
ZD = {}
for t in TARGETS:
    d = pd.read_csv(DATA / f"{t}.csv", usecols=["CID", "target_active_v2", "SD Z-score"]).rename(columns={"SD Z-score": "z"}); d["label"] = d.target_active_v2.astype(int); ZD[t] = d
rows = []; ORIENT = {}
for t in TARGETS:
    d = ZD[t]; ORIENT[t] = 1 if d[d.label == 1].z.median() > 0 else -1; d["zhit"] = d.z * ORIENT[t]       # hit direction: actives sit at high zhit
    a = d[d.label == 1].zhit; i = d[d.label == 0].zhit
    rows.append({"target": SHORT[t], "compounds": len(d), "actives": len(a), "inactives": len(i), "active rate %": round(100 * len(a) / len(d), 2),
                 "hits are at": "high Z" if ORIENT[t] > 0 else "LOW (negative) Z", "median raw Z actives": round(d[d.label == 1].z.median(), 2), "median raw Z inactives": round(d[d.label == 0].z.median(), 2),
                 "AUROC of hit-direction Z": round(roc_auc_score(d.label, d.zhit), 3), "% actives with hit-direction Z < 3": round(100 * (a < 3).mean(), 1), "% inactives with hit-direction Z > 3": round(100 * (i > 3).mean(), 2)})
display(pd.DataFrame(rows).set_index("target"))'''),
 md("### SD Z-score density, actives against inactives (n in each legend)\nSmoothed density (each class normalised to area 1, so the 100-to-1 class imbalance does not hide the actives) of the Z-score, with the number of compounds in each class."),
 code(r'''fig, axes = plt.subplots(2, 4, figsize=(19, 7.4)); xs = np.linspace(-12, 20, 400)
for ax, t in zip(axes.ravel(), TARGETS):
    d = ZD[t]
    for lab, col, nm in ((0, "#777777", "inactives"), (1, OUR, "actives")):
        z = d[d.label == lab].z.clip(-12, 20).values
        ax.fill_between(xs, gaussian_kde(z)(xs), color=col, alpha=0.35, lw=0); ax.plot(xs, gaussian_kde(z)(xs), color=col, lw=1.2, label=f"{nm} (n={len(z):,})")
    ax.set_xlim(-12, 20); ax.set_xlabel("SD Z-score (clipped to -12..20)"); ax.set_ylabel("density"); ax.set_title(f"{SHORT[t]}  ({len(d):,} compounds)", fontsize=10); ax.legend(fontsize=8)
plt.tight_layout(); plt.show()'''),
 md("### The same, as counts on a log axis\nHistogram of compound counts per Z-score bin (log y), so the size of the inactive population and the tail of high-Z inactives are visible."),
 code(r'''fig, axes = plt.subplots(2, 4, figsize=(19, 7.4)); bins = np.linspace(-12, 20, 65)
for ax, t in zip(axes.ravel(), TARGETS):
    d = ZD[t]
    for lab, col, nm in ((0, "#777777", "inactives"), (1, OUR, "actives")):
        z = d[d.label == lab].z.clip(-12, 20).values; ax.hist(z, bins=bins, color=col, alpha=0.6, label=f"{nm} (n={len(z):,})")
    ax.set_yscale("log"); ax.set_xlabel("SD Z-score"); ax.set_ylabel("compounds per bin (log)"); ax.set_title(SHORT[t], fontsize=10); ax.legend(fontsize=8)
plt.tight_layout(); plt.show()'''),
 md("### What goes into the fine-tuning: the training sets against the library\nThe training sets are the top-N compounds by the dataset's Boltz-2 score (N = 40, 100, 300), so they are not a random sample: they are the compounds the No-FT model already ranks highest. "
    "Per target: how many of them are actives, and where their Z-scores fall. Plots show the library densities (as above) with the **N = 300 training compounds as points** (blue = active, grey = inactive; vertical jitter only), with n for each."),
 code(r'''rows = []; TRN = {}
for t in TARGETS:
    for N in (40, 100, 300):
        p = RUNS / t / f"selection/train_top{N}.csv"
        if not p.exists(): continue
        tr = pd.read_csv(p, usecols=["CID", "Active_v2"]); m = tr.merge(ZD[t][["CID", "z", "label"]], on="CID", how="left"); TRN[(t, N)] = m
        a = m[m.Active_v2 == 1].z; i = m[m.Active_v2 == 0].z
        rows.append({"target": SHORT[t], "N": N, "training actives": len(a), "training inactives": len(i), "% active": round(100 * len(a) / len(m), 1), "library active rate %": round(100 * RATE[t], 2),
                     "median Z (train actives)": round(a.median(), 2) if len(a) else np.nan, "median Z (train inactives)": round(i.median(), 2) if len(i) else np.nan})
display(pd.DataFrame(rows).set_index(["target", "N"]))
rng = np.random.default_rng(0); fig, axes = plt.subplots(2, 4, figsize=(19, 7.8))
for ax, t in zip(axes.ravel(), TARGETS):
    d = ZD[t]
    for lab, col in ((0, "#777777"), (1, OUR)):
        ax.plot(xs, gaussian_kde(d[d.label == lab].z.clip(-12, 20).values)(xs), color=col, lw=1.0, alpha=0.8)
    if (t, 300) in TRN:
        m = TRN[(t, 300)]; top = ax.get_ylim()[1]
        for lab, col, nm in ((0, "#444444", "training inactives"), (1, "#0b3d91", "training actives")):
            z = m[m.Active_v2 == lab].z.clip(-12, 20).values; ax.scatter(z, -0.03 * top - 0.04 * top * rng.random(len(z)) - (0.06 * top if lab else 0), s=8, color=col, alpha=0.7, label=f"{nm} (n={len(z)})")
        ax.legend(fontsize=7, loc="upper right")
    ax.set_xlim(-12, 20); ax.set_xlabel("SD Z-score"); ax.set_ylabel("library density"); ax.set_title(f"{SHORT[t]}: N=300 training set", fontsize=10)
plt.tight_layout(); plt.show()'''),
 md("## 4. Notebook index\n"
    "- **01_replication**: Table 2 style cross-target geometric-mean ratios (AP, EF@1%, BEDROC), per-target AP on a log axis with active-rate lines, seed spread, and whether the paper's headline is reproduced.\n"
    "- **02_uncertainty**: whether the two-head disagreement of the affinity model flags unreliable predictions, before and after fine-tuning.\n"
    "- **03_ft_vs_noft_reranking**: which compounds fine-tuning moves up or down the ranking relative to No-FT, and whether those moves are correct.\n"
    "- **04_structural_cliffs**: activity-cliff pairs (similar structures, very different activity) and how well each model orders them.\n"
    "- **05_pose_density_and_decoy**: where predicted poses sit in the pocket, and control experiments with decoy proteins.\n"
    "- **06_training_strategies**: alternative training-set selections (for example balanced versus top-N actives) compared with the paper's recipe.\n\n"
    "Some of these notebooks may not exist yet while they are being built."),
 code(r'''nbs = ["01_replication", "02_uncertainty", "03_ft_vs_noft_reranking", "04_structural_cliffs", "05_pose_density_and_decoy", "06_training_strategies"]
for n in nbs: print(f"{n}.ipynb:", "present" if (ROOT / "notebooks" / f"{n}.ipynb").exists() else "not built yet")'''),
]
nb = nbf.v4.new_notebook(); nb.cells = cells
nb.metadata = {"kernelspec": {"name": "boltzba", "display_name": "Python (boltzba)"}, "language_info": {"name": "python"}}
NB.write_text(nbf.writes(nb)); print("wrote", NB)

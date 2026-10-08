#!/usr/bin/env python3
"""README figure for finding 7 (seed disagreement) -> notebooks/figures/finding_7_seeds.png. Reads results/analysis/<target>/variance_*.csv (made by variance_drivers.py).
Left: across-seed SD of the head-FT logit by decile of the score. Right: cross-validated R2 of what predicts the across-seed variance, by feature group. Black text, no bold, axes from 0."""
import sys
sys.path.insert(0, "/global/scratch/users/sergiomar10/boltzaff/pipeline_local")
import numpy as np, pandas as pd, matplotlib as mpl; mpl.use("Agg"); import matplotlib.pyplot as plt
import bft_common as bc
bc.style(); BLK = bc.BLK; COL = ["#2a78d6", "#eb6834", "#1baf7a", "#8e44ad", "#c9a227", "#7f7f7f"]
def load(n): return pd.concat([pd.read_csv(f, dtype={"target": str}) for f in (bc.AN / t / n for t in bc.TARGETS) if f.exists()], ignore_index=True)
SD, DR = load("variance_score_deciles.csv"), load("variance_drivers.csv"); TG = sorted(DR.target.unique())
fig, axs = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1, 1.25]})
for k, t in enumerate(TG): d = SD[SD.target == t]; axs[0].plot(d.bin, d.sd_logit, color=COL[k % 6], lw=1, alpha=.8, label=t)
axs[0].set_xticks(range(10)); axs[0].set_xlabel("decile of the head-FT score (0 = lowest)"); axs[0].set_ylabel("across-seed SD of the logit (median)"); axs[0].set_ylim(0, None); axs[0].set_title("Higher scores, less disagreement between seeds", fontsize=10); axs[0].legend(title="target", fontsize=7, title_fontsize=7)
A = DR[DR.analysis == "alone"].pivot(index="feature", columns="target", values="R2"); B = DR[DR.analysis == "score + chemistry: nothing removed"].pivot(index="feature", columns="target", values="R2"); R = DR[DR.analysis == "on the residual of the score"].pivot(index="feature", columns="target", values="R2")
rows = [("score", A.loc["score (all three)"]), ("chemistry (12 properties and flags)", A.loc["chemistry (all 12)"]), ("fingerprint bits (ECFP4)", A.loc["ECFP4 bits (2048)"]), ("other scores (docking, Boltz-2, ...)", A.loc["other scores (all 5)"]), ("similarity to training set", A.loc["Tanimoto to training (both)"]), ("score + chemistry", B.loc["-"]), ("what the score leaves: fingerprint bits", R.loc["ECFP4 bits (2048)"]), ("what the score leaves: chemistry", R.loc["chemistry (all 12)"])]
y = np.arange(len(rows))[::-1]
for yy, (n, v) in zip(y, rows):
    axs[1].barh(yy, np.nanmedian(v), color="#c9d8f0", edgecolor=BLK, linewidth=.5, height=.7)
    for k, t in enumerate(TG): axs[1].scatter(v[t], yy, s=14, color=COL[k % 6], zorder=3)
axs[1].set_yticks(y); axs[1].set_yticklabels([r[0] for r in rows]); axs[1].set_xlim(0, None); axs[1].set_xlabel("cross-validated R2 (bar: median, dots: targets)"); axs[1].set_title("What predicts the disagreement", fontsize=10)
fig.tight_layout(); fig.savefig(bc.ROOT / "notebooks/figures/finding_7_seeds.png", dpi=130); print("wrote finding_7_seeds.png")

#!/usr/bin/env python3
"""Build 05_replication.ipynb — our reproduction vs the paper's reported per-target numbers
(BoltzFT/figures/data, target 588689, common eval set n=49,685). Focus: replication, not
cross-method correlation. Matplotlib only (no HTML), clear labels, black text, no bold."""
import nbformat as nbf
from pathlib import Path
NB = Path("/global/scratch/users/sergiomar10/boltzaff/notebooks/05_replication.ipynb")
def md(s): return nbf.v4.new_markdown_cell(s)
def code(s): return nbf.v4.new_code_cell(s)

PRE = r'''
from pathlib import Path
import numpy as np, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt
from IPython.display import display
BLK="#000000"; OUR="#2a78d6"; THEIR="#eb6834"
mpl.rcParams.update({"figure.facecolor":"white","axes.facecolor":"white","savefig.facecolor":"white",
  "text.color":BLK,"axes.labelcolor":BLK,"axes.titlecolor":BLK,"xtick.color":BLK,"ytick.color":BLK,
  "axes.edgecolor":BLK,"font.size":10,"font.weight":"normal","axes.titleweight":"normal","axes.labelweight":"normal",
  "axes.grid":True,"grid.color":"#e6e6e6","axes.axisbelow":True,"axes.spines.top":False,"axes.spines.right":False})
ROOT=Path("/global/scratch/users/sergiomar10/boltzaff"); D=ROOT/"BoltzFT/figures/data"; T="588689"
# THEIRS (reported by the paper's released CSVs)
sb=pd.read_csv(D/"baseline_cv/supervised_baselines.csv"); sb["target"]=sb.target.astype(str)
their_lgbm=sb[(sb.target==T)&(sb.arm=="ecfp_lgbm_cv")].set_index("train_size")[["ef_1pct","auprc","bedroc"]].sort_index()
tc=pd.read_csv(D/"trainer_control.csv"); tc["target"]=tc.target.astype(str)
their_boltz=tc[tc.target==T].set_index(["arm","n_train"])[["ef_1pct","auprc","bedroc"]]   # base + lightning
# OURS (our reproduction, same eval>300)
o=pd.read_csv(ROOT/"results/analysis/cpu_baselines_588689.csv")
our_lgbm=o[o.method=="LightGBM_ECFP_avg5"].set_index("n_train")[["ef_1pct","auprc","bedroc"]].sort_index()
our_boltz_proxy=o[o.method=="Boltz2_noFT(score)"][["ef_1pct","auprc","bedroc"]].iloc[0]
print("loaded their + our numbers for", T)
'''

cells=[
 md("# 05 · Replication: our reproduction vs the paper\n\n"
    "Does our pipeline reproduce the numbers the paper (BoltzFT) reports for target 588689? Both are on "
    "the same common evaluation set (n = 49,685). This is a replication check, not a cross-method "
    "correlation.\n\n"
    "What we can compare now: our LightGBM-ECFP baseline vs their reported LightGBM-ECFP, and our Boltz-2 "
    "No-FT proxy vs their No-FT. Our own Boltz-2 No-FT and head-FT (from pipeline scoring) are reserved "
    "until that scoring finishes."),
 code(PRE),
 md("## LightGBM-ECFP: our reproduction vs the paper's reported value\n"
    "Same descriptor baseline (2048-bit ECFP), same eval set, budgets N = 40 / 100 / 300."),
 code('tab=pd.DataFrame({\n'
      '  ("EF@1%","paper"):their_lgbm.ef_1pct, ("EF@1%","ours"):our_lgbm.ef_1pct,\n'
      '  ("AP","paper"):their_lgbm.auprc,       ("AP","ours"):our_lgbm.auprc,\n'
      '  ("BEDROC","paper"):their_lgbm.bedroc,  ("BEDROC","ours"):our_lgbm.bedroc,\n'
      '}).round(4); tab.index.name="N (labels)"; display(tab)'),
 md("## Replication scatter — points on the diagonal = we reproduced their number\n"
    "x-axis: value reported by the paper; y-axis: our reproduction. One point per label budget. The "
    "dashed line is perfect replication (y = x)."),
 code('fig,axes=plt.subplots(1,3,figsize=(15,4.6))\n'
      'for ax,metric,col in zip(axes,["EF@1%","AP","BEDROC"],["ef_1pct","auprc","bedroc"]):\n'
      '    tx=their_lgbm[col].values; oy=our_lgbm[col].reindex(their_lgbm.index).values\n'
      '    lo=min(tx.min(),oy.min()); hi=max(tx.max(),oy.max()); pad=(hi-lo)*0.15+1e-6\n'
      '    ax.plot([lo-pad,hi+pad],[lo-pad,hi+pad],ls="--",color=BLK,lw=1,label="perfect replication")\n'
      '    ax.scatter(tx,oy,s=90,color=OUR,zorder=3)\n'
      '    for n,x,y in zip(their_lgbm.index,tx,oy): ax.annotate(f"N={n}",(x,y),textcoords="offset points",xytext=(6,4),fontsize=8)\n'
      '    ax.set_xlabel(f"{metric}  (paper / BoltzFT reported)"); ax.set_ylabel(f"{metric}  (our reproduction)")\n'
      '    ax.set_title(f"LightGBM-ECFP: {metric}"); ax.legend(fontsize=8)\n'
      'fig.suptitle("Replication of the LightGBM-ECFP baseline (588689, eval n=49,685)",fontsize=12)\n'
      'plt.tight_layout(); plt.show()'),
 md("## Boltz-2 No-FT: our proxy vs their reported No-FT\n"
    "Their No-FT (`base`) vs our current proxy, the shipped `score_boltz2`. These are close because both "
    "trace to the same standard Boltz-2 inference. Our own pipeline No-FT (from the poses we folded) is "
    "reserved below."),
 code('base=their_boltz.loc[("base",0)]\n'
      'cmp=pd.DataFrame({"paper No-FT (base)":[base.ef_1pct,base.auprc,base.bedroc],\n'
      '                  "our proxy (score_boltz2)":[our_boltz_proxy.ef_1pct,our_boltz_proxy.auprc,our_boltz_proxy.bedroc]},\n'
      '                 index=["EF@1%","AP","BEDROC"]).round(4); display(cmp)'),
 md("## Why our own Boltz-2 No-FT will not be identical to `score_boltz2`\n"
    "The shipped `score_boltz2` comes from standard Boltz-2 inference, which reselects a pose during the "
    "affinity stage. Our No-FT scores the affinity head on the stored pre-affinity pose (the paper's own "
    "protocol, which it distinguishes from the initial ranking). Same model, different pose -> the scores "
    "differ. That difference is the paper's `base` arm, which we will reproduce directly once scoring "
    "finishes."),
 md("## Reserved: head-FT (the headline)\n"
    "Their reported head-FT (`lightning`) per budget is shown; the `ours` column fills when our 15 FT arms "
    "finish scoring. That is the actual replication of the paper's main result."),
 code('rows=[]\n'
      'for n in (40,100,300):\n'
      '    if ("lightning",n) in their_boltz.index:\n'
      '        r=their_boltz.loc[("lightning",n)]\n'
      '        rows.append(dict(N=n, EF_1pct_paper=round(r.ef_1pct,3), AP_paper=round(r.auprc,3),\n'
      '                         BEDROC_paper=round(r.bedroc,3), EF_1pct_ours="pending", AP_ours="pending"))\n'
      'display(pd.DataFrame(rows))\n'
      'print("head-FT (ours) fills in from results/analysis/588689/ once the FT arms score.")'),
]
nb=nbf.v4.new_notebook(); nb.cells=cells
nb.metadata={"kernelspec":{"name":"boltzba","display_name":"Python (boltzba)"},"language_info":{"name":"python"}}
NB.write_text(nbf.writes(nb)); print("wrote", NB)

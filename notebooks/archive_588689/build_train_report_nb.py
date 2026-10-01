#!/usr/bin/env python3
"""06_training_report.ipynb — head-FT training report: checkpoint inventory + per-epoch validation
learning curves (from the fork's val_predictions_epoch_*.csv). Matplotlib only, black text, no bold."""
import nbformat as nbf
from pathlib import Path
NB = Path("/global/scratch/users/sergiomar10/boltzaff/notebooks/06_training_report.ipynb")
def md(s): return nbf.v4.new_markdown_cell(s)
def code(s): return nbf.v4.new_code_cell(s)

PRE = r'''
import glob, numpy as np, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt
from IPython.display import display
BLK="#000000"; PAL=["#2a78d6","#eb6834","#1baf7a"]
mpl.rcParams.update({"figure.facecolor":"white","axes.facecolor":"white","savefig.facecolor":"white",
  "text.color":BLK,"axes.labelcolor":BLK,"axes.titlecolor":BLK,"xtick.color":BLK,"ytick.color":BLK,
  "axes.edgecolor":BLK,"font.size":10,"font.weight":"normal","axes.titleweight":"normal","axes.labelweight":"normal",
  "axes.grid":True,"grid.color":"#e6e6e6","axes.axisbelow":True,"axes.spines.top":False,"axes.spines.right":False})
R="/global/scratch/users/sergiomar10/boltzaff/results/runs/588689"
rows=[]
for arm in sorted(glob.glob(R+"/headft_full/top*_seed*")):
    name=arm.split("/")[-1]
    if "work" in name: continue
    N=int(name.split("_")[0].replace("top","")); seed=int(name.split("seed")[1])
    for e in range(5):
        try: d=pd.read_csv(f"{arm}/validation_predictions/val_predictions_epoch_{e}.csv")
        except: continue
        y=d.binary_target.values; p=d.affinity_probability_binary.values
        bce=-np.mean(y*np.log(np.clip(p,1e-9,1))+(1-y)*np.log(np.clip(1-p,1e-9,1)))
        rows.append(dict(N=N,seed=seed,epoch=e,val_prob=float(p.mean()),val_bce=float(bce),n_pos=int(y.sum()),n=len(y)))
vf=pd.DataFrame(rows)
nck_main=len(glob.glob(R+"/headft_full/top*_seed*/checkpoints/last.ckpt"))
nck_var=len([p for p in glob.glob(R+"/headft_variants/*_seed*/checkpoints/last.ckpt") if "work" not in p])
print(f"checkpoints trained: main {nck_main}/15, variants {nck_var}/70")
print(f"validation set: {vf.n.iloc[0]} compounds, {vf.n_pos.iloc[0]} positive (paper protocol = last 20 of ranking)")
'''

cells=[
 md("# 06 · Head-FT training report\n\n"
    "What the fine-tuning did during training: checkpoint inventory and per-epoch validation learning "
    "curves, read from the fork's `val_predictions_epoch_*.csv`. This is training-time behaviour; the "
    "screening performance (EF@1% / AP on the 49,685-compound eval set) is computed separately once "
    "scoring finishes.\n\n"
    "Note: the paper's validation set is the last 20 compounds of the ranking, which here are all "
    "inactive. So these curves show the head learning to down-weight high-ranked-but-inactive compounds "
    "(the demote-false-positives behaviour) and that training converges - not active-vs-inactive "
    "separation, which is the screening result."),
 code(PRE),
 md("## Validation loss per epoch (mean over the 15 main arms, with seed spread)\n"
    "Binary cross-entropy on the held-out validation compounds. Monotonic decrease = the fine-tune is "
    "learning and converging."),
 code('g=vf.groupby("epoch").val_bce.agg(["mean","std"])\n'
      'fig,ax=plt.subplots(figsize=(7,4.5))\n'
      'ax.errorbar(g.index,g["mean"],yerr=g["std"],marker="o",color="#2a78d6",capsize=4,lw=1.8)\n'
      'ax.set_xlabel("epoch"); ax.set_ylabel("validation BCE loss"); ax.set_xticks(range(5))\n'
      'ax.set_title("head-FT validation loss (15 main arms, mean +/- SD across seeds)")\n'
      'plt.tight_layout(); plt.show()\n'
      'display(g.round(4))'),
 md("## Learning by label budget\n"
    "Mean predicted probability on the (all-inactive) validation set over epochs, per budget N. Lower = "
    "the head more confidently suppresses these false positives; more labels drive stronger suppression."),
 code('fig,ax=plt.subplots(figsize=(7,4.5))\n'
      'for c,N in zip(PAL,[40,100,300]):\n'
      '    s=vf[vf.N==N].groupby("epoch").val_prob.agg(["mean","std"])\n'
      '    ax.errorbar(s.index,s["mean"],yerr=s["std"],marker="o",color=c,capsize=3,lw=1.6,label=f"N={N}")\n'
      'ax.set_xlabel("epoch"); ax.set_ylabel("mean predicted prob on val (all inactive)"); ax.set_xticks(range(5))\n'
      'ax.set_title("suppression of held-out false positives, by label budget"); ax.legend()\n'
      'plt.tight_layout(); plt.show()'),
 md("## Checkpoint inventory\n"
    "Final-epoch checkpoints saved per arm (main reproduction) and the variant sweep."),
 code('inv=vf[vf.epoch==4].groupby("N").agg(arms=("seed","nunique"),\n'
      '     final_val_bce=("val_bce","mean"), final_val_prob=("val_prob","mean")).round(4)\n'
      'inv["checkpoints"]=inv.arms.astype(str)+"/5 seeds"\n'
      'display(inv)\n'
      'print(f"main checkpoints: {nck_main}/15   variant checkpoints: {nck_var}/70   (5 epochs each)")\n'
      'print("screening performance (EF@1%/AP on 49,685 eval compounds) -> results/analysis/588689/ once scoring completes")'),
]
nb=nbf.v4.new_notebook(); nb.cells=cells
nb.metadata={"kernelspec":{"name":"boltzba","display_name":"Python (boltzba)"},"language_info":{"name":"python"}}
NB.write_text(nbf.writes(nb)); print("wrote", NB)

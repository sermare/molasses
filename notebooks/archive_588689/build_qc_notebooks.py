#!/usr/bin/env python3
"""Build 01_boltzQC / 01_boltz_stats / 02_regression / 02_benchmarking from qc.csv.
All plot text black, normal weight; markdown uses no bold emphasis."""
import nbformat as nbf
from pathlib import Path
NBDIR = Path("/global/scratch/users/sergiomar10/boltzaff/notebooks")

def md(s): return nbf.v4.new_markdown_cell(s)
def code(s): return nbf.v4.new_code_cell(s)
def write(name, cells):
    nb = nbf.v4.new_notebook(); nb.cells = cells
    nb.metadata = {"kernelspec": {"name": "boltzba", "display_name": "Python (boltzba)"},
                   "language_info": {"name": "python"}}
    (NBDIR / name).write_text(nbf.writes(nb)); print("wrote", name)

PRE = '''\
import os, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib as mpl, matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
pd.set_option("display.width", 160, "display.max_columns", 40)
ROOT = Path("/global/scratch/users/sergiomar10/boltzaff")
QC = pd.read_csv(ROOT / "results/runs/588689/qc.csv")
# all text black, normal weight (no bold)
BLK="#000000"; GRID="#dedede"
PALETTE=["#2a78d6","#eb6834","#1baf7a","#eda100","#e87ba4","#4a3aa7","#e34948","#008300"]
ACTIVE="#1baf7a"; INACTIVE="#9aa0a6"; BLUE="#2a78d6"
mpl.rcParams.update({
  "figure.facecolor":"white","axes.facecolor":"white","savefig.facecolor":"white",
  "text.color":BLK,"axes.labelcolor":BLK,"axes.titlecolor":BLK,"xtick.color":BLK,"ytick.color":BLK,
  "axes.edgecolor":BLK,"font.size":10,"font.weight":"normal","axes.titleweight":"normal","axes.labelweight":"normal",
  "axes.grid":True,"grid.color":GRID,"grid.linewidth":0.6,"axes.axisbelow":True,
  "axes.spines.top":False,"axes.spines.right":False,"figure.dpi":100})
METRICS=["confidence","complex_plddt","ptm","iptm","ligand_iptm","complex_pde"]
print(f"folded compounds: {len(QC)}   actives: {int(QC.Active.sum())} ({QC.Active.mean()*100:.2f}%)")
'''

CAVEAT = ('> Preliminary. These use the ~20k compounds co-folded so far (a partial, roughly '
          'CID-ordered — hence roughly random — sample of the 49,985-compound library). Absolute '
          'numbers will shift as folding completes, and none of this includes fine-tuning yet, so '
          'nothing here speaks to the paper head-FT claim. What it does show: whether the co-folding '
          'is healthy, and whether the structural-confidence signals carry any activity information.')

# ---------------- 01_boltzQC ----------------
qc = [
 md("# 01 · Boltz-2 co-folding QC\n\n"
    "Quality control of our Pass-1 co-folding run for target 588689 (Dengue NS5). Each compound was "
    "co-folded once; here we look at the model-confidence outputs (pLDDT, pTM, ipTM, ligand ipTM, "
    "PDE) to check the predicted complexes are sane before they feed the affinity stage.\n\n" + CAVEAT),
 code(PRE),
 md("## Coverage and representativeness\n"
    "How much of the library is folded, and is the folded subset representative of the full active rate?"),
 code('full_rate = 0.972  # full-library active rate for 588689 (486/49985 * 100)\n'
      'print(f"folded: {len(QC)} / 49985 ({len(QC)/49985*100:.1f}%)")\n'
      'print(f"folded active rate: {QC.Active.mean()*100:.2f}%   full-library: {full_rate:.2f}%")\n'
      'print("-> folded subset tracks the full active rate, so preliminary stats are representative"\n'
      '      if abs(QC.Active.mean()*100-full_rate)<0.3 else "-> some sampling skew, read with care")'),
 md("## Confidence metric distributions\n"
    "Distribution of each Boltz-2 confidence output across the folded complexes."),
 code('fig,axes=plt.subplots(2,3,figsize=(14,7))\n'
      'for ax,m in zip(axes.ravel(),METRICS):\n'
      '    x=QC[m].dropna()\n'
      '    ax.hist(x,bins=50,color=BLUE)\n'
      '    ax.axvline(x.median(),color=BLK,ls="--",lw=1)\n'
      '    ax.set_title(f"{m}  (median {x.median():.3f})")\n'
      'fig.suptitle("Co-folding confidence distributions (588689)")\n'
      'plt.tight_layout(); plt.show()'),
 md("## Fraction of complexes above common confidence thresholds"),
 code('checks=[("complex_plddt",0.7),("complex_plddt",0.8),("ligand_iptm",0.5),("ligand_iptm",0.7),("confidence",0.8)]\n'
      'rows=[dict(metric=m,threshold=t,frac_above=round((QC[m]>=t).mean(),3)) for m,t in checks]\n'
      'pd.DataFrame(rows)'),
 md("## Do pLDDT and ligand-placement confidence agree?\n"
    "Protein-fold confidence (complex pLDDT) vs ligand-pose confidence (ligand ipTM). "
    "A high-pLDDT / low-ipTM cluster = confident fold but uncertain ligand placement."),
 code('fig,ax=plt.subplots(figsize=(6.5,5))\n'
      'hb=ax.hexbin(QC.complex_plddt,QC.ligand_iptm,gridsize=45,cmap="Blues",mincnt=1)\n'
      'ax.set_xlabel("complex pLDDT"); ax.set_ylabel("ligand ipTM"); ax.set_title("fold vs pose confidence")\n'
      'plt.colorbar(hb,label="compounds"); plt.tight_layout(); plt.show()'),
]
write("01_boltzQC.ipynb", qc)

# ---------------- 01_boltz_stats ----------------
stats = [
 md("# 01 · Confidence vs activity\n\n"
    "Do the co-folding confidence metrics carry any signal about experimental activity, and how do "
    "they relate to the cached Boltz-2 screening score? These are structural-confidence signals, not "
    "the affinity score itself (that comes from Pass-2).\n\n" + CAVEAT),
 code(PRE),
 md("## Metric distributions: actives vs inactives"),
 code('show=["ligand_iptm","complex_plddt","iptm","confidence"]\n'
      'fig,axes=plt.subplots(1,4,figsize=(16,3.8))\n'
      'for ax,m in zip(axes,show):\n'
      '    ax.hist(QC[QC.Active==0][m].dropna(),bins=40,density=True,alpha=.6,color=INACTIVE,label="inactive")\n'
      '    ax.hist(QC[QC.Active==1][m].dropna(),bins=40,density=True,alpha=.6,color=ACTIVE,label="active")\n'
      '    ax.set_title(m)\n'
      'axes[0].legend()\n'
      'fig.suptitle("Confidence metrics by activity (folded subset)")\n'
      'plt.tight_layout(); plt.show()'),
 md("## Correlation among score and confidence metrics (Spearman)"),
 code('cols=["score_boltz2"]+METRICS\n'
      'corr=QC[cols].corr(method="spearman")\n'
      'fig,ax=plt.subplots(figsize=(7,6))\n'
      'im=ax.imshow(corr.values,cmap="RdBu_r",vmin=-1,vmax=1)\n'
      'ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols,rotation=45,ha="right")\n'
      'ax.set_yticks(range(len(cols))); ax.set_yticklabels(cols)\n'
      'for i in range(len(cols)):\n'
      '    for j in range(len(cols)):\n'
      '        ax.text(j,i,f"{corr.values[i,j]:.2f}",ha="center",va="center",fontsize=8,color=BLK)\n'
      'plt.colorbar(im,fraction=0.046,pad=0.04); ax.set_title("Spearman correlation")\n'
      'plt.tight_layout(); plt.show()'),
 md("## Median metric by activity"),
 code('QC.groupby("Active")[["score_boltz2"]+METRICS].median().T.rename(columns={0:"inactive",1:"active"})'),
]
write("01_boltz_stats.ipynb", stats)

# ---------------- 02_regression ----------------
reg = [
 md("# 02 · Preliminary predictive analysis\n\n"
    "How much activity signal is in each single metric, and in a simple combined model, on the folded "
    "subset? This is a look at signal content, not the paper method (no Boltz-2 fine-tuning here).\n\n" + CAVEAT),
 code(PRE),
 code('from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve\n'
      'from sklearn.linear_model import LogisticRegression\n'
      'from sklearn.model_selection import cross_val_predict\n'
      'from sklearn.preprocessing import StandardScaler\n'
      'from sklearn.pipeline import make_pipeline\n'
      'y=QC.Active.values\n'
      'feats=["score_boltz2"]+METRICS\n'
      'D=QC[feats+["Active"]].dropna(); y=D.Active.values'),
 md("## Single-metric ranking power\n"
    "AUROC and average precision when ranking by each metric alone (pde is inverted: lower is better)."),
 code('rows=[]\n'
      'for m in feats:\n'
      '    s=D[m].values.astype(float)\n'
      '    if m=="complex_pde": s=-s\n'
      '    rows.append(dict(metric=m, auroc=roc_auc_score(y,s), ap=average_precision_score(y,s)))\n'
      'res=pd.DataFrame(rows).sort_values("auroc",ascending=False)\n'
      'fig,ax=plt.subplots(figsize=(9,4))\n'
      'ax.bar(range(len(res)),res.auroc,color=BLUE)\n'
      'ax.axhline(0.5,color=BLK,ls="--",lw=1)\n'
      'ax.set_xticks(range(len(res))); ax.set_xticklabels(res.metric,rotation=45,ha="right")\n'
      'ax.set_ylim(0,1); ax.set_ylabel("AUROC"); ax.set_title("single-metric activity AUROC (folded subset)")\n'
      'for i,v in enumerate(res.auroc): ax.text(i,v,f"{v:.2f}",ha="center",va="bottom",fontsize=8)\n'
      'plt.tight_layout(); plt.show()\n'
      'res.round(3)'),
 md("## Combined logistic-regression model (5-fold CV)\n"
    "Activity regressed on the confidence metrics (out-of-fold predictions), vs the cached score alone."),
 code('X=D[METRICS].values\n'
      'clf=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000,class_weight="balanced"))\n'
      'p_oof=cross_val_predict(clf,X,y,cv=5,method="predict_proba")[:,1]\n'
      'print(f"confidence-features model : AUROC {roc_auc_score(y,p_oof):.3f}  AP {average_precision_score(y,p_oof):.3f}")\n'
      's=D.score_boltz2.values\n'
      'print(f"cached score_boltz2 alone  : AUROC {roc_auc_score(y,s):.3f}  AP {average_precision_score(y,s):.3f}")\n'
      'fig,ax=plt.subplots(figsize=(5.5,5))\n'
      'for lab,sc,c in [("confidence model",p_oof,BLUE),("score_boltz2",s,PALETTE[1])]:\n'
      '    fpr,tpr,_=roc_curve(y,sc); ax.plot(fpr,tpr,color=c,label=f"{lab} (AUROC {roc_auc_score(y,sc):.2f})")\n'
      'ax.plot([0,1],[0,1],ls="--",color=BLK,lw=1)\n'
      'ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate"); ax.set_title("ROC (folded subset)")\n'
      'ax.legend(); plt.tight_layout(); plt.show()'),
]
write("02_regression.ipynb", reg)

# ---------------- 02_benchmarking ----------------
bench = [
 md("# 02 · Early-enrichment benchmark\n\n"
    "The paper's question is: given 40-300 activity labels from an initial screen, does using them "
    "help? This notebook benchmarks the label-using CPU baselines (LightGBM-ECFP, nearest-active "
    "similarity) against a structure-only reference, with the fine-tuned Boltz-2 (head-FT) column "
    "reserved. Nearest-active similarity is a comparator, not the paper's thesis.\n\n"
    "Provisional and caveated: (1) one target only, and 588689 is the easiest of the eight (highest "
    "No-FT AP/EF, and per the SI the one where head-FT reduced unseen-scaffold recovery) - do not "
    "generalize from it; (2) the structure-only reference here is `score_boltz2`, which is the "
    "INITIAL Boltz-2 ranking, not the pipeline's No-FT rescore - head-FT will be compared to the "
    "pipeline No-FT, not to this proxy; (3) point estimates carry bootstrap CIs below because single "
    "runs on one target are not distinguishable from noise."),
 code(PRE + '\n'
      'from IPython.display import display\n'
      'sys.path.insert(0, str(ROOT/"BoltzFT/evaluation"))\n'
      'from metrics import all_metrics\n'
      'CB = pd.read_csv(ROOT/"results/analysis/cpu_baselines_588689.csv")\n'
      'print("methods:", sorted(CB.method.unique()))'),
 md("## The table: No-FT / LightGBM / nearest-active, head-FT reserved\n"
    "All on the full common eval set (49,685 compounds, 396 actives). EF@1% is enrichment over random; "
    "AP and BEDROC weight the top of the ranking."),
 code('def rows_for(m,label):\n'
      '    return [dict(method=label,N=int(r.n_train),AUROC=r.auroc,AP=r.auprc,EF1=r.ef_1pct,BEDROC=r.bedroc)\n'
      '            for _,r in CB[CB.method==m].iterrows()]\n'
      'T=rows_for("Boltz2_noFT(score)","Boltz-2 initial score (proxy)")\n'
      'T+=rows_for("LightGBM_ECFP_avg5","LightGBM-ECFP")\n'
      'T+=rows_for("NearestActive_Tanimoto","Nearest-active Tanimoto")\n'
      'sp=ROOT/"results/analysis/588689/spread.csv"\n'
      'hft_ready=sp.exists()\n'
      'if hft_ready:\n'
      '    s=pd.read_csv(sp)\n'
      '    for _,r in s[s.n_train>0].iterrows():\n'
      '        T.append(dict(method="Boltz-2 head-FT",N=int(r.n_train),AUROC=np.nan,\n'
      '                      AP=r["ap_mean"],EF1=r["ef1_mean"],BEDROC=r["bedroc_mean"]))\n'
      'else:\n'
      '    for N in (40,100,300): T.append(dict(method="Boltz-2 head-FT",N=N,AUROC=np.nan,AP=np.nan,EF1=np.nan,BEDROC=np.nan))\n'
      'tab=pd.DataFrame(T)\n'
      'print("head-FT results present:", hft_ready)\n'
      'order=["Boltz-2 initial score (proxy)","Nearest-active Tanimoto","LightGBM-ECFP","Boltz-2 head-FT"]\n'
      'for metric in ["EF1","AP","BEDROC"]:\n'
      '    piv=tab.pivot_table(index="method",columns="N",values=metric).reindex(order)\n'
      '    print(f"\\n{metric} by method x N (N=0 column is No-FT, structure-free):")\n'
      '    display(piv.round(3))'),
 md("## Bootstrap 95% CIs (full eval)\n"
    "The paper reports single runs; here every full-eval point estimate gets a 1000-sample bootstrap CI "
    "over the eval compounds. Overlapping CIs = the difference is not distinguishable from noise."),
 code('ci=pd.read_csv(ROOT/"results/analysis/cpu_ci_588689.csv")\n'
      'lab={"Boltz2_initial_score":"Boltz-2 initial (proxy)","NearestActive_Tanimoto":"Nearest-active","LightGBM_ECFP_avg5":"LightGBM-ECFP"}\n'
      'ci["method"]=ci.method.map(lab).fillna(ci.method)\n'
      'ci["EF1_95CI"]=ci.apply(lambda r:f"[{r.ef1_lo:.1f}, {r.ef1_hi:.1f}]",axis=1)\n'
      'ci["AP_95CI"]=ci.apply(lambda r:f"[{r.ap_lo:.3f}, {r.ap_hi:.3f}]",axis=1)\n'
      'ci[["method","n_train","EF1_95CI","AP_95CI"]]'),
 md("## EF@1% and average precision vs label budget\n"
    "Dashed line = off-the-shelf Boltz-2 (No-FT), which uses no labels. The reserved slot is where the "
    "fine-tuned Boltz-2 arm will appear; its job is to clear the nearest-active similarity bar."),
 code('import matplotlib.patches as mpatches\n'
      'Ns=[40,100,300]; methods=["Nearest-active Tanimoto","LightGBM-ECFP","Boltz-2 head-FT"]\n'
      'cols={"Nearest-active Tanimoto":PALETTE[2],"LightGBM-ECFP":PALETTE[3],"Boltz-2 head-FT":BLUE}\n'
      'def vals(mth,col):\n'
      '    return [ (tab[(tab.method==mth)&(tab.N==n)][col].values[:1] or [np.nan])[0] for n in Ns]\n'
      'fig,(a1,a2)=plt.subplots(1,2,figsize=(14,4.8)); x=np.arange(len(Ns)); w=0.26\n'
      'for i,mth in enumerate(methods):\n'
      '    ef=vals(mth,"EF1"); ap=vals(mth,"AP")\n'
      '    if mth=="Boltz-2 head-FT" and not hft_ready:\n'
      '        for a in (a1,a2):\n'
      '            for xi in x: a.text(xi+i*w, 0, " head-FT\\n pending", ha="center", va="bottom", fontsize=7, rotation=90)\n'
      '        continue\n'
      '    a1.bar(x+i*w,ef,w,color=cols[mth],label=mth); a2.bar(x+i*w,ap,w,color=cols[mth],label=mth)\n'
      'noft_ef=tab[tab.method=="Boltz-2 initial score (proxy)"].EF1.values[0]; noft_ap=tab[tab.method=="Boltz-2 initial score (proxy)"].AP.values[0]\n'
      'a1.axhline(noft_ef,color=BLK,ls="--",lw=1); a2.axhline(noft_ap,color=BLK,ls="--",lw=1)\n'
      'a1.text(0,noft_ef,f" Boltz-2 initial score {noft_ef:.0f}x (proxy, not No-FT)",va="bottom",fontsize=8)\n'
      'a2.text(0,noft_ap,f" Boltz-2 initial score {noft_ap:.2f} (proxy)",va="bottom",fontsize=8)\n'
      'resv=mpatches.Patch(facecolor="white",edgecolor=BLK,hatch="//",label="head-FT (reserved)")\n'
      'for a,t,yl in [(a1,"EF@1% (x over random)","early enrichment"),(a2,"average precision","AP")]:\n'
      '    a.set_xticks(x+w); a.set_xticklabels(Ns); a.set_xlabel("labels (N)"); a.set_ylabel(t); a.set_title(yl)\n'
      '    h,l=a.get_legend_handles_labels(); a.legend(h+[resv],l+["head-FT (reserved)"],fontsize=8)\n'
      'plt.tight_layout(); plt.show()'),
 md("## LightGBM seed spread (subsampling on)\n"
    "With row/feature subsampling the 5 seeds are genuinely independent, so this is the real "
    "within-condition spread of the descriptor baseline."),
 code('lg=CB[(CB.method=="LightGBM_ECFP")&(CB.seed>=0)]\n'
      'lg.groupby("n_train")[["auprc","ef_1pct","bedroc"]].agg(["mean","std"]).round(4)'),
 md("## Memorization controls\n"
    "Before trusting the enrichment numbers, three checks that decide whether they are real signal or "
    "leakage / analog bias."),
 md("### 1. Disjointness (no leakage)\n"
    "The <=300 fine-tuning labels must not appear in the 49,685-compound eval set."),
 code('import gzip\n'
      'ids=[int(x.strip().removeprefix("588689_")) for x in gzip.open(ROOT/"BoltzFT/data/splits/588689.txt.gz","rt")]\n'
      'rr=pd.read_csv(ROOT/"data/588689_results.csv").set_index("CID").loc[ids].reset_index()\n'
      'train=set(rr.CID[:300]); ev=set(rr.CID[300:])\n'
      'assert train & ev == set(), "LEAK: train/eval overlap"\n'
      'print(f"train(top300)={len(train)}  eval(>300)={len(ev)}  overlap={len(train & ev)}  -> no leakage")'),
 md("### 2. Analog bias\n"
    "Max Tanimoto (ECFP r2) from each held-out active to its nearest training active. If this mass sits "
    "high, the nearest-active baseline wins by recovering analogs rather than by generalizing."),
 code('an=pd.read_csv(ROOT/"results/analysis/cpu_analog_588689.csv")\n'
      'fig,ax=plt.subplots(figsize=(7,5))\n'
      'for i,N in enumerate(sorted(an.n_train.unique())):\n'
      '    v=np.sort(an[an.n_train==N].max_tc.values); cdf=np.arange(1,len(v)+1)/len(v)\n'
      '    ax.plot(v,cdf,color=PALETTE[i],lw=1.6,label=f"N={N}")\n'
      'ax.axvline(0.4,color=BLK,ls="--",lw=1); ax.text(0.41,0.05,"Tc=0.4 (analog)",fontsize=8)\n'
      'ax.set_xlabel("max Tanimoto to a training active"); ax.set_ylabel("cumulative fraction of held-out actives")\n'
      'ax.set_title("analog-bias control: held-out actives vs training actives"); ax.legend()\n'
      'plt.tight_layout(); plt.show()\n'
      'an.groupby("n_train").agg(median_maxTc=("max_tc","median"),\n'
      '   frac_gt_0p4=("max_tc",lambda s:(s>0.4).mean()), frac_gt_0p7=("max_tc",lambda s:(s>0.7).mean()),\n'
      '   frac_share_scaffold=("shares_active_scaffold","mean")).round(3)'),
 md("### 3. Stratified recovery: analog bias, read correctly\n"
    "Two things this does NOT show. It is not a fair No-FT-vs-labels test: the strata are defined by the "
    "300 training labels, so a label-free method (the initial-score proxy) is near-flat across them almost "
    "by construction, and 81% of actives fall in the unseen stratum, so its stratum number approximates its "
    "full number by renormalization. And we do not use stratum-relative EF (that inflates the number by "
    "shrinking the base rate: full 0.80%, unseen 0.66%, low-sim 0.49%). Instead we count recovery from the "
    "COMPLETE-eval ranking (paper convention, ties by eval-ID order).\n\n"
    "What it does show: the nearest-active baseline's enrichment is analog-driven. On the low-similarity "
    "stratum (max Tc < 0.3) it recovers essentially nothing. The fair test - head-FT vs nearest-active at "
    "matched N on these strata - is the reserved comparison the driver will fill."),
 code('ct=pd.read_csv(ROOT/"results/analysis/cpu_controls_588689.csv")\n'
      'from IPython.display import display\n'
      'lab={"Boltz2_initial_score":"Boltz-2 initial (proxy)","NearestActive_Tanimoto":"Nearest-active","LightGBM_ECFP_avg5":"LightGBM-ECFP"}\n'
      'ct["method"]=ct.method.map(lab).fillna(ct.method)\n'
      'N=300; c=ct[ct.n_train==N]\n'
      'sizes=c.groupby("stratum")[["n","n_active","base_rate"]].first().round(4)\n'
      'print(f"stratum sizes / base rates (N={N}):"); display(sizes)\n'
      'piv=c.pivot_table(index="method",columns="stratum",values="rec_top1pct")\n'
      'piv=piv[["full","unseen_scaffold","lowsim_lt0.3"]]\n'
      'print("actives recovered in the GLOBAL top-1% (k=497), by stratum:"); display(piv.astype(int))'),
 code('ms=[m for m in ["Nearest-active","LightGBM-ECFP","Boltz-2 initial (proxy)"] if m in set(c.method)]\n'
      'strata=["full","unseen_scaffold","lowsim_lt0.3"]; x=np.arange(len(strata)); w=0.26\n'
      'fig,ax=plt.subplots(figsize=(8.5,4.6))\n'
      'for i,m in enumerate(ms):\n'
      '    vals=[(c[(c.method==m)&(c.stratum==st)].rec_top1pct.values[:1] or [np.nan])[0] for st in strata]\n'
      '    ax.bar(x+i*w,vals,w,color=PALETTE[i],label=m)\n'
      'ax.set_xticks(x+w); ax.set_xticklabels(["full","unseen scaffold","low-sim <0.3"])\n'
      'ax.set_ylabel("actives recovered in global top-1%"); ax.set_xlabel("held-out stratum (N=300)")\n'
      'ax.set_title("recovery from the full-eval ranking (head-FT reserved)"); ax.legend(fontsize=8)\n'
      'plt.tight_layout(); plt.show()'),
 md("## Secondary (exploratory): confidence signals on the folded subset\n"
    "Ranking the ~21k folded compounds by the structural-confidence metrics — these are not screening "
    "scores, and (as 01 showed) barely beat random. Kept here for contrast with the real scores. "
    "Note this uses the partial folded subset, a different population from the table above."),
 code('signals={"score_boltz2":1,"score_boltzina":1,"ligand_iptm":1,"confidence":1,"complex_plddt":1,\n'
      '         "score_gnina":1,"score_vina":-1}\n'
      'rows=[]\n'
      'for name,sign in signals.items():\n'
      '    if name not in QC or QC[name].notna().sum()<200: continue\n'
      '    d=QC[[name,"Active"]].dropna()\n'
      '    m=all_metrics(d.Active.values.astype(int), sign*d[name].values.astype(float))\n'
      '    rows.append(dict(signal=name, n=len(d), AUROC=m["auroc"], AP=m["auprc"], EF_1pct=m["ef_1pct"], BEDROC=m["bedroc"]))\n'
      'pd.DataFrame(rows).sort_values("AP",ascending=False).round(3).reset_index(drop=True)'),
]
write("02_benchmarking.ipynb", bench)
print("done")

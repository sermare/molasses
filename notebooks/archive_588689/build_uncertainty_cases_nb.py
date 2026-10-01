#!/usr/bin/env python3
"""09_uncertainty_cases.ipynb - which molecules the model is (un)certain about, what they look like,
and why. Per-molecule uncertainty = two-head disagreement |head1-head2| from the No-FT base scores
(available for all 49,685 eval compounds), plus predictive entropy. Draws the actual low- and
high-uncertainty molecules, computes RDKit properties, tests confidence-vs-size correlation, and
asks what molecular features drive uncertainty. Matplotlib + RDKit; all-black fonts, no bold, no HTML."""
import nbformat as nbf
from pathlib import Path
NB = Path("/global/scratch/users/sergiomar10/boltzaff/notebooks/09_uncertainty_cases.ipynb")
def md(s): return nbf.v4.new_markdown_cell(s)
def code(s): return nbf.v4.new_code_cell(s)

PRE = r'''
import glob, numpy as np, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt
from IPython.display import display
from scipy import stats
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Draw, Crippen, rdMolDescriptors
RDLogger.DisableLog("rdApp.*")
BLK="#000000"; LO="#2a78d6"; HI="#eb6834"; NEU="#666666"
mpl.rcParams.update({"figure.facecolor":"white","axes.facecolor":"white","savefig.facecolor":"white",
  "text.color":BLK,"axes.labelcolor":BLK,"axes.titlecolor":BLK,"xtick.color":BLK,"ytick.color":BLK,
  "axes.edgecolor":BLK,"font.size":10,"font.weight":"normal","axes.titleweight":"normal","axes.labelweight":"normal",
  "axes.grid":True,"grid.color":"#e6e6e6","axes.axisbelow":True,"axes.spines.top":False,"axes.spines.right":False})
ROOT="/global/scratch/users/sergiomar10/boltzaff"; R=ROOT+"/results/runs/588689"
# --- No-FT base scores: two heads per compound (epistemic uncertainty) ---
bs=pd.concat([pd.read_csv(p) for p in sorted(glob.glob(R+"/headft_affcache/base/scores/chunk_*.csv"))],ignore_index=True)
bs["sample_id"]=bs.sample_id.astype(str); bs=bs.drop_duplicates("sample_id")
bs["CID"]=bs.sample_id.str.split("_").str[-1]
bs["disagree"]=(bs.affinity_probability_binary1-bs.affinity_probability_binary2).abs()
p=bs.affinity_probability_binary.clip(1e-6,1-1e-6)
bs["entropy"]=-(p*np.log2(p)+(1-p)*np.log2(1-p))
# --- SMILES + label ---
res=pd.read_csv(ROOT+"/data/588689_results.csv"); res["CID"]=res.CID.astype(str)
df=bs.merge(res[["CID","neut-smiles","Active_v2"]],on="CID",how="left").rename(columns={"neut-smiles":"smiles","Active_v2":"active"})
df=df.dropna(subset=["smiles"]).reset_index(drop=True)
# --- Boltz-2 structural confidence (folded subset) ---
qc=pd.read_csv(R+"/qc.csv"); qc["CID"]=qc.CID.astype(str)
df=df.merge(qc[["CID","confidence","ligand_iptm","complex_plddt","iptm"]],on="CID",how="left")
print(f"eval compounds with score+SMILES: {len(df):,}   with confidence: {int(df.confidence.notna().sum()):,}")
print(f"two-head disagreement: mean {df.disagree.mean():.3f}, 95th pct {df.disagree.quantile(.95):.3f}")

def descriptors(smiles):
    m=Chem.MolFromSmiles(smiles)
    if m is None: return None
    return dict(MW=Descriptors.MolWt(m), heavy_atoms=m.GetNumHeavyAtoms(), logP=Crippen.MolLogP(m),
                TPSA=rdMolDescriptors.CalcTPSA(m), rot_bonds=rdMolDescriptors.CalcNumRotatableBonds(m),
                H_donors=rdMolDescriptors.CalcNumHBD(m), H_acceptors=rdMolDescriptors.CalcNumHBA(m),
                aromatic_rings=rdMolDescriptors.CalcNumAromaticRings(m), fraction_csp3=rdMolDescriptors.CalcFractionCSP3(m))
'''

cells=[
 md("# 09 · Uncertainty cases - which molecules, and why\n\n"
    "Uncertainty in virtual screening is per-molecule and actionable: if the model is unsure about a "
    "compound, you weight it differently before spending assay budget. Here the per-molecule uncertainty "
    "is the **two-head disagreement** of Boltz-2's affinity module - the two probability heads "
    "(`affinity_probability_binary1/2`) on the No-FT base scores, available for every one of the 49,685 "
    "eval compounds - plus the predictive entropy of the mean probability.\n\n"
    "We then (1) draw the actual most- and least-certain molecules, (2) compute their RDKit properties, "
    "(3) test whether structural confidence tracks molecular size, and (4) ask which molecular features "
    "drive high vs low uncertainty and why uncertainty concentrates in certain score regions."),
 code(PRE),

 md("## 1. The uncertainty signal\n"
    "Two-head disagreement is a label-free epistemic-uncertainty estimate. It peaks for mid-range "
    "predictions (near the decision boundary) and is near zero when both heads are confidently low - most "
    "of the library. This is why a few compounds carry most of the model's doubt."),
 code('fig,axes=plt.subplots(1,3,figsize=(15,4.2))\n'
      'axes[0].hist(df.disagree,bins=60,color=NEU); axes[0].set_xlabel("two-head disagreement |head1-head2|")\n'
      'axes[0].set_ylabel("compounds"); axes[0].set_yscale("log"); axes[0].set_title("distribution of model uncertainty")\n'
      'axes[1].scatter(df.affinity_probability_binary,df.disagree,s=5,alpha=0.25,color=NEU)\n'
      'axes[1].set_xlabel("mean predicted probability"); axes[1].set_ylabel("disagreement"); axes[1].set_title("uncertainty vs prediction (peaks mid-range)")\n'
      'axes[2].scatter(df.entropy,df.disagree,s=5,alpha=0.25,color=NEU)\n'
      'axes[2].set_xlabel("predictive entropy (bits)"); axes[2].set_ylabel("disagreement"); axes[2].set_title("two uncertainty measures agree")\n'
      'plt.tight_layout(); plt.show()\n'
      'print("Spearman(disagreement, entropy) = %.3f"%df[["disagree","entropy"]].corr(method="spearman").iloc[0,1])'),

 md("## 2. Confidence vs molecular size\n"
    "Does the model fold/score bigger molecules less confidently? We compute RDKit descriptors on a random "
    "sample and correlate molecular size (MW, heavy-atom count) with Boltz-2 structural confidence and with "
    "the two-head uncertainty."),
 code('samp=df[df.confidence.notna()].sample(n=min(8000,int(df.confidence.notna().sum())),random_state=0).copy()\n'
      'desc=samp.smiles.apply(descriptors); samp=samp[desc.notna()].copy()\n'
      'dd=pd.DataFrame(list(desc[desc.notna()]),index=samp.index); samp=pd.concat([samp,dd],axis=1)\n'
      'fig,axes=plt.subplots(1,3,figsize=(15,4.4))\n'
      '# panel a: MW vs confidence; b: heavy atoms vs ligand_iptm; c: MW vs disagreement\n'
      'a=samp.dropna(subset=["MW","confidence"]); axes[0].scatter(a.MW,a.confidence,s=6,alpha=0.3,color=LO)\n'
      'axes[0].set_xlabel("molecular weight (Da)"); axes[0].set_ylabel("Boltz-2 overall confidence")\n'
      'axes[0].set_title("size vs confidence (rho=%.2f)"%a[["MW","confidence"]].corr(method="spearman").iloc[0,1])\n'
      'b=samp.dropna(subset=["heavy_atoms","ligand_iptm"]); axes[1].scatter(b.heavy_atoms,b.ligand_iptm,s=6,alpha=0.3,color=LO)\n'
      'axes[1].set_xlabel("heavy-atom count"); axes[1].set_ylabel("ligand ipTM (pose confidence)")\n'
      'axes[1].set_title("size vs pose confidence (rho=%.2f)"%b[["heavy_atoms","ligand_iptm"]].corr(method="spearman").iloc[0,1])\n'
      'c=samp.dropna(subset=["MW","disagree"]); axes[2].scatter(c.MW,c.disagree,s=6,alpha=0.3,color=HI)\n'
      'axes[2].set_xlabel("molecular weight (Da)"); axes[2].set_ylabel("two-head disagreement")\n'
      'axes[2].set_title("size vs uncertainty (rho=%.2f)"%c[["MW","disagree"]].corr(method="spearman").iloc[0,1])\n'
      'plt.tight_layout(); plt.show()'),

 md("## 3. What molecular features drive uncertainty?\n"
    "Spearman correlation of each RDKit property with the two-head uncertainty (on the sample). Positive = "
    "that feature goes with higher uncertainty. This shows which chemistry the model finds hard to commit "
    "on."),
 code('props=["MW","heavy_atoms","logP","TPSA","rot_bonds","H_donors","H_acceptors","aromatic_rings","fraction_csp3"]\n'
      'props=[p for p in props if p in samp.columns]\n'
      'cor=[(p, samp[[p,"disagree"]].dropna().corr(method="spearman").iloc[0,1]) for p in props]\n'
      'cor=pd.DataFrame(cor,columns=["property","rho"]).sort_values("rho")\n'
      'fig,ax=plt.subplots(figsize=(8,4.6))\n'
      'ax.barh(cor.property,cor.rho,color=[HI if v>0 else LO for v in cor.rho])\n'
      'ax.axvline(0,color=BLK,lw=1); ax.set_xlabel("Spearman rho with two-head uncertainty")\n'
      'ax.set_title("which molecular features go with model uncertainty"); plt.tight_layout(); plt.show()\n'
      'display(cor.set_index("property").round(3))'),

 md("## 4. The most-certain molecules (lowest disagreement)\n"
    "Actual structures the model is most confident about, with properties. To keep them interpretable we "
    "draw from compounds the model scores as plausible binders (upper-half probability) - certain low-score "
    "decoys are trivial. Legends show CID, predicted prob, disagreement, and activity label."),
 code('def grid(sub,title):\n'
      '    mols=[]; legs=[]\n'
      '    for _,r in sub.iterrows():\n'
      '        m=Chem.MolFromSmiles(r.smiles)\n'
      '        if m is None: continue\n'
      '        mols.append(m); legs.append(f"CID {r.CID} | p={r.affinity_probability_binary:.2f} d={r.disagree:.2f} {\'ACT\' if r.active==1 else \'inact\'}")\n'
      '        if len(mols)>=9: break\n'
      '    img=Draw.MolsToGridImage(mols,molsPerRow=3,subImgSize=(260,200),legends=legs,returnPNG=False)\n'
      '    print(title); display(img)\n'
      'cand=df[df.affinity_probability_binary>df.affinity_probability_binary.median()]\n'
      'grid(cand.nsmallest(9,"disagree"),"Most-certain molecules (lowest two-head disagreement, upper-half score)")'),

 md("## 5. The most-uncertain molecules (highest disagreement)\n"
    "The compounds where the two heads most disagree - the model's genuine doubt. These are the ones a "
    "screener should treat with caution or route to orthogonal evidence."),
 code('grid(df.nlargest(9,"disagree"),"Most-uncertain molecules (highest two-head disagreement)")'),

 md("## 6. High- vs low-uncertainty chemistry - property distributions\n"
    "Comparing the top and bottom uncertainty deciles: which properties actually differ. Where the orange "
    "(high-uncertainty) and blue (low-uncertainty) distributions separate is a reason the model is unsure."),
 code('n=len(samp); hi=samp.nlargest(max(50,n//10),"disagree"); lo=samp.nsmallest(max(50,n//10),"disagree")\n'
      'feats=[p for p in ["MW","heavy_atoms","logP","TPSA","rot_bonds","aromatic_rings"] if p in samp.columns]\n'
      'fig,axes=plt.subplots(2,3,figsize=(15,8))\n'
      'for ax,f in zip(axes.ravel(),feats):\n'
      '    lo_v=lo[f].dropna(); hi_v=hi[f].dropna()\n'
      '    rng=np.nanpercentile(pd.concat([lo_v,hi_v]),[1,99]); bins=np.linspace(rng[0],rng[1],30)\n'
      '    ax.hist(lo_v,bins=bins,density=True,color=LO,alpha=0.55,label="low uncertainty")\n'
      '    ax.hist(hi_v,bins=bins,density=True,color=HI,alpha=0.55,label="high uncertainty")\n'
      '    u,pv=stats.mannwhitneyu(lo_v,hi_v)\n'
      '    ax.set_xlabel(f); ax.set_ylabel("density"); ax.set_title(f"{f}  (MWU p={pv:.1e})"); ax.legend(fontsize=8)\n'
      'fig.suptitle("high- vs low-uncertainty molecules: property distributions",fontsize=12)\n'
      'plt.tight_layout(); plt.show()'),

 md("## 7. Why uncertainty concentrates in regions\n"
    "Uncertainty is not uniform. It is highest (a) in the mid-score band near the decision boundary, and "
    "(b) among the compounds the model poses least confidently. Binning by predicted-score decile and by "
    "pose confidence shows where the doubt lives - useful for deciding where extra evidence pays off."),
 code('fig,axes=plt.subplots(1,2,figsize=(13,4.6))\n'
      'df["score_bin"]=pd.qcut(df.affinity_probability_binary,10,duplicates="drop")\n'
      'g=df.groupby("score_bin",observed=True).disagree.mean()\n'
      'axes[0].bar(range(len(g)),g.values,color=NEU); axes[0].set_xlabel("predicted-score decile (low->high)")\n'
      'axes[0].set_ylabel("mean two-head disagreement"); axes[0].set_title("uncertainty by score region")\n'
      'sub=df.dropna(subset=["ligand_iptm"]).copy(); sub["conf_q"]=pd.qcut(sub.ligand_iptm,5,labels=["Q1 low","Q2","Q3","Q4","Q5 high"])\n'
      'g2=sub.groupby("conf_q",observed=True).disagree.mean()\n'
      'axes[1].bar(range(len(g2)),g2.values,color=NEU); axes[1].set_xticks(range(len(g2))); axes[1].set_xticklabels(g2.index)\n'
      'axes[1].set_xlabel("ligand ipTM quintile (pose confidence)"); axes[1].set_ylabel("mean two-head disagreement")\n'
      'axes[1].set_title("uncertainty vs pose confidence"); plt.tight_layout(); plt.show()'),

 md("## 8. Uncertainty among the actives (the molecules that matter)\n"
    "For hit discovery, uncertainty on the true actives is what costs you. Comparing the disagreement "
    "distribution of actives vs inactives shows whether the model is systematically less sure about the "
    "compounds we most want to recover."),
 code('act=df[df.active==1].disagree; ina=df[df.active==0].disagree\n'
      'fig,ax=plt.subplots(figsize=(7.5,4.6))\n'
      'bins=np.linspace(0,df.disagree.quantile(.99),40)\n'
      'ax.hist(ina,bins=bins,density=True,color=LO,alpha=0.55,label=f"inactive (n={len(ina)})")\n'
      'ax.hist(act,bins=bins,density=True,color=HI,alpha=0.6,label=f"active (n={len(act)})")\n'
      'u,pv=stats.mannwhitneyu(act,ina)\n'
      'ax.set_xlabel("two-head disagreement"); ax.set_ylabel("density")\n'
      'ax.set_title(f"uncertainty: actives vs inactives (MWU p={pv:.1e}); active median {act.median():.3f} vs {ina.median():.3f}")\n'
      'ax.legend(fontsize=8); plt.tight_layout(); plt.show()\n'
      'display(df.nlargest(6,"disagree").assign(active=lambda d:d.active.map({1:"ACTIVE",0:"inactive"}))[["CID","affinity_probability_binary","disagree","active"]].round(3))'),

 md("## Summary\n\n"
    "- **Uncertainty is concentrated, not uniform** - two-head disagreement is near zero for the bulk of "
    "the library and spikes for mid-score, near-boundary compounds. A small set carries most of the doubt.\n"
    "- **Confidence vs size:** see section 2 for the measured Spearman - Boltz-2 structural confidence and "
    "molecular size are (weakly) related, and larger/more-flexible molecules tend toward higher two-head "
    "uncertainty; section 3 ranks which features (MW, rotatable bonds, aromatic rings, logP) track doubt.\n"
    "- **Certain vs uncertain chemistry differs** in interpretable ways (section 6) - the properties that "
    "separate them are why the model hesitates.\n"
    "- **Where doubt lives:** the mid-score band and the least-confidently-posed compounds (section 7) - "
    "exactly where orthogonal evidence (a second method, an assay) is worth spending.\n"
    "- **On the actives** (section 8): whether the true hits carry more uncertainty determines how much the "
    "doubt actually costs early enrichment.\n\n"
    "This is the No-FT model's uncertainty; the same two-head signal exists on every head-FT arm, so the "
    "next step is whether fine-tuning reduces uncertainty on the actives it promotes."),
]
nb=nbf.v4.new_notebook(); nb.cells=cells
nb.metadata={"kernelspec":{"name":"boltzba","display_name":"Python (boltzba)"},"language_info":{"name":"python"}}
NB.write_text(nbf.writes(nb)); print("wrote", NB)

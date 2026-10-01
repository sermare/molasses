#!/usr/bin/env python3
"""Build the 00_* data notebooks for the BoltzFT reproduction (nbformat v4).

Run with an env that has nbformat (boltzba after `pip install nbformat`).
Notebooks are written to notebooks/ and then executed separately with nbconvert.
"""
import nbformat as nbf
from pathlib import Path

NBDIR = Path("/global/scratch/users/sergiomar10/boltzaff/notebooks")
NBDIR.mkdir(parents=True, exist_ok=True)


def md(s): return nbf.v4.new_markdown_cell(s)
def code(s): return nbf.v4.new_code_cell(s)


def write(name, cells):
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                   "language_info": {"name": "python"}}
    (NBDIR / name).write_text(nbf.writes(nb))
    print("wrote", name)


# Shared preamble cell reused across notebooks
PREAMBLE = '''\
import os, gzip, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
pd.set_option("display.width", 160, "display.max_columns", 40)

ROOT = Path("/global/scratch/users/sergiomar10/boltzaff")
RAW  = ROOT / "data" / "mf-pcba_test"       # distributed per-target CSVs
DATA = ROOT / "data"                        # our cleaned CSVs + seqs + MSAs
SPLITS = ROOT / "BoltzFT" / "data" / "splits"

# target folder -> (protein, source structure/GI)
TARGETS = {
 "588689":         ("Dengue-2 NS5 MTase",      "PDB 3EVG"),
 "540297-493091":  ("SCP1 phosphatase",        "PDB 3PGL"),
 "434954-2097":    ("GSK-3\\u03b2",               "PDB 1J1B"),
 "463203-2650":    ("GSK-3\\u03b1",               "PDB 7SXF"),
 "493248-485317":  ("ALR / GFER oxidase",      "PDB 3MBG"),
 "504329":         ("Influenza A NS1",         "predicted (GI 227977143)"),
 "624273-588549":  ("M.tb FadD28 ligase",      "predicted (GI 1781172)"),
 "1053173-743445": ("NSD2 PWWP1 domain",       "PDB 6UE6"),
}

# Paper Table 1 (Furui & Ohue, arXiv:2609.24302), for validation.
# target -> dict(neval, nactive_eval, active_rate_pct, train_actives {40,100,300})
PAPER_T1 = {
 "588689":        dict(neval=49685, nact=396, rate=0.80, tr={40:16,100:32,300:90}),
 "540297-493091": dict(neval=49685, nact=739, rate=1.49, tr={40:8, 100:16,300:43}),
 "434954-2097":   dict(neval=49615, nact=415, rate=0.84, tr={40:24,100:52,300:107}),
 "463203-2650":   dict(neval=49582, nact=542, rate=1.09, tr={40:13,100:29,300:70}),
 "493248-485317": dict(neval=49680, nact=953, rate=1.92, tr={40:1, 100:9, 300:23}),
 "504329":        dict(neval=49682, nact=438, rate=0.88, tr={40:4, 100:11,300:28}),
 "624273-588549": dict(neval=49687, nact=150, rate=0.30, tr={40:4, 100:4, 300:9}),
 "1053173-743445":dict(neval=49695, nact=134, rate=0.27, tr={40:0, 100:5, 300:10}),
}

def load_split_order(t):
    """Return the score_boltz2-descending CID ranking shipped in data/splits."""
    p = SPLITS / f"{t}.txt.gz"
    with gzip.open(p, "rt") as f:
        return [int(x.strip().removeprefix(t + "_")) for x in f]
'''

# Shared chart style + palette (validated categorical order from the dataviz skill).
STYLE = '''\
import matplotlib as mpl, matplotlib.pyplot as plt
PALETTE = ["#2a78d6","#eb6834","#1baf7a","#eda100","#e87ba4","#008300","#4a3aa7","#e34948"]
INK, MUTED, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
ACTIVE, INACTIVE, BLUE = "#e34948", "#b8b6ad", "#2a78d6"
TCOLOR = {t: PALETTE[i] for i, t in enumerate(TARGETS)}   # fixed hue per target
mpl.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10,
    "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 100})
'''

# RDKit descriptor helper (sampled, so it runs fast across 8 targets).
DESC = '''\
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors, Draw
from rdkit.Chem.Scaffolds import MurckoScaffold

def desc_frame(t, n_inact=1200, seed=0):
    df = pd.read_csv(DATA / f"{t}_results.csv")
    act = df[df.Active_v2 == 1]
    inact_all = df[df.Active_v2 == 0]
    ina = inact_all.sample(min(n_inact, len(inact_all)), random_state=seed)
    rows = []
    for _, r in pd.concat([act, ina]).iterrows():
        m = Chem.MolFromSmiles(str(r["neut-smiles"]))
        if m is None: continue
        rows.append(dict(active=int(r.Active_v2), MW=Descriptors.MolWt(m),
            HAC=m.GetNumHeavyAtoms(), cLogP=Descriptors.MolLogP(m),
            TPSA=rdMolDescriptors.CalcTPSA(m), HBD=Lipinski.NumHDonors(m),
            HBA=Lipinski.NumHAcceptors(m), RotB=Descriptors.NumRotatableBonds(m),
            Rings=rdMolDescriptors.CalcNumRings(m)))
    return pd.DataFrame(rows)
'''

# ---------------------------------------------------------------- 00_data_cleanup
cleanup = [
 md("# 00 · Data cleanup\n\n"
    "The Boltzina v1.0.1 release ships one CSV per MF-PCBA target with compound SMILES, "
    "binary activity, and cached Boltz-2 scores. The BoltzFT pipeline expects a specific "
    "schema (`CID`, `neut-smiles`, `Active_v2`, `affinity_probability_binary`). This notebook "
    "documents the exact, reproducible transformation from the **distributed** columns to that "
    "schema, and validates that no compounds are silently lost."),
 code(PREAMBLE),
 code(STYLE),
 md("## Raw distributed schema\nColumns as shipped for target 588689."),
 code('t = "588689"\n'
      'raw = pd.read_csv(RAW / f"{t}.csv")\n'
      'print("shape:", raw.shape)\n'
      'print("columns:", list(raw.columns))\n'
      'raw.head(3)'),
 md("## The cleanup mapping\n"
    "- `target_active_v2` (bool) &rarr; **`Active_v2`** (int 0/1)\n"
    "- `score_boltz2` (standard Boltz-2 binding probability) &rarr; **`affinity_probability_binary`**\n"
    "- drop rows with missing score or SMILES; drop duplicate CIDs (keep first)\n"
    "- reindex to the shipped `data/splits/<target>.txt.gz` ranking (Boltz-2 score descending)\n\n"
    "We confirmed the split file order **is** `score_boltz2` descending, so the ranking used to "
    "pick the top-N training compounds is exactly the standard Boltz-2 ranking."),
 code('def clean(t):\n'
      '    raw = pd.read_csv(RAW / f"{t}.csv")\n'
      '    out = pd.DataFrame({\n'
      '        "CID": raw["CID"].astype(int),\n'
      '        "neut-smiles": raw["neut-smiles"],\n'
      '        "Active_v2": raw["target_active_v2"].astype(bool).astype(int),\n'
      '        "affinity_probability_binary": raw["score_boltz2"],\n'
      '    })\n'
      '    n0 = len(out)\n'
      '    out = out.dropna(subset=["affinity_probability_binary", "neut-smiles"]).drop_duplicates("CID", keep="first")\n'
      '    order = load_split_order(t)\n'
      '    missing = set(order) - set(out.CID)\n'
      '    out = out.set_index("CID").loc[order].reset_index()\n'
      '    return out, dict(raw_rows=n0, after_clean=len(out), dropped=n0-len(out),\n'
      '                     split_ids=len(order), missing_from_clean=len(missing))\n'
      '\n'
      'out, info = clean("588689")\n'
      'print(info)\n'
      'out.head(3)'),
 md("## Verify the shipped `*_results.csv` matches this transform\n"
    "The pipeline consumes `data/<target>_results.csv`, produced by exactly this mapping."),
 code('# Align on CID (the shipped CSV is in raw order; clean() reindexes to the split ranking,\n'
      '# but select_subsets/build_ft_inputs reindex by the split anyway, so order is cosmetic).\n'
      's = pd.read_csv(DATA / "588689_results.csv").set_index("CID").sort_index()\n'
      'r = clean("588689")[0].set_index("CID").sort_index()\n'
      'assert s.index.equals(r.index), "CID sets differ"\n'
      'assert np.allclose(s["affinity_probability_binary"], r["affinity_probability_binary"])\n'
      'assert (s["Active_v2"] == r["Active_v2"]).all()\n'
      'print(f"shipped 588689_results.csv reproduces the cleanup exactly \\u2713  ({len(s)} compounds, CID-aligned)")'),
 md("## Cleanup summary across all 8 targets\nNo compounds are dropped and every split ID resolves."),
 code('rows = []\n'
      'for t in TARGETS:\n'
      '    _, info = clean(t)\n'
      '    rows.append({"target": t, "protein": TARGETS[t][0], **info})\n'
      'summary = pd.DataFrame(rows)\n'
      'summary'),
 md("## Column completeness in the raw data\n"
    "Fraction of non-missing values per raw column (588689). Boltz-2 / Boltzina scores are complete; "
    "some comparator columns (e.g. GNINA/Vina) are sparse or absent \\u2014 which is why the pipeline keys on "
    "`score_boltz2` and the binary label rather than the docking comparators."),
 code(STYLE + '\n'
      'raw = pd.read_csv(RAW / "588689.csv")\n'
      'frac = raw.notna().mean().sort_values()\n'
      'fig,ax=plt.subplots(figsize=(8,6))\n'
      'ax.barh(range(len(frac)), frac.values, color=BLUE)\n'
      'ax.set_yticks(range(len(frac))); ax.set_yticklabels(frac.index, fontsize=8)\n'
      'ax.set_xlim(0,1); ax.set_xlabel("fraction non-missing"); ax.set_title("588689 raw column completeness")\n'
      'plt.tight_layout(); plt.show()'),
]
write("00_data_cleanup.ipynb", cleanup)

# ---------------------------------------------------------------- 00_data_stats
stats = [
 md("# 00 · Data statistics\n\n"
    "Per-target library statistics for the 8 MF-PCBA targets, the nested train/eval split used "
    "for fine-tuning, and validation against the paper's Table 1. Ends with a look at the "
    "**highest-affinity molecules** (top Boltz-2 binding probability) and confirmed actives."),
 code(PREAMBLE),
 code(STYLE),
 code('def target_frame(t):\n'
      '    df = pd.read_csv(DATA / f"{t}_results.csv")\n'
      '    order = load_split_order(t)\n'
      '    df = df.set_index("CID").loc[order].reset_index()   # Boltz-2 ranked\n'
      '    df["rank"] = np.arange(1, len(df) + 1)\n'
      '    return df\n'
      'frames = {t: target_frame(t) for t in TARGETS}\n'
      'print("loaded", len(frames), "targets")'),
 md("## Library size, actives, and the nested split\n"
    "Training compounds are the **top-N** of the Boltz-2 ranking (N=40/100/300, nested); the "
    "common evaluation set excludes the top 300. We compare our counts to the paper's Table 1."),
 code('rows = []\n'
      'for t, df in frames.items():\n'
      '    total, act = len(df), int(df.Active_v2.sum())\n'
      '    eval_df = df.iloc[300:]\n'
      '    tr = {n: int(df.iloc[:n].Active_v2.sum()) for n in (40, 100, 300)}\n'
      '    p = PAPER_T1[t]\n'
      '    rows.append({"target": t, "protein": TARGETS[t][0],\n'
      '        "n_total": total, "n_active": act, "active_rate_%": round(100*act/total, 2),\n'
      '        "n_eval": len(eval_df), "n_eval(paper)": p["neval"],\n'
      '        "nact_eval": int(eval_df.Active_v2.sum()), "nact_eval(paper)": p["nact"],\n'
      '        "tr40": tr[40], "tr100": tr[100], "tr300": tr[300],\n'
      '        "tr(paper)": "/".join(str(p["tr"][n]) for n in (40,100,300))})\n'
      'stat = pd.DataFrame(rows)\n'
      'stat'),
 code('# Hard validation against the paper\n'
      'ok = True\n'
      'for t, df in frames.items():\n'
      '    p = PAPER_T1[t]; eval_df = df.iloc[300:]\n'
      '    ok &= (len(eval_df) == p["neval"]) and (int(eval_df.Active_v2.sum()) == p["nact"])\n'
      '    ok &= all(int(df.iloc[:n].Active_v2.sum()) == p["tr"][n] for n in (40,100,300))\n'
      'print("all 8 targets match paper Table 1 (neval, nactive, train actives):", ok)'),
 md("## Active rate per target\nMF-PCBA is dominated by inactives (0.27%–1.92% active) — this is why "
    "*early* enrichment matters."),
 code('import matplotlib.pyplot as plt\n'
      'rates = stat.set_index("target")["active_rate_%"].sort_values()\n'
      'fig, ax = plt.subplots(figsize=(8,3.5))\n'
      'ax.barh(range(len(rates)), rates.values, color="#4C78A8")\n'
      'ax.set_yticks(range(len(rates))); ax.set_yticklabels(rates.index)\n'
      'ax.set_xlabel("active rate (%)"); ax.set_title("MF-PCBA active rate by target")\n'
      'for i,v in enumerate(rates.values): ax.text(v+0.02, i, f"{v:.2f}", va="center", fontsize=8)\n'
      'plt.tight_layout(); plt.show()'),
 md("## Where do the actives fall in the Boltz-2 (No-FT) ranking?\n"
    "The cumulative recovery of actives vs. rank. A curve bowed toward the top-left means the "
    "off-the-shelf Boltz-2 score already enriches actives early — the baseline that fine-tuning "
    "aims to improve."),
 code('fig, ax = plt.subplots(figsize=(7,5))\n'
      'for t, df in frames.items():\n'
      '    y = np.cumsum(df.Active_v2.values) / df.Active_v2.sum()\n'
      '    ax.plot(np.arange(1,len(df)+1)/len(df)*100, y*100, label=f"{t} ({TARGETS[t][0]})", lw=1.3)\n'
      'ax.plot([0,100],[0,100], "k--", lw=0.8, label="random")\n'
      'ax.set_xlabel("% of ranked library screened"); ax.set_ylabel("% actives recovered")\n'
      'ax.set_title("No-FT Boltz-2: cumulative active recovery"); ax.set_xlim(0,20); ax.legend(fontsize=7)\n'
      'plt.tight_layout(); plt.show()'),
 md("## Boltz-2 score distribution: actives vs inactives (588689)"),
 code('t="588689"; df=frames[t]\n'
      'fig,ax=plt.subplots(figsize=(7,3.5))\n'
      'ax.hist(df[df.Active_v2==0].affinity_probability_binary, bins=60, alpha=.6, density=True, label="inactive", color="#B0B0B0")\n'
      'ax.hist(df[df.Active_v2==1].affinity_probability_binary, bins=60, alpha=.7, density=True, label="active", color="#E45756")\n'
      'ax.set_xlabel("Boltz-2 binding probability (score_boltz2)"); ax.set_ylabel("density")\n'
      'ax.set_title(f"{t} — {TARGETS[t][0]}: score by activity"); ax.legend()\n'
      'plt.tight_layout(); plt.show()'),
 md("## High-affinity molecules\n"
    "The top compounds by Boltz-2 binding probability for 588689. Green = confirmed active, "
    "red = inactive (a false positive at the top of the No-FT ranking — exactly the compounds "
    "whose labels fine-tuning learns from)."),
 code('from rdkit import Chem\n'
      'from rdkit.Chem import Draw, Descriptors\n'
      'from rdkit.Chem.Draw import rdMolDraw2D\n'
      'import warnings; warnings.filterwarnings("ignore")\n'
      '\n'
      't="588689"; df=frames[t]\n'
      'top = df.head(12)\n'
      'mols, legends, colors = [], [], []\n'
      'for _, r in top.iterrows():\n'
      '    m = Chem.MolFromSmiles(str(r["neut-smiles"]))\n'
      '    if m is None: continue\n'
      '    mols.append(m)\n'
      '    rk = int(r["rank"]); lab = "ACTIVE" if r.Active_v2==1 else "inactive"\n'
      '    legends.append(f"rank {rk} | p={r.affinity_probability_binary:.3f} | {lab}")\n'
      'img = Draw.MolsToGridImage(mols, molsPerRow=4, subImgSize=(240,200), legends=legends)\n'
      'img'),
 md("## Highest-scoring **confirmed actives** (588689)\n"
    "The true positives Boltz-2 already ranks well — chemically, what this target's binders look like."),
 code('act = df[df.Active_v2==1].head(8)\n'
      'mols=[Chem.MolFromSmiles(str(s)) for s in act["neut-smiles"]]\n'
      'legs=[f"rank {int(r)} | p={p:.3f}" for r,p in zip(act["rank"], act.affinity_probability_binary)]\n'
      'mols=[m for m in mols if m]\n'
      'Draw.MolsToGridImage(mols, molsPerRow=4, subImgSize=(240,200), legends=legs)'),
 md("## Molecular property ranges of actives (all targets)\n"
    "Heavy-atom count and MW — a sanity check on the compound libraries (the two excluded MF-PCBA "
    "assays were dropped for >60 heavy-atom actives / unspecified pockets)."),
 code('rows=[]\n'
      'for t, df in frames.items():\n'
      '    a = df[df.Active_v2==1]\n'
      '    hs, mws = [], []\n'
      '    for s in a["neut-smiles"].head(200):\n'
      '        m = Chem.MolFromSmiles(str(s))\n'
      '        if m: hs.append(m.GetNumHeavyAtoms()); mws.append(Descriptors.MolWt(m))\n'
      '    rows.append({"target": t, "protein": TARGETS[t][0],\n'
      '                 "heavy_med": int(np.median(hs)), "heavy_max": int(np.max(hs)),\n'
      '                 "MW_med": round(np.median(mws),1)})\n'
      'pd.DataFrame(rows)'),
 # ---------------- richer data visualizations ----------------
 md("---\n## More data visualizations"),
 code(DESC),
 md("### Score separability by target\n"
    "How well the off-the-shelf Boltz-2 score already separates actives from inactives, per target. "
    "The more the red (active) mass sits right of the grey (inactive), the easier the target."),
 code('fig, axes = plt.subplots(2,4, figsize=(15,6), sharex=True)\n'
      'for ax,(t,df) in zip(axes.ravel(), frames.items()):\n'
      '    ax.hist(df[df.Active_v2==0].affinity_probability_binary, bins=40, density=True, color=INACTIVE, alpha=.75)\n'
      '    ax.hist(df[df.Active_v2==1].affinity_probability_binary, bins=40, density=True, color=ACTIVE, alpha=.75)\n'
      '    ax.set_title(f"{t}\\n{TARGETS[t][0]}", fontsize=8, color=TCOLOR[t])\n'
      'axes[0,0].legend(["inactive","active"], fontsize=8)\n'
      'fig.suptitle("Boltz-2 (No-FT) binding-probability distribution: actives vs inactives", fontsize=12)\n'
      'fig.supxlabel("binding probability"); plt.tight_layout(); plt.show()'),
 md("### How good is the No-FT ranking already?\n"
    "AUROC (separation) and EF@1% (early enrichment over random) for the off-the-shelf Boltz-2 score. "
    "This is the baseline that fine-tuning must beat \\u2014 the paper reports geo-mean EF@1% rising from "
    "~10.9 (No-FT) to ~19.6 at N=300."),
 code('from sklearn.metrics import roc_auc_score\n'
      'rows=[]\n'
      'for t,df in frames.items():\n'
      '    y=df.Active_v2.values.astype(int); s=df.affinity_probability_binary.values\n'
      '    k=max(1,round(0.01*len(y))); order=np.argsort(-s)\n'
      '    ef=float(y[order][:k].mean()/y.mean())\n'
      '    rows.append(dict(target=t, auroc=roc_auc_score(y,s), ef1=ef))\n'
      'mm=pd.DataFrame(rows)\n'
      'fig,(a1,a2)=plt.subplots(1,2,figsize=(13,4.2))\n'
      'cols=[TCOLOR[t] for t in mm.target]; x=range(8)\n'
      'a1.bar(x, mm.auroc, color=cols); a1.axhline(0.5,color=MUTED,ls="--",lw=.9)\n'
      'a1.set_ylim(0,1); a1.set_title("No-FT AUROC (0.5 = random)")\n'
      'a1.set_xticks(list(x)); a1.set_xticklabels(mm.target, rotation=45, ha="right", fontsize=7)\n'
      'a2.bar(x, mm.ef1, color=cols); a2.axhline(1,color=MUTED,ls="--",lw=.9)\n'
      'a2.set_title("No-FT EF@1% (\\u00d7 over random)")\n'
      'a2.set_xticks(list(x)); a2.set_xticklabels(mm.target, rotation=45, ha="right", fontsize=7)\n'
      'for i,v in enumerate(mm.ef1): a2.text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8)\n'
      'plt.tight_layout(); plt.show()\n'
      'mm.round(3)'),
 md("### Where do the actives sit in the ranking?\n"
    "Histogram of active ranks (log x). Mass toward the left = early enrichment. Dashed line = the "
    "top-300 training cutoff (everything left of it is excluded from evaluation)."),
 code('fig,axes=plt.subplots(2,4,figsize=(15,6))\n'
      'for ax,(t,df) in zip(axes.ravel(), frames.items()):\n'
      '    ranks=np.where(df.Active_v2.values==1)[0]+1\n'
      '    ax.hist(ranks, bins=np.logspace(0, np.log10(len(df)), 40), color=TCOLOR[t])\n'
      '    ax.set_xscale("log"); ax.axvline(300, color=MUTED, ls="--", lw=.9)\n'
      '    ax.set_title(f"{t}", fontsize=8, color=TCOLOR[t])\n'
      'fig.suptitle("Rank of actives in the No-FT Boltz-2 ranking (log x; dashed = top-300 train cutoff)", fontsize=12)\n'
      'plt.tight_layout(); plt.show()'),
 md("### Molecular property distributions: actives vs inactives\n"
    "Pooled across all targets (sampled). Sanity check on the libraries and a look at what "
    "distinguishes binders physicochemically."),
 code('dall = pd.concat([desc_frame(t) for t in TARGETS], ignore_index=True)\n'
      'props=["MW","HAC","cLogP","TPSA","HBD","RotB"]\n'
      'fig,axes=plt.subplots(2,3,figsize=(14,7))\n'
      'for ax,p in zip(axes.ravel(), props):\n'
      '    for lab,c,v in [("inactive",INACTIVE,0),("active",ACTIVE,1)]:\n'
      '        ax.hist(dall[dall.active==v][p].dropna(), bins=40, density=True, alpha=.6, color=c, label=lab)\n'
      '    ax.set_title(p)\n'
      'axes[0,0].legend(fontsize=8)\n'
      'fig.suptitle("Property distributions (pooled, sampled): actives vs inactives", fontsize=12)\n'
      'plt.tight_layout(); plt.show()'),
 md("### Do the scoring methods agree? (588689)\n"
    "Spearman correlation among the cached scores shipped with the data (Boltz-2, Boltzina, and, where "
    "available, GNINA/Vina). Low correlation = the methods rank compounds differently."),
 code('raw = pd.read_csv(RAW / "588689.csv")\n'
      'cand = ["score_boltz2","score_boltzina","score_boltzina_recycle1","score_gnina","score_vina"]\n'
      'avail = [c for c in cand if c in raw and raw[c].notna().sum() > 100]\n'
      'corr = raw[avail].corr(method="spearman")\n'
      'fig,ax=plt.subplots(figsize=(5.5,4.5))\n'
      'im=ax.imshow(corr.values, cmap="Blues", vmin=0, vmax=1)\n'
      'ax.set_xticks(range(len(avail))); ax.set_xticklabels(avail, rotation=45, ha="right", fontsize=8)\n'
      'ax.set_yticks(range(len(avail))); ax.set_yticklabels(avail, fontsize=8)\n'
      'for i in range(len(avail)):\n'
      '    for j in range(len(avail)):\n'
      '        ax.text(j,i,f"{corr.values[i,j]:.2f}",ha="center",va="center",fontsize=8,\n'
      '                color=("white" if corr.values[i,j]>0.6 else INK))\n'
      'fig.colorbar(im, fraction=0.046, pad=0.04); ax.set_title("588689: Spearman corr of scoring methods")\n'
      'plt.tight_layout(); plt.show()\n'
      'print("methods with data:", avail)'),
 md("### Scaffold diversity among actives\n"
    "Unique Bemis\\u2013Murcko scaffolds divided by number of actives, per target. Higher = the actives are "
    "more chemically diverse (harder for a similarity/scaffold prior); lower = a few dominant chemotypes."),
 code('rows=[]\n'
      'scaf_by_target={}\n'
      'for t,df in frames.items():\n'
      '    act=df[df.Active_v2==1]\n'
      '    scafs=[]\n'
      '    for s in act["neut-smiles"]:\n'
      '        m=Chem.MolFromSmiles(str(s))\n'
      '        if m: scafs.append(MurckoScaffold.MurckoScaffoldSmiles(mol=m))\n'
      '    scaf_by_target[t]=scafs\n'
      '    u=len(set(scafs))\n'
      '    rows.append(dict(target=t, n_active=len(act), n_scaffold=u, frac=u/max(1,len(act))))\n'
      'sc=pd.DataFrame(rows)\n'
      'fig,ax=plt.subplots(figsize=(9,4))\n'
      'ax.bar(range(8),[sc.frac[i] for i in range(8)],color=[TCOLOR[t] for t in sc.target])\n'
      'ax.set_xticks(range(8)); ax.set_xticklabels(sc.target, rotation=45, ha="right", fontsize=7)\n'
      'ax.set_ylabel("unique scaffolds / #actives"); ax.set_ylim(0,1)\n'
      'ax.set_title("Scaffold diversity among actives (Bemis\\u2013Murcko)")\n'
      'plt.tight_layout(); plt.show()\n'
      'sc'),
 md("### Most common active scaffolds (588689)\n"
    "The dominant chemotypes among confirmed binders for the lead target."),
 code('from collections import Counter\n'
      'cnt=Counter(s for s in scaf_by_target["588689"] if s)\n'
      'common=cnt.most_common(8)\n'
      'mols=[Chem.MolFromSmiles(sm) for sm,_ in common]\n'
      'pairs=[(m,f"n={n}") for (m),( _,n) in zip(mols,common) if m]\n'
      'Draw.MolsToGridImage([m for m,_ in pairs], molsPerRow=4, subImgSize=(220,180),\n'
      '                     legends=[l for _,l in pairs])'),
]
write("00_data_stats.ipynb", stats)

# ---------------------------------------------------------------- 00_data_prep
prep = [
 md("# 00 · Data preparation (pipeline inputs)\n\n"
    "How the cleaned tables become Boltz-2 inputs: the per-target **protein construct**, the "
    "**MSA**, the per-compound **YAML**, the **nested train/eval split**, and the **chunking** used "
    "for SLURM array jobs. This is the input side of the ~50k-compound co-folding step."),
 code(PREAMBLE),
 code(STYLE),
 md("## Protein constructs\n"
    "Construct sequences are **not** distributed with the data; we reconstructed them from the "
    "reference structures named in the Boltzina paper (6 PDBs) and from NCBI for the 2 "
    "predicted-holo targets. Two assignments were cross-checked against PubChem BioAssay targets."),
 code('rows=[]\n'
      'for t,(prot,src) in TARGETS.items():\n'
      '    sp = DATA / f"{t}_seq.txt"\n'
      '    seq = sp.read_text().strip() if sp.exists() else ""\n'
      '    rows.append({"target": t, "protein": prot, "source": src, "construct_len": len(seq)})\n'
      'pd.DataFrame(rows)'),
 md("## MSA depth\nOne MSA per target (colabfold MMseqs2), shared across all of that target's compounds — "
    "exactly the paper's setup."),
 code('rows=[]\n'
      'for t in TARGETS:\n'
      '    mp = DATA / f"{t}_msa.csv"\n'
      '    if mp.exists():\n'
      '        n = sum(1 for _ in open(mp)) - 1\n'
      '        rows.append({"target": t, "protein": TARGETS[t][0], "msa_sequences": n})\n'
      '    else:\n'
      '        rows.append({"target": t, "protein": TARGETS[t][0], "msa_sequences": "(generating)"})\n'
      'pd.DataFrame(rows)'),
 md("## Example Boltz-2 input YAML\nProtein (with shared MSA) + ligand SMILES + affinity binder request."),
 code('import glob\n'
      'y = sorted(glob.glob(str(ROOT/"results/runs/588689/inputs_full/*.yaml")))\n'
      'print(Path(y[0]).name if y else "(inputs not built yet)")\n'
      'if y:\n'
      '    txt = Path(y[0]).read_text()\n'
      '    # trim the long sequence/MSA path for display\n'
      '    print("\\n".join(l[:90] for l in txt.splitlines()))'),
 md("## Nested train / eval split\n"
    "Training sets are the top-N of the Boltz-2 ranking and are **nested** (40 \\u2282 100 \\u2282 300). "
    "The common evaluation set is everything below rank 300. The last 20 of the ranking are held "
    "out for validation during fine-tuning."),
 code('t="588689"\n'
      'df = pd.read_csv(DATA/f"{t}_results.csv").set_index("CID").loc[load_split_order(t)].reset_index()\n'
      'df["rank"]=np.arange(1,len(df)+1)\n'
      'for n in (40,100,300):\n'
      '    tr=df.iloc[:n]\n'
      '    print(f"train top{n:<3d}: {len(tr):3d} compounds, {int(tr.Active_v2.sum())} active")\n'
      'print(f"eval set  : rank 301..{len(df)} = {len(df)-300} compounds, {int(df.iloc[300:].Active_v2.sum())} active")\n'
      'print(f"validation: last 20 of ranking (held out during FT)")'),
 md("## Chunking for SLURM arrays\n"
    "Inputs are split into 350-compound chunks; each chunk is one GPU array task for Pass-1 "
    "co-folding. Pass-1 is the compute bottleneck (~26 s/compound &rarr; ~360 GPU-h/target)."),
 code('for t in TARGETS:\n'
      '    ct = ROOT/f"results/runs/{t}/chunks.tsv"\n'
      '    if ct.exists():\n'
      '        n=sum(1 for _ in open(ct)); print(f"{t:16s} {n} chunks x 350")\n'
      '    else:\n'
      '        print(f"{t:16s} (not chunked yet)")'),
 md("### Pipeline order\n"
    "`select_subsets` &rarr; `build_boltz_inputs` &rarr; `make_chunks` &rarr; **Pass-1** co-folding "
    "(`--write_embeddings_cropped --skip_affinity_prediction`) &rarr; `merge_consolidate` + "
    "`build_ft_inputs_full` &rarr; **Pass-2** affinity cache (reuses poses) &rarr; multi-seed "
    "head-FT train + score &rarr; evaluation."),
 md("## Construct length and MSA depth per target\n"
    "Left: reconstructed protein construct lengths. Right: number of MSA sequences (colabfold), shared "
    "across every compound of that target. Deeper MSA generally = more reliable co-folding."),
 code('lens, depths, labels, colors = [], [], [], []\n'
      'for t in TARGETS:\n'
      '    sp, mp = DATA/f"{t}_seq.txt", DATA/f"{t}_msa.csv"\n'
      '    lens.append(len(sp.read_text().strip()) if sp.exists() else 0)\n'
      '    depths.append((sum(1 for _ in open(mp))-1) if mp.exists() else 0)\n'
      '    labels.append(t); colors.append(TCOLOR[t])\n'
      'fig,(a1,a2)=plt.subplots(1,2,figsize=(13,4.2))\n'
      'a1.barh(range(8), lens, color=colors); a1.set_yticks(range(8)); a1.set_yticklabels(labels, fontsize=7)\n'
      'a1.invert_yaxis(); a1.set_xlabel("construct length (residues)"); a1.set_title("Protein construct length")\n'
      'a2.barh(range(8), depths, color=colors); a2.set_yticks(range(8)); a2.set_yticklabels(labels, fontsize=7)\n'
      'a2.invert_yaxis(); a2.set_xlabel("# MSA sequences"); a2.set_title("MSA depth")\n'
      'for i,v in enumerate(depths): a2.text(v, i, f" {v}", va="center", fontsize=7)\n'
      'plt.tight_layout(); plt.show()'),
]
write("00_data_prep.ipynb", prep)
print("done")

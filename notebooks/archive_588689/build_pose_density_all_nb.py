#!/usr/bin/env python3
"""13_pose_density_all.ipynb - per-residue ligand density on the ACTUAL folded protein, self-filling
across all targets. For each target with results, shows (1) the weight (contact-frequency)
distribution per residue along the full-length sequence, and (2) the real CA backbone coloured by
that density - the density painted on the actual structure, not an arbitrary aligned frame. Reads
results/runs/<T>/pose_density/{residue_density.csv, ca_colored.npz}. Matplotlib only, NO HTML."""
import nbformat as nbf
from pathlib import Path
NB = Path("/global/scratch/users/sergiomar10/boltzaff/notebooks/13_pose_density_all.ipynb")
def md(s): return nbf.v4.new_markdown_cell(s)
def code(s): return nbf.v4.new_code_cell(s)

PRE = r'''
from pathlib import Path
import numpy as np, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from IPython.display import display
BLK="#000000"
mpl.rcParams.update({"figure.facecolor":"white","axes.facecolor":"white","savefig.facecolor":"white",
  "text.color":BLK,"axes.labelcolor":BLK,"axes.titlecolor":BLK,"xtick.color":BLK,"ytick.color":BLK,
  "axes.edgecolor":BLK,"font.size":10,"font.weight":"normal","axes.titleweight":"normal","axes.labelweight":"normal",
  "axes.grid":True,"grid.color":"#e6e6e6","axes.axisbelow":True})
ROOT=Path("/global/scratch/users/sergiomar10/boltzaff")
SHORT={"588689":"588689 (NS5 MTase)","504329":"504329","1053173-743445":"743445","493248-485317":"485317",
       "434954-2097":"2097","540297-493091":"493091","463203-2650":"2650","624273-588549":"588549"}
ALL=list(SHORT)
def load(t):
    p=ROOT/"results/runs"/t/"pose_density"/"residue_density.csv"; z=ROOT/"results/runs"/t/"pose_density"/"ca_colored.npz"
    if not p.exists() or not z.exists(): return None
    return pd.read_csv(p), np.load(z, allow_pickle=True)
DATA={t:load(t) for t in ALL}; DONE=[t for t in ALL if DATA[t] is not None]
print("targets with per-residue density:", [SHORT[t] for t in DONE])
print("pending:", [SHORT[t] for t in ALL if t not in DONE])
'''

cells=[
 md("# 13 · Per-residue binding density on the real protein (all targets)\n\n"
    "Where do the ligands actually sit on each target? For every folded complex we take that complex's "
    "own coordinates and, for each **chain-A residue**, its minimum distance to the ligand; a residue is "
    "in contact when the ligand comes within 5 A. Aggregating the **contact frequency per residue index** "
    "across all poses needs no alignment (every distance is measured inside one complex), so the profile "
    "sits on the **actual residues of the actual fold** - not a Kabsch-aligned reference frame.\n\n"
    "For each target this notebook shows: (1) the **weight distribution per residue** along the full "
    "sequence (which residues the library binds, and how focused), and (2) that density **painted on the "
    "real CA backbone** in 3D. It self-fills as each target's structures finish folding."),
 code(PRE),

 md("## Per-residue density profile + structure, per target\n"
    "Left: contact frequency vs residue number (the binding-site fingerprint along the whole protein; a "
    "sharp peak = one focused pocket, multiple peaks = multiple sites). Right: the real CA backbone "
    "coloured by the same density (hot = high ligand occupancy). Top contacted residues are listed."),
 code('from mpl_toolkits.mplot3d.art3d import Line3DCollection\n'
      'from scipy.interpolate import splprep, splev\n'
      'def ribbon(ax,ca,dens):\n'
      '    # smooth B-spline through the CA trace -> a tube coloured by per-residue density (a backbone cartoon)\n'
      '    n=len(ca); k=min(3,n-1)\n'
      '    try:\n'
      '        tck,u=splprep([ca[:,0],ca[:,1],ca[:,2]],s=n*3.0,k=k); uu=np.linspace(0,1,n*14)\n'
      '        sx,sy,sz=splev(uu,tck); pts=np.array([sx,sy,sz]).T\n'
      '        di=np.interp(uu,np.linspace(0,1,n),dens)\n'
      '    except Exception:\n'
      '        pts=ca; di=dens\n'
      '    segs=np.stack([pts[:-1],pts[1:]],axis=1)\n'
      '    lc=Line3DCollection(segs,cmap="inferno",linewidths=9); lc.set_array(di[:-1]); lc.set_capstyle("round")\n'
      '    ax.add_collection3d(lc)\n'
      '    mn=pts.min(0); mx=pts.max(0); c=(mn+mx)/2; rad=(mx-mn).max()/2*1.05\n'
      '    ax.set_xlim(c[0]-rad,c[0]+rad); ax.set_ylim(c[1]-rad,c[1]+rad); ax.set_zlim(c[2]-rad,c[2]+rad)\n'
      '    try: ax.set_box_aspect((1,1,1))\n'
      '    except Exception: pass\n'
      '    return lc\n'
      'for t in DONE:\n'
      '    df,z=DATA[t]; ca=z["ca"]; dens=z["contact_frac"]\n'
      '    fig=plt.figure(figsize=(15,4.8))\n'
      '    # (1) per-residue profile along sequence\n'
      '    ax1=fig.add_subplot(1,2,1)\n'
      '    ax1.bar(df.res_num,df.contact_frac,width=1.0,color="#2a78d6")\n'
      '    ax1.set_xlabel("residue number (full-length protein)"); ax1.set_ylabel("contact frequency (fraction of poses <=5A)")\n'
      '    ax1.set_title(f"{SHORT[t]} - weight per residue site"); ax1.spines[["top","right"]].set_visible(False)\n'
      '    thr=df.contact_frac.quantile(0.97)\n'
      '    for _,r in df[df.contact_frac>=thr].iterrows():\n'
      '        ax1.annotate(f"{r.res_name}{int(r.res_num)}",(r.res_num,r.contact_frac),textcoords="offset points",xytext=(0,3),ha="center",fontsize=7)\n'
      '    # (2) the real protein backbone as a ribbon/tube, coloured by density\n'
      '    ax2=fig.add_subplot(1,2,2,projection="3d")\n'
      '    lc=ribbon(ax2,ca,dens)\n'
      '    ax2.set_title(f"{SHORT[t]} - binding density on the protein fold"); ax2.set_xticks([]); ax2.set_yticks([]); ax2.set_zticks([])\n'
      '    ax2.grid(False); ax2.view_init(elev=18,azim=35)\n'
      '    fig.colorbar(lc,ax=ax2,shrink=0.6,label="contact frequency")\n'
      '    plt.tight_layout(); plt.show()\n'
      '    top=df.sort_values("contact_frac",ascending=False).head(8)\n'
      '    print(f"{SHORT[t]}: top binding residues:", ", ".join(f"{r.res_name}{int(r.res_num)}({r.contact_frac:.2f})" for _,r in top.iterrows()))'),

 md("## How focused is the binding across targets?\n"
    "The share of total contact weight held by each target's top-10 residues. High = one tight pocket "
    "dominates; low = the ligands spread over several sites or a shallow surface."),
 code('rows=[]\n'
      'for t in DONE:\n'
      '    df,_=DATA[t]; tot=df.contact_frac.sum()\n'
      '    top10=df.contact_frac.nlargest(10).sum()\n'
      '    n_hot=(df.contact_frac>=0.5).sum()\n'
      '    pk=df.loc[df.contact_frac.idxmax()]; peak=f"{pk.res_name}{int(pk.res_num)}"\n'
      '    rows.append(dict(target=SHORT[t], residues=len(df), top10_share=round(top10/tot,3) if tot>0 else 0,\n'
      '                     residues_over_50pct=int(n_hot), peak_residue=peak))\n'
      'if rows:\n'
      '    s=pd.DataFrame(rows).set_index("target"); display(s)\n'
      '    fig,ax=plt.subplots(figsize=(8,4.4)); ax.bar(range(len(s)),s.top10_share,color="#1baf7a")\n'
      '    ax.set_xticks(range(len(s))); ax.set_xticklabels(s.index,rotation=40,ha="right"); ax.set_ylabel("top-10-residue share of contact weight")\n'
      '    ax.set_title("binding focus per target"); ax.spines[["top","right"]].set_visible(False); plt.tight_layout(); plt.show()\n'
      'else: print("no targets done yet")'),

 md("## Notes\n\n"
    "- **On the real structure, not a reference frame.** Contact frequency is computed per residue within "
    "each complex's own coordinates, so it is anchored to the actual residues/fold; the 3D panel paints "
    "that density onto real CA positions. A density-coloured PDB (B-factor = contact frequency) is also "
    "saved per target under `results/runs/<T>/pose_density/` for opening in PyMOL/ChimeraX.\n"
    "- **Self-filling.** Targets appear as their Pass-1 structures finish; 588689 is complete, others fill "
    "in. Numbers use a sample of poses per target (stable for frequencies).\n"
    "- **Reading it.** A single dominant peak means the library funnels into one pocket; secondary peaks "
    "flag alternative/surface sites - relevant to whether a confidence or pose filter would help "
    "(cross-reference the uncertainty notebooks)."),
]
nb=nbf.v4.new_notebook(); nb.cells=cells
nb.metadata={"kernelspec":{"name":"boltzba","display_name":"Python (boltzba)"},"language_info":{"name":"python"}}
NB.write_text(nbf.writes(nb)); print("wrote", NB)

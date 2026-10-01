#!/usr/bin/env python3
"""Assemble the 3D binding-density viewer: reference-frame protein + ligand-centroid cloud
(colored by activity / scaffold novelty), consensus site + SAH pocket markers. Emits HTML."""
import json, glob, gzip
from pathlib import Path
import numpy as np, pandas as pd
import biotite.structure as struc, biotite.structure.io.pdbx as pdbx, biotite.structure.io.pdb as pdb
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

ROOT = Path("/global/scratch/users/sergiomar10/boltzaff"); R = ROOT/"results/runs/588689"; PD = R/"pose_density"
OUT = ROOT/"notebooks/ns5_density_viewer.html"
pocket = np.array(json.loads((PD/"pocket.json").read_text())["sah_centroid"])
ref_pose = json.loads((PD/"pocket.json").read_text())["ref_pose"]

# reference-frame protein (chain A) as PDB
refc = glob.glob(str(R/f"outputs_chunks_full/chunk_*/boltz_results_*/predictions/{ref_pose}/{ref_pose}_model_0.cif"))
arr = pdbx.get_structure(pdbx.CIFFile.read(refc[0]), model=1)
prot = arr[struc.filter_amino_acids(arr)]; prot = prot[prot.chain_id == prot.chain_id[0]]
import io
pf = pdb.PDBFile(); pf.set_structure(prot)
sbuf = io.StringIO(); pf.write(sbuf); protein_pdb = sbuf.getvalue()

# merged centroids + labels/scaffold/rank
cs = []
for f in glob.glob(str(PD/"centroids_chunk_*.csv")):
    try:
        t = pd.read_csv(f)
        if len(t): cs.append(t)
    except Exception: pass
cen = pd.concat(cs, ignore_index=True).drop_duplicates("CID")
df = pd.read_csv(ROOT/"data/588689_results.csv"); df["CID"] = df.CID.astype(int)
ids = [int(x.strip().removeprefix("588689_")) for x in gzip.open(ROOT/"BoltzFT/data/splits/588689.txt.gz","rt")]
df = df.set_index("CID").loc[ids].reset_index(); df["rank"] = np.arange(1, len(df)+1)
df["scaf"] = [MurckoScaffold.MurckoScaffoldSmiles(mol=m) if (m:=Chem.MolFromSmiles(str(s))) else None for s in df["neut-smiles"]]
train300 = set(s for s in df.scaf[:300] if s); df["seen"] = df.scaf.isin(train300)
d = cen.merge(df[["CID","Active_v2","rank","seen"]], on="CID")
d = d[d.ca_rmsd <= np.percentile(d.ca_rmsd, 95)].copy()
from sklearn.cluster import KMeans
km = KMeans(2, n_init=10, random_state=0).fit(d[["x","y","z"]].values)
d["k"] = km.labels_
# label the cluster nearest SAH as "site B (SAH)", the other as "site A (major)"
dists = [np.linalg.norm(c - pocket) for c in km.cluster_centers_]
sah_k = int(np.argmin(dists))
consensus = km.cluster_centers_[1 - sah_k]  # major (non-SAH) site center

def pts_pdb(coords, elem="C", resn="DEN"):
    lines = []
    for i, (x, y, z) in enumerate(coords, 1):
        lines.append(f"HETATM{i%99999:5d}  {elem:<2s}  {resn} A{i%9999:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {elem:>2s}")
    return "\n".join(lines) + "\n"

def subsample(a, n=4000):
    return a if len(a) <= n else a[np.random.default_rng(0).choice(len(a), n, replace=False)]
inA = subsample(d[(d.Active_v2==0) & (d.k != sah_k)][["x","y","z"]].values)   # major site cloud
inB = subsample(d[(d.Active_v2==0) & (d.k == sah_k)][["x","y","z"]].values)   # SAH site cloud
ev = d[(d["rank"] > 300) & (d.Active_v2 == 1)]
seen = ev[ev.seen][["x","y","z"]].values
uns = ev[~ev.seen][["x","y","z"]].values
allact = d[d.Active_v2 == 1][["x","y","z"]].values
layers = dict(siteA=pts_pdb(inA), siteB=pts_pdb(inB), active_all=pts_pdb(allact),
              seen=pts_pdb(seen), unseen=pts_pdb(uns))
def stats_k(mask):
    g = d[mask]; return dict(n=int(len(g)), pct=round(len(g)/len(d)*100,1),
        n_active=int(g.Active_v2.sum()), active_rate=round(g.Active_v2.mean()*100,2))
meta = dict(n_total=int(len(d)), n_active=int((d.Active_v2==1).sum()),
            n_seen=int(len(seen)), n_unseen=int(len(uns)),
            siteA=stats_k(d.k != sah_k), siteB=stats_k(d.k == sah_k),
            centerA=[float(x) for x in km.cluster_centers_[1-sah_k]],
            centerB=[float(x) for x in km.cluster_centers_[sah_k]],
            pocket=[float(x) for x in pocket],
            sep=float(np.linalg.norm(km.cluster_centers_[0]-km.cluster_centers_[1])),
            distB_sah=float(np.linalg.norm(km.cluster_centers_[sah_k]-pocket)),
            seen_pctA=round((seen.__len__() and (d[(d["rank"]>300)&(d.Active_v2==1)&(d.seen)].k!=sah_k).mean()*100) or 0,0),
            unseen_pctA=round((d[(d["rank"]>300)&(d.Active_v2==1)&(~d.seen)].k!=sah_k).mean()*100,0))
payload = json.dumps(dict(protein=protein_pdb, layers=layers, meta=meta))
Path(ROOT/"notebooks").mkdir(exist_ok=True)
(ROOT/"notebooks/_density_payload.json").write_text(payload)
print("layers:", {k: v.count("HETATM") for k, v in layers.items()}, "meta:", meta)
print("payload bytes:", len(payload))

#!/usr/bin/env python3
"""Prepare aligned pose data for the interactive 3D viewer.

For each of the top-N predicted complexes: superpose the predicted protein onto the
3EVG crystal structure (sequence-aware, biotite.superimpose_homologs), apply the same
transform to protein+ligand, and write a compact PDB in the crystal frame. Also writes
the 3EVG reference PDB and a meta.json (rank, CID, active label, Boltz-2 score, confidence).
"""
import json, re, gzip
from pathlib import Path
import numpy as np
import pandas as pd
import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
import biotite.structure.io.pdb as pdb

ROOT = Path("/global/scratch/users/sergiomar10/boltzaff")
R = ROOT / "results/runs/588689"
OUT = R / "top_poses/viewer_data"; OUT.mkdir(parents=True, exist_ok=True)
POSE_DIR = R / "top_poses/out/boltz_results_inputs/predictions"

def load_cif(path):
    f = pdbx.CIFFile.read(str(path))
    return pdbx.get_structure(f, model=1)

# reference crystal structure: first protein chain
ref_all = load_cif(ROOT / "data/3EVG.cif")
ref_prot = ref_all[struc.filter_amino_acids(ref_all)]
ref_chain = ref_prot[ref_prot.chain_id == ref_prot.chain_id[0]]
pdb.PDBFile(); rf = pdb.PDBFile(); rf.set_structure(ref_chain); rf.write(str(OUT / "ref_3EVG.pdb"))

res = pd.read_csv(ROOT / "data/588689_results.csv").set_index("CID")

meta = []
for d in sorted(POSE_DIR.glob("rank*_588689_*")):
    m = re.match(r"rank(\d+)_588689_(\d+)", d.name)
    rank, cid = int(m.group(1)), int(m.group(2))
    cif = d / f"{d.name}_model_0.cif"
    conf = json.loads((d / f"confidence_{d.name}_model_0.json").read_text())
    arr = load_cif(cif)
    prot = arr[struc.filter_amino_acids(arr)]
    prot = prot[prot.chain_id == prot.chain_id[0]]
    lig = arr[arr.chain_id == "B"]                        # ligand chain
    # sequence-aware superposition of predicted protein onto crystal
    try:
        fitted_prot, transform, _, _ = struc.superimpose_homologs(ref_chain, prot)
        full = prot + lig
        full = transform.apply(full)
    except Exception as e:
        print(f"rank{rank}: superpose failed ({e}); using raw frame")
        full = prot + lig
    # PDB res names are <=3 chars; boltz names the ligand "LIG1"
    full.res_name = np.array(["LIG" if str(n).startswith("LIG") else n for n in full.res_name])
    pf = pdb.PDBFile(); pf.set_structure(full); pf.write(str(OUT / f"pose_{rank:02d}.pdb"))
    row = res.loc[cid] if cid in res.index else None
    meta.append(dict(rank=rank, cid=cid,
        active=int(row["Active_v2"]) if row is not None else -1,
        score=float(row["affinity_probability_binary"]) if row is not None else None,
        plddt=round(float(conf.get("complex_plddt", 0)), 3),
        ligand_iptm=round(float(conf.get("ligand_iptm", 0)), 3),
        confidence=round(float(conf.get("confidence_score", 0)), 3)))

meta.sort(key=lambda x: x["rank"])
(OUT / "meta.json").write_text(json.dumps(meta, indent=1))
print(f"wrote {len(meta)} aligned poses + ref_3EVG.pdb + meta.json -> {OUT}")
print("actives in top-24:", sum(m["active"] == 1 for m in meta))

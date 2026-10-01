#!/usr/bin/env python3
"""Build the common reference frame for pose-density analysis (target 588689).

- ref_ca.npy : CA coordinates (N x 3, residue order) of one folded pose = the frame every
  other pose is Kabsch-aligned to (all poses share the identical 275-aa construct).
- pocket.json: the true catalytic-site marker = centroid of 3EVG's SAH ligand, brought into
  the reference-pose frame by sequence-aware superposition (biotite.superimpose_homologs).
"""
import json, glob
from pathlib import Path
import numpy as np
import gemmi
import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx

ROOT = Path("/global/scratch/users/sergiomar10/boltzaff")
R = ROOT / "results/runs/588689"
OUT = R / "pose_density"; OUT.mkdir(parents=True, exist_ok=True)

# pick a reference pose (first available prediction CIF)
refc = None
for ch in ["chunk_000", "chunk_001", "chunk_010"]:
    hits = sorted(glob.glob(str(R / f"outputs_chunks_full/{ch}/boltz_results_*/predictions/*/*_model_0.cif")))
    if hits:
        refc = hits[0]; break
if refc is None:
    raise SystemExit("no pose CIFs yet")
st = gemmi.read_structure(refc); m = st[0]
ca = [[a.pos.x, a.pos.y, a.pos.z] for ch in m if ch.name == "A" for res in ch for a in res if a.name == "CA"]
ref_ca = np.array(ca, float)
np.save(OUT / "ref_ca.npy", ref_ca)
print(f"reference pose: {Path(refc).parent.name}  CA={len(ref_ca)}")

# true pocket: SAH centroid from crystal 3EVG, mapped into the reference-pose frame
try:
    ref_all = pdbx.get_structure(pdbx.CIFFile.read(refc), model=1)
    ref_prot = ref_all[struc.filter_amino_acids(ref_all)]
    ref_prot = ref_prot[ref_prot.chain_id == ref_prot.chain_id[0]]
    evg = pdbx.get_structure(pdbx.CIFFile.read(str(ROOT / "data/3EVG.cif")), model=1)
    evg_prot = evg[struc.filter_amino_acids(evg)]
    evg_prot = evg_prot[evg_prot.chain_id == evg_prot.chain_id[0]]
    _, transform, _, _ = struc.superimpose_homologs(ref_prot, evg_prot)   # 3EVG -> ref-pose frame
    sah = evg[(evg.res_name == "SAH")]
    sah_ref = transform.apply(sah)
    pocket = dict(sah_centroid=[float(x) for x in sah_ref.coord.mean(0)],
                  ref_pose=Path(refc).parent.name)
except Exception as e:
    print("pocket mapping failed:", e)
    pocket = dict(sah_centroid=[float(x) for x in ref_ca.mean(0)], ref_pose=Path(refc).parent.name, note="fallback=CA centroid")
(OUT / "pocket.json").write_text(json.dumps(pocket))
print("true-pocket (SAH) centroid in ref frame:", np.round(pocket["sah_centroid"], 2))

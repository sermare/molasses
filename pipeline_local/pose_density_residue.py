#!/usr/bin/env python3
"""Per-residue ligand-density on the ACTUAL folded protein (no reference-frame alignment).
For each folded complex (boltz CIF), compute every chain-A residue's minimum distance to any
chain-B ligand heavy atom, and count a 'contact' when it is within CUTOFF. Aggregating contact
frequency per residue INDEX across all poses is frame-independent (all distances are within one
complex), so the resulting profile is grounded on the real residues of the real structure - a
weight distribution per residue site along the full-length protein. Also saves one representative
structure's real CA coordinates (+ per-residue density) so the notebook can colour the actual
backbone. Self-contained per target.

Usage: pose_density_residue.py --target <T> [--sample 3000] [--cutoff 5.0]
Outputs (under results/runs/<T>/pose_density/): residue_density.csv, ca_colored.npz
"""
import argparse, glob, os
from pathlib import Path
import numpy as np, pandas as pd
import gemmi

def residue_atoms_and_ligand(cif):
    m = gemmi.read_structure(cif)[0]
    res_names, res_nums, res_atom_xyz = [], [], []
    ca_xyz = []
    lig = []
    for ch in m:
        if ch.name == "A":
            for res in ch:
                xs = [[a.pos.x, a.pos.y, a.pos.z] for a in res if a.element.name != "H"]
                if not xs: continue
                res_names.append(res.name); res_nums.append(res.seqid.num)
                res_atom_xyz.append(np.array(xs, float))
                ca = [a for a in res if a.name == "CA"]
                ca_xyz.append([ca[0].pos.x, ca[0].pos.y, ca[0].pos.z] if ca else np.mean(xs, 0).tolist())
        elif ch.name == "B":
            lig += [[a.pos.x, a.pos.y, a.pos.z] for res in ch for a in res if a.element.name != "H"]
    return res_names, res_nums, res_atom_xyz, np.array(ca_xyz, float), np.array(lig, float)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--root", default="/global/scratch/users/sergiomar10/boltzaff")
    ap.add_argument("--sample", type=int, default=3000)
    ap.add_argument("--cutoff", type=float, default=5.0)
    a = ap.parse_args()
    R = Path(a.root) / "results/runs" / a.target
    cifs = glob.glob(str(R / "outputs_chunks_full/*/*/predictions/*/*_model_0.cif"))
    if not cifs:
        print(f"[{a.target}] no CIFs yet"); return
    rng = np.random.default_rng(0)
    if len(cifs) > a.sample:
        cifs = list(rng.choice(cifs, a.sample, replace=False))
    n_res = None; contacts = None; sumdist = None; near = None; n_ok = 0
    ref_names = ref_nums = ref_ca = ref_cif = None
    for cif in cifs:
        try:
            names, nums, ratoms, ca, lig = residue_atoms_and_ligand(cif)
        except Exception:
            continue
        if len(lig) == 0 or len(ratoms) == 0: continue
        if n_res is None:
            n_res = len(ratoms); contacts = np.zeros(n_res); sumdist = np.zeros(n_res); near = np.zeros(n_res)
            ref_names, ref_nums, ref_ca, ref_cif = names, nums, ca, cif
        if len(ratoms) != n_res:   # skip complexes with a different residue count (rare)
            continue
        mind = np.array([np.sqrt(((atoms[:, None, :] - lig[None, :, :]) ** 2).sum(-1)).min() for atoms in ratoms])
        contacts += (mind <= a.cutoff).astype(float)
        near += np.exp(-mind / a.cutoff)          # soft density (closer = more weight)
        sumdist += mind
        n_ok += 1
    if n_ok == 0:
        print(f"[{a.target}] no usable CIFs"); return
    df = pd.DataFrame({
        "res_index": np.arange(n_res),
        "res_name": ref_names[:n_res],
        "res_num": ref_nums[:n_res],
        "contact_frac": contacts / n_ok,             # fraction of poses within cutoff = the weight
        "soft_density": near / n_ok,
        "mean_min_dist": sumdist / n_ok,
    })
    out = R / "pose_density"; out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "residue_density.csv", index=False)
    np.savez(out / "ca_colored.npz", ca=ref_ca[:n_res], contact_frac=(contacts / n_ok),
             soft_density=(near / n_ok), res_name=np.array(ref_names[:n_res]), res_num=np.array(ref_nums[:n_res]))
    # density-coloured PDB: representative structure with B-factor = per-residue contact frequency
    try:
        cf = contacts / n_ok
        st = gemmi.read_structure(ref_cif); m = st[0]
        for ch in m:
            if ch.name != "A": continue
            for i, res in enumerate(ch):
                b = float(cf[i]) if i < n_res else 0.0
                for at in res: at.b_iso = b * 100.0   # 0..100 for easy PyMOL spectrum b
        st.setup_entities()
        (out / "density_colored.pdb").write_text(st.make_pdb_string())
    except Exception as e:
        print("  (pdb export skipped:", e, ")")
    top = df.sort_values("contact_frac", ascending=False).head(8)
    print(f"[{a.target}] poses used={n_ok}, residues={n_res}. top contact residues:")
    print(top[["res_index", "res_name", "res_num", "contact_frac"]].to_string(index=False))

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Per-chunk: Kabsch-align each pose's protein CA to the reference frame, apply the same
transform to the ligand, and record the ligand centroid (+ CA-fit RMSD, radius of gyration).
Output one CSV per chunk; the notebook aggregates and clusters."""
import argparse, glob
from pathlib import Path
import numpy as np
import gemmi

def kabsch(P, Q):
    Pc = P - P.mean(0); Qc = Q - Q.mean(0)
    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    t = Q.mean(0) - R @ P.mean(0)
    return R, t

def coords(cif):
    m = gemmi.read_structure(cif)[0]
    ca, lig = [], []
    for ch in m:
        if ch.name == "A":
            ca += [[a.pos.x, a.pos.y, a.pos.z] for res in ch for a in res if a.name == "CA"]
        elif ch.name == "B":
            lig += [[a.pos.x, a.pos.y, a.pos.z] for res in ch for a in res if a.element.name != "H"]
    return np.array(ca, float), np.array(lig, float)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True, help="a chunk's .../predictions dir")
    ap.add_argument("--ref-ca", required=True)
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()
    ref = np.load(args.ref_ca)
    rows = []
    for cif in glob.glob(f"{args.pred_dir}/*/*_model_0.cif"):
        name = Path(cif).parent.name           # 588689_<CID>
        try:
            cid = int(name.split("_")[-1])
            P, L = coords(cif)
            if len(P) != len(ref) or len(L) == 0:
                continue
            Rm, t = kabsch(P, ref)
            La = (Rm @ L.T).T + t
            Pa = (Rm @ P.T).T + t
            cen = La.mean(0)
            rg = float(np.sqrt(((La - cen) ** 2).sum(1).mean()))
            ca_rmsd = float(np.sqrt(((Pa - ref) ** 2).sum(1).mean()))
            rows.append(dict(CID=cid, x=cen[0], y=cen[1], z=cen[2],
                             lig_rg=rg, ca_rmsd=ca_rmsd, n_lig=len(L)))
        except Exception:
            continue
    import pandas as pd
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    print(f"{args.pred_dir}: {len(rows)} centroids -> {args.out_csv}")

if __name__ == "__main__":
    main()

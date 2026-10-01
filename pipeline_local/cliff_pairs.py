#!/usr/bin/env python3
"""Structural-cliff mining for one target. Finds pairs of near-identical structures (ECFP4
Tanimoto >= --sim) where one is active and the partner is (a) inactive = CLIFF pair (binding lost)
or (b) also active = CONSERVED control pair, then measures for every pair member:
  * every score modality (9 dataset scores + our No-FT pipeline score + 5-seed head-FT score),
    as an orientation-corrected percentile rank across the eval set (higher = predicted better binder)
  * Boltz-2 confidence (from the pose's confidence JSON) and two-head disagreement
  * per-residue minimum distance to the ligand from the pose's own CIF (frame-independent)
and for every pair the MCS-based structural difference (which atoms differ, what they are).
Only eval-set compounds (the 300 training compounds excluded) are used, so head-FT scores are clean.

Outputs -> results/analysis/<T>/cliffs/{pairs.csv, compounds.csv, contacts.npz}
"""
import argparse, glob, json, os
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFMCS, rdFingerprintGenerator, Descriptors, Crippen, rdMolDescriptors
from sklearn.metrics import roc_auc_score
RDLogger.DisableLog("rdApp.*")

ROOT = Path("/global/scratch/users/sergiomar10/boltzaff")
SCORE_COLS = ["score_boltz2", "score_boltzina", "score_boltzina_recycle1", "score_boltzina_mask",
              "docking_score_boltzina", "docking_score_boltzina_recycle1", "docking_score_boltzina_mask",
              "score_gnina", "score_vina"]

# ---------------------------------------------------------------- structural diff (per pair)
def frag_props(mol, atoms):
    atoms = list(atoms)
    if not atoms:
        return dict(n=0, smi="", has_N=0, has_O=0, has_S=0, has_hal=0, arom=0, charge=0, hbd=0)
    try:
        smi = Chem.MolFragmentToSmiles(mol, atomsToUse=atoms, kekuleSmiles=False)
    except Exception:
        smi = ""
    syms = [mol.GetAtomWithIdx(i).GetSymbol() for i in atoms]
    return dict(n=len(atoms), smi=smi,
                has_N=int("N" in syms), has_O=int("O" in syms), has_S=int("S" in syms),
                has_hal=int(any(s in ("F", "Cl", "Br", "I") for s in syms)),
                arom=int(any(mol.GetAtomWithIdx(i).GetIsAromatic() for i in atoms)),
                charge=int(sum(mol.GetAtomWithIdx(i).GetFormalCharge() for i in atoms)),
                hbd=int(sum(1 for i in atoms if mol.GetAtomWithIdx(i).GetSymbol() in ("N", "O")
                            and mol.GetAtomWithIdx(i).GetTotalNumHs() > 0)))

def mol_props(m):
    return dict(MW=Descriptors.MolWt(m), heavy=m.GetNumHeavyAtoms(), logP=Crippen.MolLogP(m),
                TPSA=rdMolDescriptors.CalcTPSA(m), HBD=rdMolDescriptors.CalcNumHBD(m),
                HBA=rdMolDescriptors.CalcNumHBA(m), rotb=rdMolDescriptors.CalcNumRotatableBonds(m),
                arom_rings=rdMolDescriptors.CalcNumAromaticRings(m), fsp3=rdMolDescriptors.CalcFractionCSP3(m),
                charge=sum(a.GetFormalCharge() for a in m.GetAtoms()))

def pair_diff(args):
    ia, ib, sa, sb = args
    ma, mb = Chem.MolFromSmiles(sa), Chem.MolFromSmiles(sb)
    out = dict(a=ia, b=ib, mcs_ok=0)
    try:
        r = rdFMCS.FindMCS([ma, mb], timeout=3, ringMatchesRingOnly=True, completeRingsOnly=True)
        q = Chem.MolFromSmarts(r.smartsString)
        ta, tb = ma.GetSubstructMatch(q), mb.GetSubstructMatch(q)
        if not ta or not tb:
            return out
        da = [i for i in range(ma.GetNumAtoms()) if i not in set(ta)]
        db = [i for i in range(mb.GetNumAtoms()) if i not in set(tb)]
        fa, fb = frag_props(ma, da), frag_props(mb, db)
        out.update(mcs_ok=1, n_mcs=r.numAtoms, frac_core=r.numAtoms / min(ma.GetNumAtoms(), mb.GetNumAtoms()),
                   diff_a_atoms=json.dumps(da), diff_b_atoms=json.dumps(db), mcs_a=json.dumps(list(ta)), mcs_b=json.dumps(list(tb)))
        for k, v in fa.items(): out[f"a_diff_{k}"] = v
        for k, v in fb.items(): out[f"b_diff_{k}"] = v
        pa, pb = mol_props(ma), mol_props(mb)
        for k in pa: out[f"a_{k}"] = pa[k]; out[f"b_{k}"] = pb[k]
    except Exception:
        pass
    return out

# ---------------------------------------------------------------- per-compound pose + confidence
def pose_record(args):
    rid, cif, conf = args
    import gemmi
    rec = dict(id=rid)
    try:
        d = json.load(open(conf))
        for k in ("confidence_score", "ptm", "iptm", "ligand_iptm", "complex_plddt", "complex_iplddt", "complex_pde", "complex_ipde"):
            rec[k] = d.get(k)
    except Exception:
        pass
    vec = None
    try:
        m = gemmi.read_structure(cif)[0]
        lig = np.array([[a.pos.x, a.pos.y, a.pos.z] for ch in m if ch.name == "B" for res in ch for a in res if a.element.name != "H"])
        prot = [np.array([[a.pos.x, a.pos.y, a.pos.z] for a in res if a.element.name != "H"]) for ch in m if ch.name == "A" for res in ch]
        vec = np.array([np.sqrt(((p[:, None, :] - lig[None, :, :]) ** 2).sum(-1)).min() for p in prot], dtype=np.float32)
        rec["lig_n"] = len(lig)
        rec["lig_rg"] = float(np.sqrt(((lig - lig.mean(0)) ** 2).sum(1).mean()))
    except Exception:
        pass
    return rec, vec

def load_arm(d):
    fs = sorted(glob.glob(str(d) + "/chunk_*.csv"))
    f = pd.concat([pd.read_csv(p) for p in fs], ignore_index=True)
    f["sample_id"] = f.sample_id.astype(str)
    return f.drop_duplicates("sample_id").set_index("sample_id")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="588689")
    ap.add_argument("--sim", type=float, default=0.6)
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    T = a.target; R = ROOT / "results/runs" / T
    out = ROOT / "results/analysis" / T / "cliffs"; out.mkdir(parents=True, exist_ok=True)

    d = pd.read_csv(ROOT / f"data/mf-pcba_test/{T}.csv"); d["id"] = f"{T}_" + d.CID.astype(str)
    ev = set(l.strip() for l in open(R / "ft_inputs_full/eval_ids.txt") if l.strip())
    d = d[d.id.isin(ev)].reset_index(drop=True)
    d["label"] = d.target_active_v2.astype(int)
    print(f"[{T}] eval compounds {len(d)}, actives {int(d.label.sum())}", flush=True)

    # ---- pair mining (actives vs all neighbours) ----
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    mols = [Chem.MolFromSmiles(s) for s in d["neut-smiles"]]
    fps = [gen.GetFingerprint(m) if m is not None else None for m in mols]
    dummy = next(f for f in fps if f is not None)
    allf = [f if f is not None else dummy for f in fps]
    rows = []
    for i in np.where((d.label == 1) & np.array([f is not None for f in fps]))[0]:
        sims = np.array(DataStructs.BulkTanimotoSimilarity(fps[i], allf)); sims[i] = 0
        for j in np.where(sims >= a.sim)[0]:
            if fps[j] is None: continue
            if d.label[j] == 1 and j < i: continue         # dedupe active-active
            rows.append((int(i), int(j), float(sims[j]), int(d.label[j])))
    P = pd.DataFrame(rows, columns=["ia", "ib", "sim", "b_active"])
    P["kind"] = np.where(P.b_active == 0, "cliff", "conserved")
    print(f"pairs: cliff {int((P.kind=='cliff').sum())}, conserved {int((P.kind=='conserved').sum())}", flush=True)

    # ---- structural diff (parallel) ----
    with Pool(a.procs) as pool:
        diffs = pool.map(pair_diff, [(int(r.ia), int(r.ib), d["neut-smiles"][r.ia], d["neut-smiles"][r.ib]) for r in P.itertuples()], chunksize=8)
    D = pd.DataFrame(diffs).rename(columns={"a": "ia", "b": "ib"})
    P = P.merge(D, on=["ia", "ib"], how="left")
    P["id_a"] = d.id.values[P.ia]; P["id_b"] = d.id.values[P.ib]
    P["smiles_a"] = d["neut-smiles"].values[P.ia]; P["smiles_b"] = d["neut-smiles"].values[P.ib]
    print(f"MCS ok for {int(P.mcs_ok.fillna(0).sum())}/{len(P)} pairs", flush=True)

    # ---- modalities: oriented percentile ranks over the whole eval set ----
    M = pd.DataFrame({"id": d.id, "label": d.label, "smiles": d["neut-smiles"]})
    orient = {}
    for c in SCORE_COLS:
        s = d[c]; msk = s.notna()
        au = roc_auc_score(d.label[msk], s[msk]); sign = 1 if au >= 0.5 else -1
        orient[c] = (sign, max(au, 1 - au))
        M[c] = s.values
        M["pct_" + c] = (sign * s).rank(pct=True).values         # NaN stays NaN
    base = load_arm(R / "headft_affcache/base/scores")
    seeds = [load_arm(R / f"headft_affcache/lightning_top300_seed{s}/scores") for s in range(5)]
    idx = pd.Index(d.id)
    M["base_noft"] = base.affinity_probability_binary.reindex(idx).values
    M["ft300"] = np.mean([s.affinity_probability_binary.reindex(idx).values for s in seeds], axis=0)
    M["disagree_noft"] = (base.affinity_probability_binary1 - base.affinity_probability_binary2).abs().reindex(idx).values
    M["disagree_ft300"] = np.mean([(s.affinity_probability_binary1 - s.affinity_probability_binary2).abs().reindex(idx).values for s in seeds], axis=0)
    M["cross_seed_sd"] = np.std([s.affinity_probability_binary.reindex(idx).values for s in seeds], axis=0, ddof=1)
    for c in ("base_noft", "ft300"):
        M["pct_" + c] = M[c].rank(pct=True)
        orient[c] = (1, roc_auc_score(M.label, M[c]))
    pd.DataFrame([(k, v[0], v[1]) for k, v in orient.items()], columns=["modality", "orientation", "auroc"]).to_csv(out / "modality_auroc.csv", index=False)
    print("modality AUROC:", {k: round(v[1], 3) for k, v in orient.items()}, flush=True)

    # ---- pose + confidence for every pair member ----
    members = pd.unique(np.concatenate([P.id_a.values, P.id_b.values]))
    ci = json.load(open(R / "outputs_affcache/cache_index.json"))
    jobs = []
    for rid in members:
        ch = ci[rid].split("/outputs_affcache/")[1].split("/")[0]
        base_dir = R / "outputs_chunks_full" / ch / f"boltz_results_{ch}" / "predictions" / rid
        jobs.append((rid, str(base_dir / f"{rid}_model_0.cif"), str(base_dir / f"confidence_{rid}_model_0.json")))
    print(f"reading {len(jobs)} poses + confidence", flush=True)
    with Pool(a.procs) as pool:
        res = pool.map(pose_record, jobs, chunksize=8)
    C = pd.DataFrame([r for r, _ in res])
    nres = pd.Series([len(v) for _, v in res if v is not None]).mode().iloc[0]
    V = np.full((len(res), nres), np.nan, dtype=np.float32)
    for k, (_, v) in enumerate(res):
        if v is not None and len(v) == nres: V[k] = v
    C["row"] = np.arange(len(C))
    M = M[M.id.isin(members)].merge(C, on="id", how="left")
    M.to_csv(out / "compounds.csv", index=False)
    P.to_csv(out / "pairs.csv", index=False)
    np.savez(out / "contacts.npz", ids=np.array(C.id), mind=V)
    print(f"[{T}] DONE: {len(P)} pairs, {len(M)} compounds, {V.shape[1]} residues -> {out}", flush=True)

if __name__ == "__main__":
    main()

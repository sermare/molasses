"""Where do the poses of the compounds each model ranks highest sit on the protein? Per-residue contact frequency (ligand heavy atom within 5 A of the residue,
as in pose_density_residue.py) over the top-1% compounds of No-FT and of head-FT (N=300, 5-seed mean), against the library-wide density map.

Poses come from Pass-1 (the frozen structure model); head fine-tuning only changes the affinity head's scores, so the POSE of a given compound is identical for
No-FT and head-FT by construction. What can differ is WHICH compounds each model puts on top, and so which pocket residues the top-ranked poses touch.
Results are cached in results/analysis/<t>/localization_top1pct.csv.
"""
import os, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, "/global/scratch/users/sergiomar10/boltzaff/pipeline_local")
import bft_common as bc
from pose_density_residue import residue_atoms_and_ligand

CUT = 5.0


def cif_index(t):
    """compound id -> model_0.cif path over every Pass-1 chunk folder (tail-fold folders included)."""
    base = bc.RUNS / t / "outputs_chunks_full"; idx = {}
    def one(c):
        out = {}
        for r in os.listdir(base / c):
            if r.startswith("boltz_results_"):
                P = base / c / r / "predictions"
                for i in os.listdir(P): out[i] = P / i / f"{i}_model_0.cif"
        return out
    cks = [c for c in os.listdir(base) if c.startswith("chunk_")]
    with ThreadPoolExecutor(16) as ex:
        for d in ex.map(one, cks): idx.update(d)
    return idx


def contacts(path):
    try:
        names, nums, ratoms, ca, lig = residue_atoms_and_ligand(str(path))
        if len(lig) == 0: return None
        return np.array([np.sqrt(((a[:, None, :] - lig[None, :, :]) ** 2).sum(-1)).min() <= CUT for a in ratoms], float)
    except Exception:
        return None


def top_ids(t, frac=0.01):
    S = bc.scores(t)
    if S is None: return None
    k = int(np.ceil(frac * len(S))); ft = bc.ft_mean(S, 300)
    return {"No-FT": set(S.id.values[np.argsort(-S.noft_p.values, kind="stable")[:k]]), "head-FT": set(S.id.values[np.argsort(-ft.values, kind="stable")[:k]])}


def profile(t, force=False):
    """DataFrame per residue: library (pose-density map), top1_noft, top1_ft contact frequencies; plus n poses used. None if not available."""
    cp = bc.AN / t / "localization_top1pct.csv"
    if cp.exists() and not force: return pd.read_csv(cp)
    T = top_ids(t); lib = bc.RUNS / t / "pose_density/residue_density.csv"
    if T is None or not lib.exists(): return None
    idx = cif_index(t); L = pd.read_csv(lib); n = len(L); out = L[["res_index", "res_name", "res_num"]].copy(); out["library"] = L.contact_frac.values; used = {}
    allids = sorted(T["No-FT"] | T["head-FT"]); C = {}
    with ThreadPoolExecutor(8) as ex:
        for i, c in zip(allids, ex.map(lambda i: contacts(idx[i]) if i in idx else None, allids)):
            if c is not None and len(c) == n: C[i] = c
    for nm, key in (("No-FT", "top1_noft"), ("head-FT", "top1_ft")):
        rows = [C[i] for i in T[nm] if i in C]; used[key] = len(rows); out[key] = np.mean(rows, axis=0) if rows else np.nan
    out["n_noft"] = used["top1_noft"]; out["n_ft"] = used["top1_ft"]; out["n_overlap"] = len(T["No-FT"] & T["head-FT"])
    cp.parent.mkdir(parents=True, exist_ok=True); out.to_csv(cp, index=False); return out

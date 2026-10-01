#!/usr/bin/env python3
"""Per-target prep for notebooks/04_structural_cliffs.ipynb.  usage: prep_cliffs.py <target> [--procs 4] [--out DIR] [--force]

Runs only for a target whose No-FT + 15 head-FT arms are fully scored (bft_common.done_targets('scores')) and whose cache
results/analysis/<t>/cliffs/ is missing or older than the newest score file.  Reuses the functions of cliff_pairs.py (imported, not
modified; same pair definition: ECFP4 Tanimoto >= 0.6, cliff = active + inactive, conserved = both active) and writes
  pairs.csv, compounds.csv, contacts.npz, modality_auroc.csv     (same as cliff_pairs.py)
  scores_seeds_eval.csv  (noft_p, noft_dis, ft_p{s}, ft_dis{s} for N=300, whole eval set)   scores_eval.csv (label, score_boltz2, base_noft, ft300, smiles)
CPU only; at most 8 processes."""
import argparse, glob, json, os, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
import bft_common as bc
import cliff_pairs as cp
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from sklearn.metrics import roc_auc_score
RDLogger.DisableLog("rdApp.*")
FILES = ["pairs.csv", "compounds.csv", "contacts.npz", "modality_auroc.csv", "scores_eval.csv", "scores_seeds_eval.csv"]


def cache_dir(t): return bc.AN / t / "cliffs"


def newest_score_mtime(t):
    fs = []
    for arm in ["base"] + [f"top300_seed{s}" for s in bc.SEEDS]:
        d = bc.RUNS / t / "headft_affcache" / ("base" if arm == "base" else f"lightning_{arm}") / "scores"
        fs += glob.glob(str(d / "chunk_*.csv"))
    return max((os.path.getmtime(f) for f in fs), default=0)


def cache_state(t, done=None):
    """'ready' | 'stale' | 'missing' | 'not_scored'"""
    if t not in (bc.done_targets("scores") if done is None else done): return "not_scored"
    d = cache_dir(t)
    if not all((d / f).exists() for f in FILES): return "missing"
    return "stale" if os.path.getmtime(d / "pairs.csv") < newest_score_mtime(t) else "ready"


def run(T, procs=4, out=None, sim=0.6):
    procs = min(int(procs), 8); R = bc.RUNS / T; out = Path(out) if out else cache_dir(T); out.mkdir(parents=True, exist_ok=True)
    S = bc.scores(T)
    if S is None: raise SystemExit(f"[{T}] not fully scored")
    d = pd.read_csv(bc.DATA / f"{T}.csv"); d["id"] = f"{T}_" + d.CID.astype(str)
    ev = set(bc.eval_ids(T)); d = d[d.id.isin(ev)].reset_index(drop=True); d["label"] = d.target_active_v2.astype(int)
    print(f"[{T}] eval compounds {len(d)}, actives {int(d.label.sum())}", flush=True)
    # ---- whole-eval-set score tables (for the notebook's library-level / seed analyses) ----
    sd = pd.DataFrame({"id": S.id, "label": S.label, "noft_p": S.noft_p, "noft_dis": S.noft_dis})
    for s in bc.SEEDS: sd[f"ft_p{s}"] = S[f"ft300_p{s}"]; sd[f"ft_dis{s}"] = S[f"ft300_dis{s}"]
    sd.to_csv(out / "scores_seeds_eval.csv", index=False)
    pd.DataFrame({"id": S.id, "label": S.label, "score_boltz2": S.score_boltz2, "smiles": S.smiles, "base_noft": S.noft_p,
                  "ft300": bc.ft_mean(S, 300)}).to_csv(out / "scores_eval.csv", index=False)
    # ---- pair mining (identical to cliff_pairs.py) ----
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    mols = [Chem.MolFromSmiles(s) for s in d["neut-smiles"]]; fps = [gen.GetFingerprint(m) if m is not None else None for m in mols]
    dummy = next(f for f in fps if f is not None); allf = [f if f is not None else dummy for f in fps]; rows = []
    for i in np.where((d.label == 1) & np.array([f is not None for f in fps]))[0]:
        sims = np.array(DataStructs.BulkTanimotoSimilarity(fps[i], allf)); sims[i] = 0
        for j in np.where(sims >= sim)[0]:
            if fps[j] is None: continue
            if d.label[j] == 1 and j < i: continue
            rows.append((int(i), int(j), float(sims[j]), int(d.label[j])))
    P = pd.DataFrame(rows, columns=["ia", "ib", "sim", "b_active"]); P["kind"] = np.where(P.b_active == 0, "cliff", "conserved")
    print(f"pairs: cliff {int((P.kind=='cliff').sum())}, conserved {int((P.kind=='conserved').sum())}", flush=True)
    if len(P) == 0: raise SystemExit(f"[{T}] no pairs")
    with Pool(procs) as pool:
        diffs = pool.map(cp.pair_diff, [(int(r.ia), int(r.ib), d["neut-smiles"][r.ia], d["neut-smiles"][r.ib]) for r in P.itertuples()], chunksize=8)
    P = P.merge(pd.DataFrame(diffs).rename(columns={"a": "ia", "b": "ib"}), on=["ia", "ib"], how="left")
    P["id_a"] = d.id.values[P.ia]; P["id_b"] = d.id.values[P.ib]; P["smiles_a"] = d["neut-smiles"].values[P.ia]; P["smiles_b"] = d["neut-smiles"].values[P.ib]
    print(f"MCS ok for {int(P.mcs_ok.fillna(0).sum())}/{len(P)} pairs", flush=True)
    # ---- modalities: oriented percentile ranks ----
    M = pd.DataFrame({"id": d.id, "label": d.label, "smiles": d["neut-smiles"]}); orient = {}
    for c in cp.SCORE_COLS:
        s = d[c]; msk = s.notna(); au = roc_auc_score(d.label[msk], s[msk]); sign = 1 if au >= 0.5 else -1
        orient[c] = (sign, max(au, 1 - au)); M[c] = s.values; M["pct_" + c] = (sign * s).rank(pct=True).values
    Si = S.set_index("id").reindex(d.id)
    M["base_noft"] = Si.noft_p.values; M["ft300"] = bc.ft_mean(Si, 300).values
    M["disagree_noft"] = Si.noft_dis.values; M["disagree_ft300"] = Si[[f"ft300_dis{s}" for s in bc.SEEDS]].mean(axis=1).values
    M["cross_seed_sd"] = Si[[f"ft300_p{s}" for s in bc.SEEDS]].std(axis=1, ddof=1).values
    for c in ("base_noft", "ft300"):
        M["pct_" + c] = M[c].rank(pct=True); orient[c] = (1, roc_auc_score(M.label, M[c]))
    pd.DataFrame([(k, v[0], v[1]) for k, v in orient.items()], columns=["modality", "orientation", "auroc"]).to_csv(out / "modality_auroc.csv", index=False)
    # ---- pose + confidence of pair members ----
    members = pd.unique(np.concatenate([P.id_a.values, P.id_b.values]))
    ci = json.load(open(R / "outputs_affcache/cache_index.json")); jobs = []
    for rid in members:
        ch = ci[rid].split("/outputs_affcache/")[1].split("/")[0]
        bd = R / "outputs_chunks_full" / ch / f"boltz_results_{ch}" / "predictions" / rid
        jobs.append((rid, str(bd / f"{rid}_model_0.cif"), str(bd / f"confidence_{rid}_model_0.json")))
    print(f"reading {len(jobs)} poses + confidence", flush=True)
    with Pool(procs) as pool:
        res = pool.map(cp.pose_record, jobs, chunksize=8)
    C = pd.DataFrame([r for r, _ in res]); nok = sum(v is not None for _, v in res)
    if nok < 0.5 * len(res): raise SystemExit(f"[{T}] only {nok}/{len(res)} poses readable")
    nres = pd.Series([len(v) for _, v in res if v is not None]).mode().iloc[0]
    V = np.full((len(res), nres), np.nan, dtype=np.float32)
    for k, (_, v) in enumerate(res):
        if v is not None and len(v) == nres: V[k] = v
    C["row"] = np.arange(len(C)); M = M[M.id.isin(members)].merge(C, on="id", how="left")
    # write pairs last: its mtime is the 'cache is fresh' marker
    M.to_csv(out / "compounds.csv", index=False); np.savez(out / "contacts.npz", ids=np.array(C.id), mind=V); P.to_csv(out / "pairs.csv", index=False)
    print(f"[{T}] DONE: {len(P)} pairs, {len(M)} compounds, {V.shape[1]} residues -> {out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("target"); ap.add_argument("--procs", type=int, default=4); ap.add_argument("--out"); ap.add_argument("--force", action="store_true")
    a = ap.parse_args(); st = cache_state(a.target)
    if st == "not_scored": sys.exit(f"[{a.target}] pending: not fully scored")
    if st == "ready" and not a.force and not a.out: sys.exit(f"[{a.target}] cache is up to date")
    run(a.target, a.procs, a.out)

#!/usr/bin/env python3
"""Prepare training-set variants for head-FT: a data-scaling curve + composition contrasts.

Fixed candidate pool = top-POOL by Boltz-2 score; shared leak-free eval = rank > POOL, identical
across every condition so differences isolate training data (size + composition). Reuses the Pass-2
affinity cache (no re-folding). Writes train_ids_<cond>.txt (cond = '<comp>_N<n>'), ml_table*,
eval_ids/eval_chunks, and conds.txt (the grid the sbatch iterates).
"""
import argparse, gzip
from pathlib import Path
import numpy as np, pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

ROOT = Path("/global/scratch/users/sergiomar10/boltzaff")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="588689")
    ap.add_argument("--pool", type=int, default=1000)
    ap.add_argument("--val-tail", type=int, default=20)
    ap.add_argument("--chunk-size", type=int, default=1000)
    a = ap.parse_args()
    T = a.target; R = ROOT / "results/runs" / T
    out = R / "ft_inputs_variants"; out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(ROOT / f"data/{T}_results.csv"); df["CID"] = df.CID.astype(int)
    ids = [int(x.strip().removeprefix(T + "_")) for x in gzip.open(ROOT / f"BoltzFT/data/splits/{T}.txt.gz", "rt")]
    df = df.set_index("CID").loc[ids].reset_index(); df["rank"] = np.arange(1, len(df) + 1)
    df["scaf"] = [MurckoScaffold.MurckoScaffoldSmiles(mol=m) if (m := Chem.MolFromSmiles(str(s))) else ""
                  for s in df["neut-smiles"]]
    df["cx"] = [f"{T}_{c}" for c in df.CID]

    ml = pd.DataFrame({"complex_id": df.cx, "ligand_id": df.CID, "smiles": df["neut-smiles"],
                       "is_binder": df.Active_v2.astype(int), "boltz_rank": df["rank"],
                       "affinity_probability_binary": df.affinity_probability_binary,
                       "dataset": "MF-PCBA", "target_id": T})
    ml.to_csv(out / "ml_table.csv", index=False)

    pool = df.iloc[:a.pool].reset_index(drop=True)
    evald = df.iloc[a.pool:]
    (out / "eval_ids.txt").write_text("\n".join(evald.cx) + "\n")
    cdir = out / "eval_chunks"; cdir.mkdir(exist_ok=True)
    ev = evald.cx.tolist()
    for i in range(0, len(ev), a.chunk_size):
        (cdir / f"chunk_{i // a.chunk_size:03d}.txt").write_text("\n".join(ev[i:i + a.chunk_size]) + "\n")

    act = pool[pool.Active_v2 == 1]; ina = pool[pool.Active_v2 == 0]
    pool_by_score = pool.sort_values("affinity_probability_binary", ascending=False)

    def diverse_rr(frame, n):                                   # round-robin one per scaffold, score-desc
        by = {}
        for _, r in frame.sort_values("affinity_probability_binary", ascending=False).iterrows():
            by.setdefault(r.scaf or f"_acyc_{r.CID}", []).append(r.cx)
        lists = list(by.values()); picked = []
        while len(picked) < n and any(lists):
            for l in lists:
                if l: picked.append(l.pop(0))
                if len(picked) >= n: break
        return picked[:n]

    def select(comp, n):
        if comp == "top":     return pool_by_score.head(n).cx.tolist()
        if comp == "random":  return pool.sample(n, random_state=0).cx.tolist()
        if comp == "balanced":
            nb = min(len(act), n // 2)
            return act.sort_values("affinity_probability_binary", ascending=False).head(nb).cx.tolist() \
                 + ina.sample(n - nb, random_state=0).cx.tolist()
        if comp == "hardneg":
            na = min(len(act), n)
            return act.cx.tolist()[:na] \
                 + ina.sort_values("affinity_probability_binary", ascending=False).head(n - na).cx.tolist()
        if comp == "diverse": return diverse_rr(pool, n)
        if comp == "stratified":
            s = pool_by_score.reset_index(drop=True)
            idx = np.linspace(0, len(s) - 1, n).round().astype(int)
            return s.iloc[idx].cx.tolist()
        if comp == "actdiv":
            nb = min(len(act), n // 2)
            return diverse_rr(act, nb) + ina.sample(n - nb, random_state=0).cx.tolist()
        raise ValueError(comp)

    grid = [("top", n) for n in (40, 100, 300, 600, 1000)] \
         + [(c, 300) for c in ("random", "balanced", "hardneg", "diverse", "stratified", "actdiv")] \
         + [(c, 600) for c in ("balanced", "hardneg", "diverse")]

    rows, keep, conds = [], set(), []
    for comp, n in grid:
        cx = list(dict.fromkeys(select(comp, n)))[:n]
        cond = f"{comp}_N{n}"; conds.append(cond)
        (out / f"train_ids_{cond}.txt").write_text("\n".join(cx) + "\n")
        keep |= set(cx)
        sub = ml[ml.complex_id.isin(cx)]
        rows.append(dict(cond=cond, n=len(cx), n_active=int(sub.is_binder.sum()),
                         n_scaffold=int(df[df.cx.isin(cx)].scaf.replace("", np.nan).nunique())))
    (out / "conds.txt").write_text("\n".join(conds) + "\n")
    val = ev[-a.val_tail:]
    ml[ml.complex_id.isin(keep | set(val))].to_csv(out / "ml_table_train.csv", index=False)

    print(f"pool=top{a.pool} ({int(pool.Active_v2.sum())} actives)  eval=rank>{a.pool} "
          f"({len(evald)} cmpds, {int(evald.Active_v2.sum())} actives, {len(cdir.glob('chunk_*.txt') and list(cdir.glob('chunk_*.txt')))} chunks)")
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"{len(conds)} conditions x 5 seeds = {len(conds)*5} runs  ->  {out}")

if __name__ == "__main__":
    main()

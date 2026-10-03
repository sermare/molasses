#!/usr/bin/env python3
"""Data-leakage checks for the head-FT evaluation, for every fully scored target.  python pipeline_local/leakage_check.py  -> results/analysis/leakage_checks.csv

1. ID overlap: training ids (top-40/100/300) vs the evaluation ids, and the 20 validation ids vs the training ids.
2. Duplicates: evaluation compounds whose canonical SMILES (stereo removed) equals a training compound's.
3. Near-duplicates: highest ECFP4 Tanimoto of each evaluation compound to the N=300 training set (any label); share >= 0.8 / >= 0.6.
4. Leakage-controlled metric: No-FT and head-FT (N=300, 5 seeds) average precision on the evaluation compounds that are NOT close to any training compound (Tc < 0.6 / < 0.4),
   with the head-FT / No-FT ratio, compared with the ratio on the full evaluation set. If fine-tuning only helped by memorising near-duplicates, the ratio would collapse there.
Facts from the code (not computed here): the checkpoint evaluated is last.ckpt (no model selection on any held-out data); hyperparameters are the authors' fixed values;
the validation ids are the last 20 evaluation ids, used only to log a validation loss."""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/global/scratch/users/sergiomar10/boltzaff/pipeline_local")
import numpy as np, pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from sklearn.metrics import average_precision_score as ap
import bft_common as bc, bft_rerank as br
RDLogger.DisableLog("rdApp.*")


def canon(smiles):
    out = []
    for s in smiles:
        m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
        out.append(Chem.MolToSmiles(m, isomericSmiles=False) if m is not None else None)
    return out


rows = []
for t in bc.done_targets("scores"):
    S = bc.scores(t); R = bc.RUNS / t / "ft_inputs_full"; ev = [l.strip() for l in open(R / "eval_ids.txt") if l.strip()]; evset = set(ev); val = ev[-20:]
    tr = {N: bc.train_ids(t, N) for N in bc.BUDGETS}
    r = dict(target=bc.SHORT[t], n_eval=len(ev), n_eval_actives=int(S.label.sum()))
    for N in bc.BUDGETS: r[f"train{N}_in_eval"] = len(set(tr[N]) & evset)
    r["val_in_train300"] = len(set(val) & set(tr[300])); r["train_nested_40_100_300"] = bool(set(tr[40]) <= set(tr[100]) <= set(tr[300]))
    tt = br.sim_table(t); trsmi = tt["neut-smiles"].reindex(tr[300]).values; evsmi = S.smiles.values
    ctr = set(x for x in canon(trsmi) if x); cev = canon(evsmi); r["eval_exact_dup_of_train300"] = int(sum(1 for x in cev if x in ctr))
    ref = [f for f in br.fps(trsmi) if f is not None]; tc = br.max_sim(evsmi, ref); r["share_eval_tc>=0.8_%"] = round(100 * np.mean(tc >= 0.8), 2); r["share_eval_tc>=0.6_%"] = round(100 * np.mean(tc >= 0.6), 2)
    y = S.label.values.astype(int); seeds = [S[f"ft300_p{s}"].values for s in bc.SEEDS]
    for name, mask in (("all", np.ones(len(S), bool)), ("tc<0.6", tc < 0.6), ("tc<0.4", tc < 0.4)):
        if y[mask].sum() < 10: r[f"ratio_{name}"] = np.nan; continue
        b = ap(y[mask], S.noft_p.values[mask]); f = np.mean([ap(y[mask], s[mask]) for s in seeds])
        r[f"n_{name}"] = int(mask.sum()); r[f"actives_{name}"] = int(y[mask].sum()); r[f"noft_ap_{name}"] = round(b, 4); r[f"ft_ap_{name}"] = round(f, 4); r[f"ratio_{name}"] = round(f / b, 2)
    rows.append(r); print(r, flush=True)
D = pd.DataFrame(rows).set_index("target"); D.to_csv(bc.AN / "leakage_checks.csv"); print(D.T.to_string())

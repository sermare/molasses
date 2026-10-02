"""Balanced vs top-N vs No-FT on near-identical pairs (588689 only: the balanced N=300 arms exist for that target).
All three models are scored on the same held-out compounds (the 48,985 ranked below 1000 by the dataset's Boltz-2 score), so pairs are restricted to pairs
with both members in that set. Ground truth is the binary label only; 'large delta' thresholds are calibrated on both-active (conserved) pairs.

    from bft_balanced_pairs import load, pair_table, accuracy, sensitivity, series_bootstrap
"""
import glob, sys
import numpy as np, pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
sys.path.insert(0, "/global/scratch/users/sergiomar10/boltzaff/pipeline_local")
import bft_common as bc

T0 = "588689"


def load():
    """dict(E, y, scores={model: array over E (seed-mean probability)}, seeds={model: [5 arrays]}) or None when the balanced arms are missing."""
    R = bc.RUNS / T0
    if not (R / "headft_variants_scores/balanced_N300_seed0").exists(): return None
    S = bc.scores(T0).set_index("id"); E = [l.strip() for l in open(R / "ft_inputs_variants/eval_ids.txt") if l.strip()]
    def arm(cond, s):
        f = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(str(R / f"headft_variants_scores/{cond}_seed{s}/chunk_*.csv")))]); f["sample_id"] = f.sample_id.astype(str)
        return f.drop_duplicates("sample_id").set_index("sample_id").affinity_probability_binary.reindex(E).values
    top = [arm("top_N300", s) for s in bc.SEEDS]; bal = [arm("balanced_N300", s) for s in bc.SEEDS]
    return dict(E=E, y=S.loc[E, "label"].values.astype(int), seeds={"No-FT": [S.loc[E, "noft_p"].values], "head-FT top-N": top, "head-FT balanced": bal},
                scores={"No-FT": S.loc[E, "noft_p"].values, "head-FT top-N": np.mean(top, 0), "head-FT balanced": np.mean(bal, 0)})


def pair_table(D):
    """Cliff and conserved pairs fully inside the held-out set, with a series id (connected component of the pair graph) and size/logP differences."""
    P = pd.read_csv(bc.AN / f"{T0}/cliffs/pairs.csv"); ids = pd.Index(sorted(set(P.id_a) | set(P.id_b))); n = len(ids)
    _, comp = connected_components(coo_matrix((np.ones(len(P)), (ids.get_indexer(P.id_a), ids.get_indexer(P.id_b))), shape=(n, n)), directed=False)
    P["series"] = P.id_a.map(dict(zip(ids, comp))); pos = pd.Series(np.arange(len(D["E"])), index=D["E"])
    P = P[P.id_a.isin(pos.index) & P.id_b.isin(pos.index)].copy(); P["ia"] = pos[P.id_a].values; P["ib"] = pos[P.id_b].values
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Crippen
    RDLogger.DisableLog("rdApp.*")
    C = pd.read_csv(bc.AN / f"{T0}/cliffs/compounds.csv").set_index("id"); need = sorted(set(P.id_a) | set(P.id_b)); prop = {}
    for i in need:
        m = Chem.MolFromSmiles(C.smiles.get(i, "")) if i in C.index else None
        prop[i] = (m.GetNumHeavyAtoms(), Crippen.MolLogP(m)) if m is not None else (np.nan, np.nan)
    P["dheavy"] = [prop[a][0] - prop[b][0] for a, b in zip(P.id_a, P.id_b)]; P["dlogP"] = [prop[a][1] - prop[b][1] for a, b in zip(P.id_a, P.id_b)]
    return P


def _delta(score, P, scale):
    s = pd.Series(score).rank(pct=True).values if scale == "percentile" else np.log(np.clip(score, 1e-6, 1 - 1e-6) / (1 - np.clip(score, 1e-6, 1 - 1e-6)))
    return s[P.ia.values] - s[P.ib.values]            # active (id_a) minus inactive partner for cliffs


def accuracy(score, P):
    """Share of cliff pairs with the active scored above its inactive partner (ties count one half)."""
    d = _delta(score, P, "logodds"); return float(np.mean((d > 0) + 0.5 * (d == 0)))


def sensitivity(score, Pc, Pk, scale="logodds", spec=0.90):
    """Share of cliffs whose active-minus-inactive delta exceeds the threshold that conserved pairs exceed (in |delta|) only (1 - spec) of the time."""
    thr = np.quantile(np.abs(_delta(score, Pk, scale)), spec); return float(np.mean(_delta(score, Pc, scale) > thr))


def series_bootstrap(P, fn, B=300, seed=0):
    """Resample whole series; fn(df) -> dict of statistics; returns the array of draws as a DataFrame."""
    rng = np.random.default_rng(seed); groups = [g for _, g in P.groupby("series")]; out = []
    for _ in range(B): out.append(fn(pd.concat([groups[j] for j in rng.integers(0, len(groups), len(groups))])))
    return pd.DataFrame(out)

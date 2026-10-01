"""Shared loaders for the self-filling notebooks. Everything here is target-generic: a target that has not finished a stage simply
returns None / an empty frame and the notebook skips it (and fills in on the next re-run).

    from bft_common import *
    style()                           # black text, no bold, white background
    TARGETS, SHORT, PAPER_BASE, PAPER_FT, RATE
    status()                          # DataFrame: pipeline stage per target
    done_targets("scores")            # targets whose No-FT + 15 head-FT arms are fully scored
    scores(t)                         # per-compound table: label, noft_p, noft_dis, ft{N}_p{s}, ft{N}_dis{s}, smiles, dataset scores
    per_seed(t), spread(t), table2(t) # outputs of eval_headft_seeds.py (None if absent)
"""
import glob, os, sys
from pathlib import Path
import numpy as np, pandas as pd, matplotlib as mpl

ROOT = Path("/global/scratch/users/sergiomar10/boltzaff"); RUNS = ROOT / "results/runs"; AN = ROOT / "results/analysis"; DATA = ROOT / "data/mf-pcba_test"
TARGETS = ["588689", "504329", "1053173-743445", "493248-485317", "434954-2097", "540297-493091", "463203-2650", "624273-588549"]
SHORT = {t: t.split("-")[-1] for t in TARGETS}
BUDGETS = [40, 100, 300]; SEEDS = [0, 1, 2, 3, 4]
BLK = "#000000"; OUR = "#2a78d6"; THEIR = "#eb6834"; NEU = "#666666"
_tc = pd.read_csv(ROOT / "BoltzFT/figures/data/trainer_control.csv"); _tc["target"] = _tc.target.astype(str)
_tc["t"] = _tc.target.map({SHORT[t]: t for t in TARGETS}).fillna(_tc.target)
PAPER_BASE = _tc[_tc.arm == "base"].set_index("t"); PAPER_FT = _tc[_tc.arm == "lightning"].set_index(["t", "n_train"])
RATE = {t: float(PAPER_BASE.loc[t, "n_active"] / PAPER_BASE.loc[t, "n_eval"]) for t in TARGETS}     # active rate = AP of a random ranking


def style():
    mpl.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white", "text.color": BLK,
        "axes.labelcolor": BLK, "axes.titlecolor": BLK, "xtick.color": BLK, "ytick.color": BLK, "axes.edgecolor": BLK, "font.size": 10,
        "font.weight": "normal", "axes.titleweight": "normal", "axes.labelweight": "normal", "axes.grid": True, "grid.color": "#e6e6e6",
        "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 100})


def _n(path_glob):
    return len(glob.glob(str(path_glob)))


def eval_ids(t):
    p = RUNS / t / "ft_inputs_full/eval_ids.txt"
    return [l.strip() for l in open(p) if l.strip()] if p.exists() else None


def n_eval_chunks(t):
    return _n(RUNS / t / "ft_inputs_full/eval_chunks/chunk_*.txt")


def arm_scored(t, arm):
    """arm = 'base' or 'top300_seed0' ... -> True when every eval chunk has a score csv."""
    k = n_eval_chunks(t)
    d = RUNS / t / "headft_affcache" / ("base" if arm == "base" else f"lightning_{arm}") / "scores"
    return k > 0 and _n(d / "chunk_*.csv") >= k


def scored_arms(t):
    arms = ["base"] + [f"top{N}_seed{s}" for N in BUDGETS for s in SEEDS]
    return [a for a in arms if arm_scored(t, a)]


def done_targets(stage="scores"):
    if stage == "scores":       # No-FT + all 15 head-FT arms
        return [t for t in TARGETS if len(scored_arms(t)) == 16]
    if stage == "table2":
        return [t for t in TARGETS if (AN / t / "table2.csv").exists()]
    raise ValueError(stage)


def status():
    rows = []
    for t in TARGETS:
        R = RUNS / t; lib = _n(R / "inputs_full/*.yaml") if (R / "inputs_full").exists() else 0
        rows.append(dict(target=SHORT[t], library=lib or np.nan, ft_inputs=(R / "ft_inputs_full/eval_ids.txt").exists(),
                         arms_scored=f"{len(scored_arms(t))}/16", table2=(AN / t / "table2.csv").exists()))
    return pd.DataFrame(rows).set_index("target")


def _arm_df(t, arm):
    d = RUNS / t / "headft_affcache" / ("base" if arm == "base" else f"lightning_{arm}") / "scores"
    f = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(str(d / "chunk_*.csv")))], ignore_index=True)
    f["sample_id"] = f.sample_id.astype(str)
    return f.drop_duplicates("sample_id").set_index("sample_id")


_CACHE = {}


def scores(t, budgets=BUDGETS, seeds=SEEDS, require_full=True):
    """One row per eval compound: label, smiles, the dataset's own scores, noft_p / noft_dis, and for each (N, seed)
    ft{N}_p{s} (binary probability) and ft{N}_dis{s} (two-head disagreement). None if the target is not fully scored."""
    key = (t, tuple(budgets), tuple(seeds))
    if key in _CACHE: return _CACHE[key]
    need = ["base"] + [f"top{N}_seed{s}" for N in budgets for s in seeds]
    if require_full and not all(arm_scored(t, a) for a in need): return None
    ev = eval_ids(t)
    d = pd.read_csv(DATA / f"{t}.csv"); d["id"] = f"{t}_" + d.CID.astype(str)
    if not d.id.isin(ev).any():   # fall back to the short assay id as prefix
        d["id"] = f"{SHORT[t]}_" + d.CID.astype(str)
    d = d.set_index("id").reindex(ev)
    out = pd.DataFrame({"id": ev, "label": d.target_active_v2.astype(int).values, "smiles": d["neut-smiles"].values})
    for c in d.columns:
        if c.startswith(("score_", "docking_score")): out[c] = d[c].values
    b = _arm_df(t, "base").reindex(ev)
    out["noft_p"] = b.affinity_probability_binary.values
    out["noft_dis"] = (b.affinity_probability_binary1 - b.affinity_probability_binary2).abs().values
    for N in budgets:
        for s in seeds:
            a = _arm_df(t, f"top{N}_seed{s}").reindex(ev)
            out[f"ft{N}_p{s}"] = a.affinity_probability_binary.values
            out[f"ft{N}_dis{s}"] = (a.affinity_probability_binary1 - a.affinity_probability_binary2).abs().values
    _CACHE[key] = out
    return out


def train_ids(t, n):
    p = RUNS / t / f"ft_inputs_full/train_ids_top{n}.txt"
    return [l.strip() for l in open(p) if l.strip()] if p.exists() else None


def _csv(t, name):
    p = AN / t / name
    return pd.read_csv(p) if p.exists() else None


def per_seed(t): return _csv(t, "per_seed.csv")
def spread(t): return _csv(t, "spread.csv")
def table2(t): return _csv(t, "table2.csv")


def average_precision(y, s):
    from sklearn.metrics import average_precision_score
    return average_precision_score(y, s)


def ft_mean(S, N):
    """Seed-mean head-FT probability at budget N."""
    return S[[f"ft{N}_p{s}" for s in SEEDS if f"ft{N}_p{s}" in S]].mean(axis=1)


def base_scores(t):
    """No-FT (our own Pass-2 scoring) for targets whose base arm is complete, even if the head-FT arms are not: id, label, noft_p. None otherwise."""
    if not arm_scored(t, "base"): return None
    ev = eval_ids(t); d = pd.read_csv(DATA / f"{t}.csv"); d["id"] = f"{t}_" + d.CID.astype(str); d = d.set_index("id").reindex(ev)
    b = _arm_df(t, "base").reindex(ev)
    return pd.DataFrame({"id": ev, "label": d.target_active_v2.astype(int).values, "noft_p": b.affinity_probability_binary.values})


def shipped_eval(t):
    """The dataset's own Boltz-2 (No-FT) probability on the paper's eval subset (selection/eval_subset.csv): CID, label, p. Available for all 8 targets now."""
    p = RUNS / t / "selection/eval_subset.csv"
    if not p.exists(): return None
    d = pd.read_csv(p)
    return pd.DataFrame({"id": f"{t}_" + d.CID.astype(str), "label": d.Active_v2.astype(int).values, "p": d.affinity_probability_binary.values})

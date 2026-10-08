"""CPU-only analyses for notebooks/08_latent_space_and_baselines.ipynb: other signals that can be derived from what we already have.
Every function takes a target id (bft_common.TARGETS) and returns a small DataFrame or dict; heavy results are cached in results/analysis/<t>/signals_*.csv.
Metrics are the project's (BoltzFT/evaluation/metrics.py): AP (auprc), EF@1%, BEDROC, on the common evaluation set (every compound ranked below 300)."""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, "/global/scratch/users/sergiomar10/boltzaff/BoltzFT/evaluation")
import bft_common as bc
from metrics import all_metrics

SEEDS = bc.SEEDS


def _m(y, s):
    m = all_metrics(np.asarray(y, int), np.asarray(s, float)); return dict(ap=m["auprc"], ef1=m["ef_1pct"], bedroc=m["bedroc"], auroc=m["auroc"])


def dataset(t):
    d = pd.read_csv(bc.DATA / f"{t}.csv"); d["id"] = f"{t}_" + d.CID.astype(str); return d.set_index("id")


# ---------------------------------------------------------------------------------------------------------------- 1. both heads and the affinity value
def heads_table(t):
    """Ranking quality by which output of the head is used: ensemble probability, head-1 probability, head-2 probability, -affinity value (lower value = tighter binding)."""
    ev = bc.eval_ids(t); y = dataset(t).target_active_v2.reindex(ev).astype(int).values; rows = []
    for grp, arms in (("No-FT", ["base"]), ("head-FT N=300", [f"top300_seed{s}" for s in SEEDS])):
        acc = {}
        for arm in arms:
            a = bc._arm_df(t, arm).reindex(ev)
            outs = {"ensemble probability": a.affinity_probability_binary, "head 1 probability": a.affinity_probability_binary1, "head 2 probability": a.affinity_probability_binary2,
                    "affinity value (negated)": -a.affinity_pred_value, "affinity value head 1 (negated)": -a.affinity_pred_value1, "affinity value head 2 (negated)": -a.affinity_pred_value2}
            for k, s in outs.items(): acc.setdefault(k, []).append(_m(y, s.values))
            acc.setdefault("_corr", []).append(dict(rho=float(a.affinity_pred_value.rank().corr(a.affinity_probability_binary.rank()))))
        for k, v in acc.items():
            if k == "_corr": rows.append(dict(target=bc.SHORT[t], arm=grp, output="Spearman(value, probability)", ap=np.mean([x["rho"] for x in v]), ef1=np.nan, bedroc=np.nan, auroc=np.nan))
            else: rows.append(dict(target=bc.SHORT[t], arm=grp, output=k, **{c: np.mean([x[c] for x in v]) for c in ("ap", "ef1", "bedroc", "auroc")}))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------------------------- 2. unwanted-substructure filters (PAINS, Brenk, NIH)
def filter_flags(t):
    """RDKit FilterCatalog flags per evaluation compound. The dataset itself contains no PAINS (its PAINS column is False everywhere and RDKit's PAINS catalog flags none),
    so PAINS cannot be tested; Brenk (unwanted/reactive groups) and NIH (reactive / promiscuous motifs) do flag compounds."""
    f = bc.AN / t / "signals_filters.csv"
    if f.exists(): return pd.read_csv(f)
    from rdkit import Chem, RDLogger
    from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
    RDLogger.DisableLog("rdApp.*"); C = FilterCatalogParams.FilterCatalogs; cats = {}
    for name in ("PAINS", "BRENK", "NIH"):
        p = FilterCatalogParams(); p.AddCatalog(getattr(C, name)); cats[name] = FilterCatalog(p)
    S = bc.scores(t); rows = []
    for i, s in zip(S.id, S.smiles):
        m = Chem.MolFromSmiles(str(s)); rows.append(dict(id=i, **{n.lower(): (bool(c.HasMatch(m)) if m is not None else False) for n, c in cats.items()}))
    D = pd.DataFrame(rows); D.to_csv(f, index=False); return D


def pains_table(t):
    F = filter_flags(t).set_index("id"); S = bc.scores(t); y = S.label.values; k = int(np.ceil(0.01 * len(S))); ft = bc.ft_mean(S, 300).values; rows = []
    for cat in ("pains", "brenk", "nih"):
        fl = F[cat].reindex(S.id).values.astype(bool)
        for name, sc in (("No-FT", S.noft_p.values), ("head-FT N=300", ft)):
            top = np.argsort(-sc)[:k]
            rows.append(dict(target=bc.SHORT[t], catalog=cat.upper(), arm=name, library_flagged=fl.mean(), actives_flagged=fl[y == 1].mean(), top1pct_flagged=fl[top].mean(),
                             active_rate_flagged_in_top1=y[top][fl[top]].mean() if fl[top].any() else np.nan, active_rate_clean_in_top1=y[top][~fl[top]].mean(),
                             ap_all=_m(y, sc)["ap"], ap_unflagged_only=_m(y[~fl], sc[~fl])["ap"] if y[~fl].sum() > 10 else np.nan))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------------------------- 3. scaffold-disjoint evaluation
def _scaffold(smi, generic):
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    m = Chem.MolFromSmiles(str(smi))
    if m is None: return None
    try:
        s = MurckoScaffold.GetScaffoldForMol(m)
        if generic: s = MurckoScaffold.MakeScaffoldGeneric(s)
        return Chem.MolToSmiles(s)
    except Exception: return None


def scaffold_table(t):
    """AP of No-FT and head-FT on all evaluation compounds, and on those whose Bemis-Murcko scaffold (exact or generic) is NOT shared with any of the 300 training compounds."""
    f = bc.AN / t / "signals_scaffolds.csv"
    S = bc.scores(t); ml = pd.read_csv(bc.RUNS / t / "ft_inputs_full/ml_table.csv").set_index("complex_id"); tr = ml.smiles.reindex(bc.train_ids(t, 300)).values
    if f.exists(): sc = pd.read_csv(f)
    else:
        sc = pd.DataFrame({"id": S.id, "exact": [_scaffold(s, False) for s in S.smiles], "generic": [_scaffold(s, True) for s in S.smiles]}); sc.to_csv(f, index=False)
    tr_ex = {_scaffold(s, False) for s in tr}; tr_ge = {_scaffold(s, True) for s in tr}
    y = S.label.values; ft = bc.ft_mean(S, 300).values; rows = []
    for name, mask in (("all evaluation compounds", np.ones(len(S), bool)), ("scaffold not in training (exact)", ~sc.exact.isin(tr_ex).values), ("scaffold not in training (generic)", ~sc.generic.isin(tr_ge).values)):
        if y[mask].sum() < 10: continue
        a0 = _m(y[mask], S.noft_p.values[mask])["ap"]; a1 = _m(y[mask], ft[mask])["ap"]
        rows.append(dict(target=bc.SHORT[t], subset=name, n=int(mask.sum()), actives=int(y[mask].sum()), noft_ap=a0, ft_ap=a1, ratio=a1 / a0))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------------------------- 4. physicochemical properties
def physchem(t):
    f = bc.AN / t / "signals_physchem.csv"
    if f.exists(): return pd.read_csv(f)
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, Lipinski
    RDLogger.DisableLog("rdApp.*"); S = bc.scores(t); rows = []
    for i, s in zip(S.id, S.smiles):
        m = Chem.MolFromSmiles(str(s))
        if m is None: rows.append(dict(id=i)); continue
        rows.append(dict(id=i, MW=Descriptors.MolWt(m), logP=Crippen.MolLogP(m), TPSA=rdMolDescriptors.CalcTPSA(m), heavy=m.GetNumHeavyAtoms(), rings=rdMolDescriptors.CalcNumRings(m),
                         aromatic_rings=rdMolDescriptors.CalcNumAromaticRings(m), HBD=Lipinski.NumHDonors(m), HBA=Lipinski.NumHAcceptors(m), rotatable=rdMolDescriptors.CalcNumRotatableBonds(m),
                         charge=Chem.GetFormalCharge(m), fsp3=rdMolDescriptors.CalcFractionCSP3(m)))
    D = pd.DataFrame(rows); D.to_csv(f, index=False); return D


PROPS = ["MW", "logP", "TPSA", "heavy", "rings", "aromatic_rings", "HBD", "HBA", "rotatable", "charge", "fsp3"]


def property_table(t):
    """AUROC of each property alone (>0.5: larger values go with activity) and the median property of the library, of the actives and of the top-1% picks of No-FT and head-FT."""
    P = physchem(t).set_index("id"); S = bc.scores(t); P = P.reindex(S.id); y = S.label.values; k = int(np.ceil(0.01 * len(S))); ft = bc.ft_mean(S, 300).values
    from sklearn.metrics import roc_auc_score
    top0 = np.argsort(-S.noft_p.values)[:k]; top1 = np.argsort(-ft)[:k]; rows = []
    for p in PROPS:
        v = P[p].values; ok = np.isfinite(v)
        rows.append(dict(target=bc.SHORT[t], property=p, auroc_alone=roc_auc_score(y[ok], v[ok]), median_library=np.nanmedian(v), median_actives=np.nanmedian(v[y == 1]), median_top1_noft=np.nanmedian(v[top0]), median_top1_ft=np.nanmedian(v[top1])))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------------------------- 5. potency tiers (DR) and primary-screen signal (SD)
def tier_table(t):
    """Among evaluation actives: share found in the top 1% by No-FT and by head-FT, split by the active's dose-response value DR (low / middle / high third; larger = larger DR)."""
    S = bc.scores(t); d = dataset(t); dr = d.DR.reindex(S.id).values; y = S.label.values; k = int(np.ceil(0.01 * len(S))); ft = bc.ft_mean(S, 300).values
    top0 = np.zeros(len(S), bool); top1 = np.zeros(len(S), bool); top0[np.argsort(-S.noft_p.values)[:k]] = True; top1[np.argsort(-ft)[:k]] = True
    a = (y == 1) & np.isfinite(dr); q = np.nanquantile(dr[a], [1 / 3, 2 / 3]); rows = []
    for name, m in (("lowest third of DR", a & (dr <= q[0])), ("middle third", a & (dr > q[0]) & (dr <= q[1])), ("highest third of DR", a & (dr > q[1]))):
        rows.append(dict(target=bc.SHORT[t], tier=name, actives=int(m.sum()), found_noft=top0[m].mean(), found_ft=top1[m].mean()))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------------------------- 6. cross-target promiscuity
def crosstarget():
    """Compounds (by PubChem CID) that appear in several of the eight assay libraries and are active in several: counts, and how the models score the promiscuous actives."""
    lab = {}
    for t in bc.TARGETS:
        f = bc.DATA / f"{t}.csv"
        if f.exists(): lab[t] = pd.read_csv(f, usecols=["CID", "target_active_v2"]).drop_duplicates("CID").set_index("CID").target_active_v2.astype(int)
    L = pd.DataFrame(lab); n_in = L.notna().sum(1); n_act = (L == 1).sum(1)
    return L, n_in, n_act


def promiscuity_table(t):
    L, n_in, n_act = crosstarget(); S = bc.scores(t); cid = S.id.str.split("_").str[-1].astype(int); y = S.label.values; ft = bc.ft_mean(S, 300).values
    na = n_act.reindex(cid).values; others = na - y                         # number of OTHER targets the compound is active in
    pct0 = pd.Series(S.noft_p.values).rank(pct=True).values; pct1 = pd.Series(ft).rank(pct=True).values; rows = []
    for name, m in (("active here only", (y == 1) & (others == 0)), ("active here and in >=1 other target", (y == 1) & (others >= 1)), ("inactive here, active elsewhere", (y == 0) & (others >= 1)), ("inactive everywhere", (y == 0) & (others == 0))):
        rows.append(dict(target=bc.SHORT[t], group=name, n=int(m.sum()), median_percentile_noft=np.nanmedian(pct0[m]) if m.any() else np.nan, median_percentile_ft=np.nanmedian(pct1[m]) if m.any() else np.nan))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------------------------- 7. supervised baselines (lgbm_warmstart.py) next to head-FT
def baseline_table():
    rows = []
    for t in bc.TARGETS:
        f = bc.AN / t / "lgbm_warmstart.csv"; p = bc.per_seed(t)
        if not f.exists() or p is None: continue
        L = pd.read_csv(f); base = L[L.method == "Boltz2_noFT(score)"].auprc.iloc[0]
        for N in (40, 100, 300):
            d = L[(L.n_train == N) & ~L.method.str.endswith("avg5") & (L.method != "Boltz2_noFT(score)")]
            for m, g in d.groupby("method"): rows.append(dict(target=bc.SHORT[t], n_train=N, method=m, ap=g.auprc.mean(), ap_sd=g.auprc.std(ddof=1), ap_noft=base, ratio=g.auprc.mean() / base))
            h = p[(p.arm == "headft") & (p.n_train == N)].ap; rows.append(dict(target=bc.SHORT[t], n_train=N, method="head-FT (Boltz-2)", ap=h.mean(), ap_sd=h.std(ddof=1), ap_noft=base, ratio=h.mean() / base))
    return pd.DataFrame(rows)

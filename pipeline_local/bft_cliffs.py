"""Helpers for notebooks/04_structural_cliffs.ipynb: ensure per-target caches (prep_cliffs.py), load + orient pair tables for every ready
target, pool them (series ids are prefixed by target so they never collide), and series-level resampling statistics.
Targets are the analysis unit for every cross-target number: metrics are computed per target and averaged with equal weight
(macro average); uncertainty resamples chemical series WITHIN each target."""
import subprocess, sys, os
from pathlib import Path
import numpy as np, pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import rankdata
sys.path.insert(0, str(Path(__file__).parent))
import bft_common as bc
import prep_cliffs as pc

MODS = [("score_boltz2", "Boltz-2 (dataset score)"), ("base_noft", "Boltz-2 No-FT (our pipeline)"), ("ft300", "Boltz-2 head-FT N=300 (5-seed)"),
        ("score_boltzina", "Boltzina"), ("score_boltzina_recycle1", "Boltzina recycle-1"), ("score_boltzina_mask", "Boltzina mask"),
        ("docking_score_boltzina", "Boltzina docking (3 identical columns)"), ("score_gnina", "GNINA"), ("score_vina", "AutoDock Vina")]
CONF = [("ligand_iptm", "ligand ipTM (= ipTM here)"), ("complex_plddt", "complex pLDDT"), ("confidence_score", "overall confidence"), ("complex_pde", "complex PDE"),
        ("disagree_noft", "two-head disagreement (No-FT)"), ("disagree_ft300", "two-head disagreement (FT)"), ("cross_seed_sd", "cross-seed SD (FT)")]
PROPS = [("MW", "molecular weight"), ("heavy", "heavy atoms"), ("logP", "logP"), ("TPSA", "polar surface area"), ("HBD", "H-bond donors"),
         ("HBA", "H-bond acceptors"), ("rotb", "rotatable bonds"), ("arom_rings", "aromatic rings"), ("fsp3", "fraction sp3")]


def ensure_caches(procs=4):
    """Run prep_cliffs.py for every fully scored target whose cliff cache is missing/stale; print pending for the rest. Returns {target: state}."""
    st = {}; done = bc.done_targets("scores")
    for t in bc.TARGETS:
        s = pc.cache_state(t, done)
        if s in ("missing", "stale"):
            print(f"[{bc.SHORT[t]}] cache {s}: running prep_cliffs.py (CPU, {min(procs, 8)} procs) ...", flush=True)
            r = subprocess.run(["nice", sys.executable, str(Path(__file__).parent / "prep_cliffs.py"), t, "--procs", str(min(procs, 8))],
                               capture_output=True, text=True, env={**os.environ, "OMP_NUM_THREADS": "4"})
            print("   " + (r.stdout.strip().splitlines() or ["(no output)"])[-1] + ("" if r.returncode == 0 else "  ERROR: " + r.stderr.strip()[-300:]), flush=True)
            s = pc.cache_state(t, done)
        st[t] = s
    return st


def pending_reason(t):
    if len(bc.scored_arms(t)) == 16: return "cliff cache could not be built"
    n = len(bc.scored_arms(t))
    if bc.eval_ids(t) is None: return f"eval split / head-FT inputs not built yet (scored arms {n}/16)"
    return f"scoring incomplete (scored arms {n}/16; needs No-FT + 15 head-FT arms)"


def _load_one(t):
    A = pc.cache_dir(t)
    P = pd.read_csv(A / "pairs.csv"); M = pd.read_csv(A / "compounds.csv"); Z = np.load(A / "contacts.npz", allow_pickle=True)
    Mi = M.set_index("id"); rng = np.random.default_rng(0)
    flip = (P.kind == "conserved") & (rng.random(len(P)) < 0.5)           # controls get a random a/b orientation; cliffs: a = the active
    swaps = [("id_a", "id_b"), ("smiles_a", "smiles_b"), ("diff_a_atoms", "diff_b_atoms"), ("mcs_a", "mcs_b"), ("ia", "ib")]
    swaps += [(c, "b_" + c[2:]) for c in P.columns if c.startswith("a_") and ("b_" + c[2:]) in P.columns]
    for x, y in swaps:
        tmp = P.loc[flip, x].copy(); P.loc[flip, x] = P.loc[flip, y].values; P.loc[flip, y] = tmp.values
    side = lambda col, ids: Mi[col].reindex(ids).values
    for m, _ in MODS: P["d_" + m] = side("pct_" + m, P.id_a) - side("pct_" + m, P.id_b)
    for f, _ in CONF: P["c_" + f] = side(f, P.id_a) - side(f, P.id_b)
    rowof = {i: k for k, i in enumerate(Z["ids"])}; MD = Z["mind"]
    ma = MD[[rowof[i] for i in P.id_a]]; mb = MD[[rowof[i] for i in P.id_b]]
    ca_, cb_ = ma <= 5.0, mb <= 5.0; inter = (ca_ & cb_).sum(1); uni = (ca_ | cb_).sum(1)
    P["pose_jdist"] = np.where(np.isnan(ma).any(1) | np.isnan(mb).any(1), np.nan, 1 - inter / np.maximum(uni, 1))
    P["pose_absdiff"] = np.nanmean(np.abs(ma - mb), axis=1)
    for p, _ in PROPS: P["dp_" + p] = P["a_" + p] - P["b_" + p]
    # series = connected components of the pair graph
    ids = pd.unique(np.concatenate([P.id_a.values, P.id_b.values])); ix = {k: i for i, k in enumerate(ids)}
    g = coo_matrix((np.ones(len(P)), ([ix[i] for i in P.id_a], [ix[i] for i in P.id_b])), shape=(len(ids), len(ids)))
    _, lab = connected_components(g, directed=False); P["series"] = [f"{t}:{lab[ix[i]]}" for i in P.id_a]
    P["simbin"] = pd.cut(P.sim, [0.6, 0.7, 0.8, 1.01], labels=["0.60-0.70", "0.70-0.80", ">=0.80"], right=False)
    sz = P[P.kind == "cliff"].groupby("series").size().sort_values(ascending=False)
    P["big"] = (P.series == sz.index[0]) if len(sz) else False
    P["target"] = t; P["loc_i"] = np.arange(len(P))
    # binding-site clusters of the contact profile (per target)
    from sklearn.cluster import KMeans
    allC = (MD <= 5.0).astype(float); okr = ~np.isnan(MD).any(1); labs = np.full(len(MD), -1)
    if okr.sum() >= 10:
        km = KMeans(2, n_init=10, random_state=0).fit(allC[okr]); big = np.bincount(km.labels_).argmax()
        labs[okr] = np.where(km.labels_ == big, 0, 1)
    P["site_a"] = [labs[rowof[i]] for i in P.id_a]; P["site_b"] = [labs[rowof[i]] for i in P.id_b]
    P["switch"] = (P.site_a != P.site_b) & (P.site_a >= 0) & (P.site_b >= 0)
    # primary-screen Z-score of every compound (raw data), partner percentile among the target's inactives
    raw = pd.read_csv(bc.DATA / f"{t}.csv"); raw["id"] = f"{t}_" + raw.CID.astype(str); raw = raw.set_index("id")
    Mi["zscore"] = raw["SD Z-score"].reindex(Mi.index).values
    ina = raw[raw.target_active_v2 == 0]["SD Z-score"].dropna().values; act = raw[raw.target_active_v2 == 1]["SD Z-score"].dropna().values
    zinfo = dict(ina=ina, act=act, hi99=float(np.quantile(ina, 0.99)), act10=float(np.quantile(act, 0.1)), ina_med=float(np.median(ina)), act_med=float(np.median(act)))
    P["partner_z"] = raw["SD Z-score"].reindex(P.id_b.values).values
    ev = pd.read_csv(A / "scores_eval.csv")
    return dict(P=P, Mi=Mi, ma=ma, mb=mb, MD=MD, rowof=rowof, zinfo=zinfo, ev=ev, nres=MD.shape[1], auroc=pd.read_csv(A / "modality_auroc.csv").set_index("modality"),
                n_eval=len(ev), n_act=int(ev.label.sum()))


def load_all(states):
    D = {}
    for t, s in states.items():
        if s != "ready": continue
        try: D[t] = _load_one(t)
        except Exception as e: print(f"[{bc.SHORT[t]}] could not load cliff cache: {type(e).__name__}: {e}")
    return D


def pool(D):
    P = pd.concat([D[t]["P"] for t in D], ignore_index=True)
    Mi = pd.concat([D[t]["Mi"] for t in D])
    return P, Mi


# ---------------------------------------------------------------- statistics (series are the resampling unit; targets are the analysis unit)
def _ind(x, neg=False):
    x = (-x if neg else x).astype(float); return np.where(x > 0, 1.0, np.where(x == 0, 0.5, 0.0))


def acc_ci(df, col, neg=False, B=2000, seed=0, alpha=0.05):
    """Macro-averaged over targets: share of pairs where the signal ranks the active higher (ties 0.5). CI from resampling SERIES within each
    target independently. Returns dict(acc, lo, hi, n_pairs, n_series, n_targets, per_target={t:acc}); with one target this is the plain series bootstrap."""
    per, sums = {}, []
    for t, g in df.groupby("target"):
        x = g[col].values.astype(float); ok = np.isfinite(x)
        if not ok.any(): continue
        ids = g.series.values[ok]; ind = _ind(x[ok], neg); u, inv = np.unique(ids, return_inverse=True)
        sm = np.bincount(inv, weights=ind); n = np.bincount(inv).astype(float)
        idx = np.random.default_rng(seed).integers(0, len(u), (B, len(u))); bs = sm[idx].sum(1) / n[idx].sum(1)
        per[t] = sm.sum() / n.sum(); sums.append((bs, int(n.sum()), len(u)))
    if not per: return dict(acc=np.nan, lo=np.nan, hi=np.nan, n_pairs=0, n_series=0, n_targets=0, per_target={})
    bs = np.mean([s[0] for s in sums], axis=0)
    return dict(acc=float(np.mean(list(per.values()))), lo=float(np.percentile(bs, 100 * alpha / 2)), hi=float(np.percentile(bs, 100 * (1 - alpha / 2))),
                n_pairs=sum(s[1] for s in sums), n_series=sum(s[2] for s in sums), n_targets=len(per), per_target=per)


def rate(x):
    x = x.dropna(); return ((x > 0).sum() + 0.5 * (x == 0).sum()) / len(x) if len(x) else np.nan


def macro_rate(df, col, neg=False):
    """equal-weight mean over targets of the plain share of pairs ranking the active higher"""
    v = [rate(-g[col] if neg else g[col]) for _, g in df.groupby("target")]; v = [a for a in v if np.isfinite(a)]
    return float(np.mean(v)) if v else np.nan


lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))


def auc(pos, neg):
    r = rankdata(np.concatenate([pos, neg])); n1, n2 = len(pos), len(neg)
    return (r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n2)


def calib_metrics(c, k):
    """c, k = deltas (active - inactive) of cliff / conserved pairs for ONE model on ONE scale. Threshold calibrated on the conserved pairs."""
    o = {}
    for sp in (90, 95):
        tau = np.quantile(np.abs(k), sp / 100); o[f"sens@{sp}"] = np.mean(c > tau); o[f"wrongdir@{sp}"] = np.mean(c < -tau)
    o["auc_abs"] = auc(np.abs(c), np.abs(k)); o["auc_signed"] = auc(c, k); o["pairwise_acc"] = np.mean((c > 0) + 0.5 * (c == 0))
    return o


CAL_KEYS = ["pairwise_acc", "sens@90", "sens@95", "wrongdir@90", "auc_abs", "auc_signed"]
MODELS2 = {"No-FT": "base_noft", "head-FT": "ft300"}


def add_logodds(P, Mi):
    for m in MODELS2.values(): P["l_" + m] = lg(Mi[m].reindex(P.id_a).values) - lg(Mi[m].reindex(P.id_b).values)
    return P


def _col(model, scale): return ("d_" if scale == "percentile" else "l_") + MODELS2[model]


def ft_vs_noft(P, B=600, seed=0, min_pairs=5):
    """Per target (series bootstrap, paired over the two models) then macro-average over targets. Returns (long DataFrame:
    subset, scale, metric, noft, ft, diff, lo, hi, n_targets, n_cliff, n_conserved, n_series ; per-target point-estimate table)."""
    rows, pert = [], []
    parts = [("all pairs", lambda d: d), ("without the dominant series", lambda d: d[~d.big]), ("dominant series only", lambda d: d[d.big])]
    for nm, sel in parts:
        pt, bs, used = {}, {}, []
        for t, g in P.groupby("target"):
            d = sel(g).reset_index(drop=True); C, K = d[d.kind == "cliff"], d[d.kind == "conserved"]
            if len(C) < min_pairs or len(K) < min_pairs: continue
            used.append((t, len(C), len(K), d.series.nunique()))
            pt[t] = {(m, sc): calib_metrics(C[_col(m, sc)].values, K[_col(m, sc)].values) for m in MODELS2 for sc in ("percentile", "log-odds")}
            if nm == "dominant series only": continue
            grp = list(d.groupby("series").indices.values()); rng = np.random.default_rng(seed); res = []
            for _ in range(B):
                idx = np.concatenate([grp[i] for i in rng.integers(0, len(grp), len(grp))]); dd = d.iloc[idx]
                if (dd.kind == "cliff").sum() < min_pairs or (dd.kind == "conserved").sum() < min_pairs: continue
                res.append({(m, sc): calib_metrics(dd.loc[dd.kind == "cliff", _col(m, sc)].values, dd.loc[dd.kind == "conserved", _col(m, sc)].values) for m in MODELS2 for sc in ("percentile", "log-odds")})
            bs[t] = res
        if not pt: continue
        for sc in ("percentile", "log-odds"):
            for key in CAL_KEYS:
                a = np.mean([pt[t][("No-FT", sc)][key] for t in pt]); b = np.mean([pt[t][("head-FT", sc)][key] for t in pt]); lo = hi = np.nan
                if bs:
                    n = min(len(v) for v in bs.values())
                    if n > 50:
                        dd = np.mean([[r[("head-FT", sc)][key] - r[("No-FT", sc)][key] for r in bs[t][:n]] for t in bs], axis=0); lo, hi = np.percentile(dd, [2.5, 97.5])
                rows.append(dict(subset=nm, scale=sc, metric=key, noft=a, ft=b, diff=b - a, lo=lo, hi=hi, n_targets=len(pt),
                                 n_cliff=sum(u[1] for u in used), n_conserved=sum(u[2] for u in used), n_series=sum(u[3] for u in used)))
        if nm == "all pairs":
            for t in pt:
                for sc in ("percentile", "log-odds"):
                    r = dict(target=t, scale=sc)
                    for key in CAL_KEYS: r["noft_" + key] = pt[t][("No-FT", sc)][key]; r["ft_" + key] = pt[t][("head-FT", sc)][key]
                    pert.append(r)
    return pd.DataFrame(rows), pd.DataFrame(pert)


def prop_auroc(t, D_t):
    """Library-wide AUROC of MW / heavy atoms / logP for the activity label (cached in results/analysis/<t>/cliffs/prop_auroc.json)."""
    import json
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors, Crippen
    from sklearn.metrics import roc_auc_score
    RDLogger.DisableLog("rdApp.*")
    f = pc.cache_dir(t) / "prop_auroc.json"
    if f.exists() and os.path.getmtime(f) >= os.path.getmtime(pc.cache_dir(t) / "scores_eval.csv"):
        return json.load(open(f))
    ev = D_t["ev"][["smiles", "label"]].dropna(); mols = [Chem.MolFromSmiles(x) for x in ev.smiles]; y = ev.label.values.astype(int); out = {}
    for nm, fn in [("molecular weight", Descriptors.MolWt), ("heavy atoms", lambda m: m.GetNumHeavyAtoms()), ("logP", Crippen.MolLogP)]:
        v = np.array([fn(m) if m is not None else np.nan for m in mols]); k = np.isfinite(v); out[nm] = float(roc_auc_score(y[k], v[k]))
    try: json.dump(out, open(f, "w"))
    except Exception: pass
    return out

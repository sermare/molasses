"""Helpers for notebooks/02_uncertainty.ipynb (target-generic uncertainty analysis of the Boltz-2 head-FT runs).
Heavy per-target results (bootstraps) are cached in results/analysis/<target>/uncertainty_cache.pkl."""
import pickle, numpy as np, pandas as pd
from bft_common import *
from sklearn.metrics import average_precision_score

B_RR, B_AP = 1000, 500
BANDS = [0, .01, .02, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0]     # rank-fraction bands (top of ranking first)
BAND_LAB = [f"{100*a:g}-{100*b:g}%" for a, b in zip(BANDS[:-1], BANDS[1:])]
# uncertainty measures: high value = more uncertain.  (label, score used for matching)
MEAS = {"sd": "cross-seed SD (FT N=300)", "dis_ft": "two-head disagreement (FT N=300)", "dis0": "two-head disagreement (No-FT)",
        "iptm": "1 - ligand ipTM (pose)", "pde": "complex PDE (pose)"}
MCOL = {"sd": "#2a78d6", "dis_ft": "#eb6834", "dis0": "#1baf7a", "iptm": "#9467bd", "pde": "#8c564b"}
TCOL = dict(zip(TARGETS, ["#2a78d6", "#eb6834", "#1baf7a", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22"]))


def pending_reason(t):
    if t in done_targets("scores"): return None
    k = len(scored_arms(t))
    if not (RUNS / t / "ft_inputs_full/eval_ids.txt").exists(): return "ft_inputs / eval set not built yet"
    return f"only {k}/16 eval arms scored (need No-FT + 15 head-FT arms)"


def load(t):
    S = scores(t)
    if S is None: return None
    P = {N: S[[f"ft{N}_p{s}" for s in SEEDS]].values for N in BUDGETS}
    DS = {N: S[[f"ft{N}_dis{s}" for s in SEEDS]].values for N in BUDGETS}
    D = dict(t=t, S=S, y=S.label.values.astype(int), p0=S.noft_p.values, dis0=S.noft_dis.values, P=P, DS=DS,
             pm=P[300].mean(1), sd=P[300].std(1, ddof=1), dis_ft=DS[300].mean(1))
    q = RUNS / t / "qc.csv"
    cid = S.id.str.split("_").str[-1]
    D["iptm"] = np.full(len(S), np.nan); D["pde"] = np.full(len(S), np.nan)
    if q.exists():
        qc = pd.read_csv(q); qc["CID"] = qc.CID.astype(str); qc = qc.drop_duplicates("CID").set_index("CID")
        D["iptm"] = -qc.ligand_iptm.reindex(cid).values          # uncertainty = 1 - ipTM (shifted constant irrelevant)
        D["pde"] = qc.complex_pde.reindex(cid).values
    D["has_pose"] = bool(np.isfinite(D["iptm"]).sum() > 1000)
    return D


def rank_band(score):
    """Rank-fraction band index (0 = top 1%) of each compound for a score (higher = better)."""
    n = len(score); r = np.empty(n); r[np.argsort(-score, kind="stable")] = (np.arange(n) + 0.5) / n
    return np.searchsorted(np.array(BANDS[1:-1]), r, side="right")


def conf_group(u, band):
    """1 = 'confident' (uncertainty at or below its within-band median), 0 = 'uncertain'."""
    med = pd.Series(u).groupby(band).transform("median").values
    return (u <= med).astype(int)


def mh_rr(w, y, band, g):
    nb = len(BANDS) - 1; idx = band * 2 + g
    n = np.bincount(idx, weights=w, minlength=2 * nb).reshape(nb, 2); a = np.bincount(idx, weights=w * y, minlength=2 * nb).reshape(nb, 2)
    N = np.maximum(n.sum(1), 1e-12); num = (a[:, 1] * n[:, 0] / N).sum(); den = (a[:, 0] * n[:, 1] / N).sum()
    return num / den if den > 0 and num > 0 else np.nan


def ap_w(score, y, counts):
    """Weighted AP (bootstrap counts (B, n)); ties ignored (probabilities are continuous)."""
    o = np.argsort(-score, kind="stable"); ys = y[o].astype(float); out = []
    for i in range(0, counts.shape[0], 100):
        W = counts[i:i + 100][:, o].astype(float); cw = np.cumsum(W, 1); cy = np.cumsum(W * ys, 1)
        out.append(((cy / np.maximum(cw, 1)) * W * ys).sum(1) / np.maximum((W * ys).sum(1), 1e-12))
    return np.concatenate(out)


def ece_w(w, y, p, bins, nb=10):
    n = np.bincount(bins, weights=w, minlength=nb); a = np.bincount(bins, weights=w * y, minlength=nb); s = np.bincount(bins, weights=w * p, minlength=nb)
    return np.abs(a - s).sum() / max(w.sum(), 1e-12)


def eq_bins(p, nb=10):
    return np.searchsorted(np.quantile(p, np.linspace(0, 1, nb + 1)[1:-1]), p, side="right")


def topk(score, k): return np.argpartition(-score, k - 1)[:k]


def jaccard_mat(M, k):
    sets = [set(topk(M[:, s], k)) for s in range(M.shape[1])]
    J = np.array([[len(a & b) / len(a | b) for b in sets] for a in sets]); return J


def _prec_rec(w, y, m): return (w * y * m).sum() / max((w * m).sum(), 1e-12), (w * y * m).sum() / max((w * y).sum(), 1e-12)


def analyze(t, force=False):
    """Everything heavy for one target; cached. Returns None (with pending reason) if the target is not fully scored."""
    D = load(t)
    if D is None: return None
    y = D["y"]; n = len(y); k = max(1, int(round(0.01 * n)))
    sig = (n, round(float(D["pm"].sum()), 6), round(float(np.nansum(D["iptm"])), 3))
    cf = AN / t / "uncertainty_cache.pkl"
    if cf.exists() and not force:
        try:
            R = pickle.load(open(cf, "rb"))
            if R.get("sig") == sig: return R
        except Exception: pass
    rng = np.random.default_rng(12345)
    counts = np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(B_RR)]).astype(np.uint8)
    w1 = np.ones(n); R = dict(sig=sig, n=n, k=k, n_act=int(y.sum()), rate=float(y.mean()))
    # ---- (1) matched-strata value of confidence measures ----
    R["rr"] = {}
    for m in MEAS:
        u = D[m]; ok = np.isfinite(u)
        if ok.sum() < 1000: R["rr"][m] = None; continue
        match = D["p0"] if m == "dis0" else D["pm"]
        yy, uu, ms = y[ok], u[ok], match[ok]
        band = rank_band(ms); g = conf_group(uu, band)
        pt = mh_rr(np.ones(ok.sum()), yy, band, g)
        bs = np.array([mh_rr(counts[b][ok].astype(float), yy, band, g) for b in range(B_RR)])
        nb = len(BANDS) - 1; idx = band * 2 + g
        nn = np.bincount(idx, minlength=2 * nb).reshape(nb, 2); aa = np.bincount(idx, weights=yy, minlength=2 * nb).reshape(nb, 2)
        R["rr"][m] = dict(rr=pt, boot=bs, n_used=int(ok.sum()), n_hi=nn[:, 1], n_lo=nn[:, 0], a_conf=aa[:, 1], a_unc=aa[:, 0])
    # ---- risk-coverage among the top-1% picks ----
    picks = topk(D["pm"], k); R["riskcov"] = {}
    for m in MEAS:
        u = D[m][picks]; ok = np.isfinite(u)
        if ok.sum() < 100: continue
        o = np.argsort(u[ok]); yy = y[picks][ok][o]
        R["riskcov"][m] = dict(cov=(np.arange(len(yy)) + 1) / len(yy), prec=np.cumsum(yy) / (np.arange(len(yy)) + 1), base=float(y[picks][ok].mean()))
        lo, hi = o[: len(o) // 2], o[len(o) // 2:]
        R["riskcov"][m].update(prec_conf=float(y[picks][ok][lo].mean()), prec_unc=float(y[picks][ok][hi].mean()), n_half=len(lo))
    # ---- (2) calibration ----
    cal = {}
    for name, p in [("No-FT", D["p0"]), ("head-FT N=300 (seed mean)", D["pm"])]:
        b = eq_bins(p); cal[name] = dict(pm=np.bincount(b, weights=p, minlength=10) / np.bincount(b, minlength=10),
                                          rate=np.bincount(b, weights=y, minlength=10) / np.bincount(b, minlength=10),
                                          ece=ece_w(w1, y, p, b), ece_boot=np.array([ece_w(counts[i].astype(float), y, p, b) for i in range(B_RR)]),
                                          brier=float(np.mean((p - y) ** 2)), mean_p=float(p.mean()))
    cal["seeds300"] = [ece_w(w1, y, D["P"][300][:, s], eq_bins(D["P"][300][:, s])) for s in SEEDS]
    cal["brier_base"] = float(np.mean((y.mean() - y) ** 2))
    R["cal"] = cal
    # ---- (3) seed agreement on the top-1% ----
    R["jac300"] = jaccard_mat(D["P"][300], k); R["jac"] = {}
    for N in BUDGETS:
        J = jaccard_mat(D["P"][N], k); R["jac"][N] = J[np.triu_indices(5, 1)]
    mem = np.zeros((n, 5), bool)
    for s in SEEDS: mem[topk(D["P"][300][:, s], k), s] = True
    votes = mem.sum(1); R["votes"] = votes
    R["vote_tab"] = pd.DataFrame([dict(votes=v, n=int((votes == v).sum()), actives=int(y[votes == v].sum())) for v in range(1, 6)]).set_index("votes")
    # ---- (4) precision / recall of vote-defined subsets ----
    u5 = int((votes == 5).sum()); top_u = np.zeros(n, bool); top_u[topk(D["pm"], max(u5, 1))] = True
    top_k = np.zeros(n, bool); top_k[picks] = True
    sets = {"unanimous (5/5)": votes == 5, ">=4 seeds": votes >= 4, ">=3 seeds": votes >= 3, ">=2 seeds": votes >= 2, "union (>=1)": votes >= 1,
            "only 1 seed": votes == 1, "seed-mean top-1%": top_k, "seed-mean top-u (size of 5/5)": top_u}
    ss = {}
    for nm, msk in sets.items():
        if msk.sum() == 0: ss[nm] = None; continue
        p, r = _prec_rec(w1, y, msk)
        bs = np.array([_prec_rec(counts[i].astype(float), y, msk) for i in range(B_RR)])
        ss[nm] = dict(size=int(msk.sum()), prec=p, rec=r, prec_ci=np.nanpercentile(bs[:, 0], [2.5, 97.5]), rec_ci=np.nanpercentile(bs[:, 1], [2.5, 97.5]))
    if u5 > 0:       # unanimous vs same-size top of the seed mean (paired, conditional on the two fixed sets)
        d = np.array([_prec_rec(counts[i].astype(float), y, sets["unanimous (5/5)"])[0] - _prec_rec(counts[i].astype(float), y, top_u)[0] for i in range(B_RR)])
        ss["_diff_unan_vs_topu"] = dict(d=ss["unanimous (5/5)"]["prec"] - ss["seed-mean top-u (size of 5/5)"]["prec"], ci=np.nanpercentile(d, [2.5, 97.5]))
    R["sets"] = ss
    R["seed_topk_prec"] = [float(y[topk(D["P"][300][:, s], k)].mean()) for s in SEEDS]
    R["pr_curve"] = pd.DataFrame([dict(m=m, prec=y[topk(D["pm"], m)].mean(), rec=y[topk(D["pm"], m)].sum() / y.sum()) for m in np.unique(np.linspace(max(1, k // 2), 4 * k, 25).astype(int))])
    # ---- (5) uncertainty of the ranking metric ----
    ap = dict(noft=average_precision_score(y, D["p0"]), noft_boot=ap_w(D["p0"], y, counts[:B_AP]), ft={})
    for N in BUDGETS:
        pts = [average_precision_score(y, D["P"][N][:, s]) for s in SEEDS]
        sb = np.stack([ap_w(D["P"][N][:, s], y, counts[:B_AP]) for s in SEEDS])
        ap["ft"][N] = dict(seeds=np.array(pts), mean=float(np.mean(pts)), sd=float(np.std(pts, ddof=1)), boot_mean=sb.mean(0), boot_seed0=sb[0],
                           ens=average_precision_score(y, D["P"][N].mean(1)))
    R["ap"] = ap
    # ---- budgets: uncertainty magnitude ----
    R["budget"] = pd.DataFrame([dict(N=N, sd_all=float(D["P"][N].std(1, ddof=1).mean()), dis_all=float(D["DS"][N].mean()),
                                     sd_top=float(D["P"][N].std(1, ddof=1)[topk(D["P"][N].mean(1), k)].mean()),
                                     dis_top=float(D["DS"][N].mean(1)[topk(D["P"][N].mean(1), k)].mean()),
                                     jac=float(np.mean(R["jac"][N]))) for N in BUDGETS]).set_index("N")
    try: pickle.dump(R, open(cf, "wb"))
    except Exception as e: print("cache not written:", e)
    return R


def pooled(vals):
    """Across-target summary of per-target values: (mean, lo, hi, n) with a t-interval when n>=3, else CI None."""
    from scipy import stats
    v = np.array([x for x in vals if np.isfinite(x)]); n = len(v)
    if n == 0: return (np.nan, None, None, 0)
    if n < 3: return (v.mean(), None, None, n)
    se = v.std(ddof=1) / np.sqrt(n); h = stats.t.ppf(0.975, n - 1) * se
    return (v.mean(), v.mean() - h, v.mean() + h, n)


def descriptors(D, n_rand=9000, seed=0):
    """RDKit properties for a random sample (+ all top-1% picks) and the within-band rank of cross-seed SD."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors
    RDLogger.DisableLog("rdApp.*")
    n = len(D["y"]); rng = np.random.default_rng(seed)
    idx = np.unique(np.concatenate([rng.choice(n, min(n_rand, n), replace=False), topk(D["pm"], max(1, n // 100))]))
    rows = []
    for i in idx:
        m = Chem.MolFromSmiles(D["S"].smiles.iat[i]) if isinstance(D["S"].smiles.iat[i], str) else None
        if m is None: continue
        rows.append(dict(i=i, MW=Descriptors.MolWt(m), heavy=m.GetNumHeavyAtoms(), logP=Crippen.MolLogP(m), TPSA=rdMolDescriptors.CalcTPSA(m),
                         rot=rdMolDescriptors.CalcNumRotatableBonds(m), HBD=rdMolDescriptors.CalcNumHBD(m), HBA=rdMolDescriptors.CalcNumHBA(m),
                         arom=rdMolDescriptors.CalcNumAromaticRings(m), fsp3=rdMolDescriptors.CalcFractionCSP3(m)))
    df = pd.DataFrame(rows).set_index("i")
    band = rank_band(D["pm"])
    for m in ("sd", "dis_ft"):
        df[m + "_rank"] = pd.Series(D[m]).groupby(band).rank(pct=True).values[df.index]
    return df

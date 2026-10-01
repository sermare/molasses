#!/usr/bin/env python3
"""03_ft_vs_noft_reranking.ipynb - self-filling, all targets: library-level No-FT vs head-FT (confusion at matched budgets, trade-off
curves, matched-sensitivity false positives), re-ranking (promoted/lost actives, dataset-score percentiles, similarity to the 300 training
actives) and a cross-target summary. Matplotlib + RDKit; black text, no bold, NO HTML."""
import nbformat as nbf
from pathlib import Path
NB = Path("/global/scratch/users/sergiomar10/boltzaff/notebooks/03_ft_vs_noft_reranking.ipynb")
def md(s): return nbf.v4.new_markdown_cell(s)
def code(s): return nbf.v4.new_code_cell(s)

cells = [
md("""# 03 - Head-FT against No-FT at the library level, and what fine-tuning re-ranks (all targets)

This notebook is self-filling: every re-run processes each target whose No-FT and all 15 head-FT arms (N = 40, 100, 300 x 5 seeds) are scored, prints a `pending:` line for the others and draws them as empty (paper-only) panels.

Questions
1. At matched budgets (top 0.5%, 1%, 2%, 5% of the library) and at the default probability threshold (p > 0.5): how many actives does each model find, how many false positives does it flag, how many actives does it miss?
2. Does fine-tuning discard true negatives better but miss more actives? Answered at a matched number of flagged compounds, at p > 0.5, and at matched sensitivity.
3. Which compounds move after fine-tuning: actives rescued or lost relative to No-FT, and what the rescued actives look like (percentile under the dataset's other scores, ECFP4 similarity to the 300 training compounds).
4. Per-target summary and a cross-target panel (targets are the unit; sign test).

Conventions
- The ground truth is only the binary label (active / inactive). No potencies exist and none are invented.
- `scores(t)` covers the evaluation set, which already excludes the 300 training compounds used for fine-tuning (checked below: zero overlap). Nothing here scores a training compound. The evaluation set can still contain chemical analogs of the training actives (section 3 quantifies this).
- head-FT = Boltz-2 affinity-head fine-tuning on the top-N compounds of the training split; 'head-FT seed-mean' = mean probability over the 5 seeds (an ensemble). Seed spread is shown as mean and SD of the 5 single-seed runs.
- 'Paper' values (orange diamonds / columns) are fixed values from the paper's trainer_control table (EF1% converted to a count of actives in the top 1%; AP). The paper reports one value per target and budget, no seed spread.
- Intervals: paired stratified bootstrap over compounds within a target (actives and inactives resampled separately, 300 resamples, the ranking recomputed on each resample). They capture the finite size of the evaluation set and not the choice of training compounds; the seed SD is shown separately.
- Many comparisons are made (targets x cutoffs x budgets). No multiplicity correction is applied to the per-target intervals; for the cross-target claims a Bonferroni factor is stated where several tests are reported."""),

code(r"""import sys, textwrap, numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from IPython.display import display, Markdown
ROOT = Path("/global/scratch/users/sergiomar10/boltzaff"); sys.path.insert(0, str(ROOT / "pipeline_local"))
from bft_common import *
from bft_rerank import *
style()
GREY, LB, MB = "#666666", "#9ecae1", "#5b9bd5"; ACOL = {"No-FT": GREY, "FT N=40": LB, "FT N=100": MB, "FT N=300": OUR}
D, PEND = {}, {}
for t in TARGETS:
    S, msg = load(t)
    if S is None:
        PEND[t] = msg; print(f"{SHORT[t]:>7}: {msg}"); continue
    y = S.label.values.astype(int); ens, per = score_sets(S)
    D[t] = dict(S=S, y=y, N=len(S), P=int(y.sum()), ens=ens, per=per)
    tr = train_ids(t, 300); ov = len(set(tr or []) & set(S.id))
    print(f"{SHORT[t]:>7}: filled. eval set {len(S):,} compounds, {int(y.sum())} active (rate {100*y.mean():.2f}%); training compounds found in eval set: {ov}")
print(f"\n{len(D)} of {len(TARGETS)} targets filled: {[SHORT[t] for t in D]}")

def grid(nrow=2, ncol=4, fs=(19, 8.2)):
    fig, axs = plt.subplots(nrow, ncol, figsize=fs); return fig, dict(zip(TARGETS, np.ravel(axs)))
def placeholder(ax, t, extra=""):
    ax.cla(); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    ax.text(0.5, 0.5, SHORT[t] + "\n" + textwrap.fill(PEND.get(t, ""), 30) + extra, ha="center", va="center", transform=ax.transAxes, fontsize=8)
def paper_row(t, N=None):
    try:
        r = PAPER_BASE.loc[t] if N is None else PAPER_FT.loc[(t, N)]
    except KeyError:
        return None
    return r
def paper_tp(t, N=None, frac=0.01):
    r = paper_row(t, N)
    return np.nan if r is None else r.ef_1pct * r.n_active * kof(frac, r.n_eval) / r.n_eval    # EF1% = (TP/k)/(n_active/n_eval)
def msd(v): return f"{np.mean(v):.1f}" if len(v) < 2 else f"{np.mean(v):.1f} +/- {np.std(v, ddof=1):.1f}" """),

md("""## 1. Library-level confusion at matched budgets

For each target: the evaluation set is ranked by each model; the top k compounds (k = 0.5%, 1%, 2%, 5% of the library) are 'flagged'. TP = actives found, FP = false positives, FN = actives missed. For head-FT the tables give the 5-seed ensemble (mean probability) and, for TP, the mean +/- SD of the five single-seed runs."""),
code(r"""rows = []
for t, d in D.items():
    for frac in FRACS:
        k = kof(frac, d["N"])
        for arm in d["ens"]:
            c = conf_k(d["ens"][arm], d["y"], k); tps = [conf_k(s, d["y"], k)["TP"] for s in d["per"][arm]]
            rows.append(dict(target=SHORT[t], cutoff=f"top {100*frac:g}%", k=k, arm=arm, TP=c["TP"], TP_seed_mean=np.mean(tps), TP_seed_sd=np.std(tps, ddof=1) if len(tps) > 1 else np.nan,
                             FP=c["FP"], FN=c["FN"], sens=c["sens"], spec=c["spec"], prec=c["prec"]))
T1 = pd.DataFrame(rows)
for t in D:
    print(f"target {SHORT[t]}: {D[t]['N']:,} compounds, {D[t]['P']} actives. TP/FP/FN of the 5-seed ensemble; TP_seed = mean +/- SD of single seeds")
    x = T1[T1.target == SHORT[t]].copy(); x["TP_seed"] = [("" if a == "No-FT" else f"{m:.1f} +/- {s:.1f}") for a, m, s in zip(x.arm, x.TP_seed_mean, x.TP_seed_sd)]
    display(x.set_index(["cutoff", "k", "arm"])[["TP", "TP_seed", "FP", "FN", "sens", "spec", "prec"]].round(4))
for t in PEND: print(f"{SHORT[t]}: {PEND[t]}")"""),

md("### Actives found in the top 1%, with the paper value\nBars: mean of the five single-seed runs, error bar: SD across seeds; black dot: 5-seed ensemble. Orange diamond: paper value (EF1% converted to a count with the same k). Paper-only panels for targets not yet scored."),
code(r"""fig, AX = grid(); arms = list(ACOL)
for t, ax in AX.items():
    if t not in D:
        placeholder(ax, t, f"\npaper top-1% actives (from EF1%): No-FT {paper_tp(t):.0f}, FT N=300 {paper_tp(t, 300):.0f}")
        ax.set_title(f"{SHORT[t]}: pending", fontsize=10); continue
    d = D[t]; k = kof(0.01, d["N"])
    for i, a in enumerate(arms):
        tps = [conf_k(s, d["y"], k)["TP"] for s in d["per"][a]]; e = conf_k(d["ens"][a], d["y"], k)["TP"]
        ax.bar(i, np.mean(tps), color=ACOL[a], width=0.6, yerr=(np.std(tps, ddof=1) if len(tps) > 1 else 0.0), capsize=3, ecolor=BLK, label="ours: single-seed mean +/- SD" if i == 0 else None)
        ax.scatter(i, e, color=BLK, s=16, zorder=4, label="ours: 5-seed ensemble" if i == 0 else None)
        pt = paper_tp(t, None if a == "No-FT" else int(a.split("=")[1])); ax.scatter(i, pt, marker="D", color=THEIR, s=40, zorder=5, label="paper" if i == 0 else None)
    ax.set_xticks(range(4)); ax.set_xticklabels(["No-FT", "N=40", "N=100", "N=300"], fontsize=8); ax.set_ylabel("actives in top 1%"); ax.set_title(f"{SHORT[t]}: top 1% = {k} of {d['N']:,}, {d['P']} actives", fontsize=9)
list(AX.values())[0].legend(fontsize=7, loc="upper left")
fig.suptitle("Actives found in the top 1% of the library: No-FT and head-FT (N = 40, 100, 300) against the paper", y=1.0); plt.tight_layout(); plt.show()"""),

md("### Fixed threshold: p > 0.5\nThe default decision rule. Fine-tuning shifts probabilities down, so p > 0.5 flags a different number of compounds under each model; counts are therefore not at a matched budget. Ensemble = 5-seed mean probability; single-seed columns give mean +/- SD."),
code(r"""rows = []
for t, d in D.items():
    for arm in d["ens"]:
        c = conf_thr(d["ens"][arm], d["y"]); cs = [conf_thr(s, d["y"]) for s in d["per"][arm]]
        rows.append(dict(target=SHORT[t], arm=arm, flagged=c["flagged"], TP=c["TP"], FP=c["FP"], FN=c["FN"], sens=c["sens"], spec=c["spec"], prec=c["prec"],
                         flagged_seed=msd([x["flagged"] for x in cs]), TP_seed=msd([x["TP"] for x in cs]), FP_seed=msd([x["FP"] for x in cs])))
T1b = pd.DataFrame(rows)
for t in D: print(f"target {SHORT[t]}"); display(T1b[T1b.target == SHORT[t]].set_index("arm").drop(columns="target").round(4))
for t in PEND: print(f"{SHORT[t]}: {PEND[t]}")"""),

md("### Specificity-sensitivity trade-off curves (ROC zoomed on the low false-positive region, log x)\nx: false-positive rate = 1 - specificity (log scale); y: sensitivity. Grey: No-FT. Blue thick: head-FT N=300 5-seed ensemble; thin blue: the five single seeds. The region left of x = 0.01 is the part that matters for a screen."),
code(r"""def curve(score, y):
    o = order(score); cs = np.cumsum(y[o]); k = np.arange(1, len(y) + 1); P = y.sum(); neg = len(y) - P
    return (k - cs) / neg, cs / P, cs / k
fig, AX = grid()
for t, ax in AX.items():
    if t not in D: placeholder(ax, t); ax.set_title(f"{SHORT[t]}: pending", fontsize=10); continue
    d = D[t]; ks = np.unique(np.geomspace(20, d["N"], 600).astype(int)) - 1
    for s in d["per"]["FT N=300"]: f, r, _ = curve(s, d["y"]); ax.plot(f[ks], r[ks], color=OUR, lw=0.7, alpha=0.4)
    for a, lw in [("No-FT", 2), ("FT N=300", 2)]:
        f, r, _ = curve(d["ens"][a], d["y"]); ax.plot(f[ks], r[ks], color=ACOL[a], lw=lw, label=a if a == "No-FT" else "head-FT N=300 (ensemble; thin = seeds)")
    ax.set_xscale("log"); ax.set_xlim(2e-4, 0.3); ax.set_ylim(0, None); ax.set_xlabel("false-positive rate (1 - specificity), log"); ax.set_ylabel("sensitivity"); ax.set_title(f"{SHORT[t]}: ROC, low-FPR region", fontsize=10)
list(AX.values())[0].legend(fontsize=7, loc="upper left"); fig.suptitle("Sensitivity against false-positive rate, No-FT vs head-FT (N=300)", y=1.0); plt.tight_layout(); plt.show()"""),
md("### Precision against sensitivity (log x)\nx: sensitivity (log); y: precision (share of flagged compounds that are active). Dashed: active rate = a random ranking."),
code(r"""fig, AX = grid()
for t, ax in AX.items():
    if t not in D: placeholder(ax, t); ax.set_title(f"{SHORT[t]}: pending", fontsize=10); continue
    d = D[t]; ks = np.unique(np.geomspace(5, d["N"], 600).astype(int)) - 1
    for s in d["per"]["FT N=300"]: f, r, p = curve(s, d["y"]); ax.plot(r[ks], p[ks], color=OUR, lw=0.7, alpha=0.4)
    for a in ["No-FT", "FT N=300"]:
        f, r, p = curve(d["ens"][a], d["y"]); ax.plot(r[ks], p[ks], color=ACOL[a], lw=2, label=a if a == "No-FT" else "head-FT N=300 (ensemble; thin = seeds)")
    ax.axhline(d["y"].mean(), color=BLK, ls="--", lw=0.8); ax.set_xscale("log"); ax.set_xlim(0.005, 1); ax.set_xlabel("sensitivity (recall), log"); ax.set_ylabel("precision"); ax.set_title(f"{SHORT[t]}: precision-recall", fontsize=10)
list(AX.values())[0].legend(fontsize=7, loc="upper right"); fig.suptitle("Precision against sensitivity, No-FT vs head-FT (N=300)", y=1.0); plt.tight_layout(); plt.show()"""),

md("""## 2. Does fine-tuning discard true negatives better but miss more actives?

Two things can be true at once, so both are reported per target and never merged.
- Matched number flagged (top 1%): FP = k - TP, so finding more actives and having fewer false positives are the same statement. The interval is the paired bootstrap of the difference (head-FT N=300 ensemble minus No-FT).
- Default threshold p > 0.5: the number flagged differs between models, so a real trade-off can appear (fewer false positives and more missed actives, or the opposite).
- Matched sensitivity: how many false positives must be accepted to reach the same number of actives (sensitivity 25%, 50%, and No-FT's own top-1% sensitivity). This is the specificity view at equal recall."""),
code(r"""R2 = {}; LV = (0.25, 0.5)
for t, d in D.items():
    try:
        y, a, b = d["y"], d["ens"]["No-FT"], d["ens"]["FT N=300"]; k = kof(0.01, d["N"]); ca, cb = conf_k(a, y, k), conf_k(b, y, k)
        bt = boot_diff(a, b, y, frac=0.01, sens_levels=LV, B=300, seed=1)
        tps = [conf_k(s, y, k)["TP"] for s in d["per"]["FT N=300"]]
        r = dict(k=k, TP_a=ca["TP"], TP_b=cb["TP"], dTP=cb["TP"] - ca["TP"], dTP_ci=ci(bt["dTP"]), seed_TP=tps, FP_a=ca["FP"], FP_b=cb["FP"], spec_a=ca["spec"], spec_b=cb["spec"])
        ta, tb = conf_thr(a, y), conf_thr(b, y); r.update(thr_a=ta, thr_b=tb, thr_seed=[conf_thr(s, y) for s in d["per"]["FT N=300"]])
        ms = []
        for lab, tpt, bk in [(f"sensitivity {int(100*l)}%", int(np.ceil(l * d["P"])), f"dFP@{l}") for l in LV] + [("No-FT top-1% sensitivity", ca["TP"], None)]:
            fa, ka = fp_at_sens(a, y, tpt); fb, kb = fp_at_sens(b, y, tpt)
            ms.append(dict(level=lab, TP=tpt, FP_noft=fa, FP_ft=fb, dFP=fb - fa, dFP_ci=ci(bt[bk]) if bk else (np.nan, np.nan), spec_noft=1 - fa / (d["N"] - d["P"]), spec_ft=1 - fb / (d["N"] - d["P"])))
        r["matched"] = pd.DataFrame(ms); R2[t] = r
    except Exception as e:
        print(f"{SHORT[t]}: section 2 failed ({type(e).__name__}: {e}); skipped")
def verdict(ta, tb):
    fewer_fp, more_fn = tb["FP"] < ta["FP"], tb["FN"] > ta["FN"]
    if fewer_fp and more_fn: return "trade-off: fewer false positives but more missed actives"
    if fewer_fp and not more_fn: return "better on both: fewer false positives and no more missed actives"
    if (not fewer_fp) and more_fn: return "worse on both: more false positives and more missed actives"
    return "more false positives but fewer missed actives (head-FT is the more permissive rule)"
for t, r in R2.items():
    print(f"=== target {SHORT[t]} (k = {r['k']}) ===")
    print(f"matched top 1%: actives No-FT {r['TP_a']} vs head-FT {r['TP_b']} (difference {r['dTP']:+d}, 95% bootstrap CI [{r['dTP_ci'][0]:+.0f}, {r['dTP_ci'][1]:+.0f}]; seed range {min(r['seed_TP'])}-{max(r['seed_TP'])}); "
          f"false positives {r['FP_a']} vs {r['FP_b']}; specificity {100*r['spec_a']:.2f}% vs {100*r['spec_b']:.2f}%")
    ta, tb = r["thr_a"], r["thr_b"]; sf = r["thr_seed"]
    print(f"p > 0.5: No-FT flags {ta['flagged']}, finds {ta['TP']}, misses {ta['FN']}, FP {ta['FP']}, specificity {100*ta['spec']:.2f}% | head-FT ensemble flags {tb['flagged']}, finds {tb['TP']}, misses {tb['FN']}, FP {tb['FP']}, specificity {100*tb['spec']:.2f}% "
          f"(single seeds: flagged {msd([x['flagged'] for x in sf])}, found {msd([x['TP'] for x in sf])}, FP {msd([x['FP'] for x in sf])})  -> {verdict(ta, tb)}")
    m = r["matched"].copy(); m["dFP 95% CI"] = [("" if np.isnan(c[0]) else f"[{c[0]:+.0f}, {c[1]:+.0f}]") for c in m.dFP_ci]; display(m.drop(columns="dFP_ci").set_index("level").round(4))
for t in PEND: print(f"{SHORT[t]}: {PEND[t]}")"""),
md("### Operating points of the two models\nEach panel places No-FT and head-FT (N=300 ensemble) in the plane of false positives (log x) against actives found, at the matched top-1% cutoff (circles) and at p > 0.5 (squares). Up and to the left is better."),
code(r"""fig, AX = grid()
for t, ax in AX.items():
    if t not in R2: placeholder(ax, t); ax.set_title(f"{SHORT[t]}: pending", fontsize=10); continue
    r = R2[t]
    for nm, c, ta in [("No-FT", GREY, r["thr_a"]), ("head-FT N=300", OUR, r["thr_b"])]:
        tp, fp = (r["TP_a"], r["FP_a"]) if nm == "No-FT" else (r["TP_b"], r["FP_b"])
        ax.scatter(max(fp, 1), tp, color=c, s=70, marker="o", label=f"{nm}, top 1%"); ax.scatter(max(ta["FP"], 1), ta["TP"], color=c, s=70, marker="s", label=f"{nm}, p > 0.5")
    ax.set_xscale("log"); ax.set_xlabel("false positives flagged (log)"); ax.set_ylabel("actives found"); ax.set_title(f"{SHORT[t]}: {D[t]['P']} actives in the library", fontsize=10)
list(AX.values())[0].legend(fontsize=7); fig.suptitle("Operating points: top-1% cutoff (circles) and default threshold p > 0.5 (squares)", y=1.0); plt.tight_layout(); plt.show()"""),

md("""## 3. Re-ranking: which compounds move, which actives are rescued or lost

Each compound has a rank under No-FT and under head-FT (N=300, 5-seed ensemble); 1 = best. With the top-1% cut: promoted actives are actives in head-FT's top 1% but not No-FT's, lost actives the reverse, kept actives are in both; demoted false positives are inactives that No-FT flagged and head-FT does not, new false positives the reverse. The shift is log2(No-FT rank / head-FT rank), positive = moved up."""),
code(r"""RR = {}
for t, d in D.items():
    try:
        S = d["S"].copy(); k = kof(0.01, d["N"]); S["rank_noft"] = S.noft_p.rank(ascending=False, method="first").astype(int)
        S["ft"] = d["ens"]["FT N=300"]; S["rank_ft"] = S.ft.rank(ascending=False, method="first").astype(int)
        S["in_noft"], S["in_ft"] = S.rank_noft <= k, S.rank_ft <= k; S["rshift"] = np.log2(S.rank_noft / S.rank_ft); a1 = S.label == 1; a0 = S.label == 0
        G = {"promoted actives": a1 & S.in_ft & ~S.in_noft, "lost actives": a1 & S.in_noft & ~S.in_ft, "kept actives": a1 & S.in_ft & S.in_noft,
             "demoted false positives": a0 & S.in_noft & ~S.in_ft, "new false positives": a0 & S.in_ft & ~S.in_noft, "kept false positives": a0 & S.in_ft & S.in_noft}
        pr, lo = [], []                                 # per seed: promoted / lost actives (seed ranking vs No-FT)
        for s in d["per"]["FT N=300"]:
            ins = pd.Series(s).rank(ascending=False, method="first").values <= k; pr.append(int((a1.values & ins & ~S.in_noft.values).sum())); lo.append(int((a1.values & ~ins & S.in_noft.values).sum()))
        RR[t] = dict(S=S, G=G, k=k, seed_prom=pr, seed_lost=lo)
    except Exception as e:
        print(f"{SHORT[t]}: section 3 failed ({type(e).__name__}: {e}); skipped")
rows = [dict(target=SHORT[t], k=r["k"], **{g: int(m.sum()) for g, m in r["G"].items()}, promoted_seed=msd(r["seed_prom"]), lost_seed=msd(r["seed_lost"]),
             changed_status=int(sum(m.sum() for g, m in r["G"].items() if g in ("promoted actives", "lost actives", "demoted false positives", "new false positives")))) for t, r in RR.items()]
T3 = pd.DataFrame(rows).set_index("target"); display(T3)
for t in PEND: print(f"{SHORT[t]}: {PEND[t]}")"""),
md("### Every compound, before and after\nRank under No-FT (x) against rank under head-FT (y), log scales, 1 = best at the top-right. Dashed: top 1%. Points above the diagonal moved up."),
code(r"""fig, AX = grid()
for t, ax in AX.items():
    if t not in RR: placeholder(ax, t); ax.set_title(f"{SHORT[t]}: pending", fontsize=10); continue
    S, k, N = RR[t]["S"], RR[t]["k"], len(RR[t]["S"]); ina = S[S.label == 0].sample(n=min(6000, int((S.label == 0).sum())), random_state=0)
    ax.scatter(ina.rank_noft, ina.rank_ft, s=2, color="#bdbdbd", alpha=0.4, rasterized=True, label="inactive (6,000 sampled)")
    a = S[S.label == 1]; ax.scatter(a.rank_noft, a.rank_ft, s=10, color="#1baf7a", alpha=0.85, zorder=3, label=f"active (all {len(a)})")
    ax.plot([1, N], [1, N], color=BLK, lw=0.7); ax.axvline(k, ls="--", color=BLK, lw=0.7); ax.axhline(k, ls="--", color=BLK, lw=0.7)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.invert_xaxis(); ax.invert_yaxis(); ax.set_xlabel("rank under No-FT"); ax.set_ylabel("rank under head-FT N=300"); ax.set_title(f"{SHORT[t]}: rank before and after", fontsize=10)
list(AX.values())[0].legend(fontsize=7, loc="lower right"); plt.tight_layout(); plt.show()"""),
md("### How far do compounds move?\nDistribution of the rank shift for actives and inactives."),
code(r"""fig, AX = grid(); b = np.linspace(-10, 10, 50); SH = []
for t, ax in AX.items():
    if t not in RR: placeholder(ax, t); ax.set_title(f"{SHORT[t]}: pending", fontsize=10); continue
    S = RR[t]["S"]; ax.hist(S[S.label == 0].rshift.clip(-10, 10), bins=b, density=True, alpha=0.5, color=GREY, label="inactive"); ax.hist(S[S.label == 1].rshift.clip(-10, 10), bins=b, density=True, alpha=0.6, color="#1baf7a", label="active")
    ax.axvline(0, color=BLK, lw=1); ax.set_xlabel("log2(No-FT rank / head-FT rank); > 0 moved up"); ax.set_ylabel("density"); ax.set_title(f"{SHORT[t]}: rank shift", fontsize=10)
    for lab, nm in [(1, "active"), (0, "inactive")]:
        v = S[S.label == lab].rshift; SH.append(dict(target=SHORT[t], group=nm, n=len(v), median_shift=v.median(), factor=2 ** v.median(), pct_up=100 * (v > 0).mean(), pct_up_4x=100 * (v > 2).mean(), pct_down_4x=100 * (v < -2).mean()))
list(AX.values())[0].legend(fontsize=7); plt.tight_layout(); plt.show(); TSH = pd.DataFrame(SH).set_index(["target", "group"]); display(TSH.round(2))"""),

md("""### What do the rescued actives look like?
Two descriptors, both computed without the fine-tuned model.
1. Percentile under the dataset's own other scores (Boltz-2 score, Boltzina, gnina, vina). Each score is oriented so that a higher percentile is 'better' (the orientation is set by which direction gives AUROC above 0.5 over the whole library, so it is data-derived; AUROC shown). A rescued active with a high percentile is one that other methods also liked; a low percentile would mean fine-tuning found something those scores miss. gnina and vina are available only for part of the library.
2. Highest ECFP4 Tanimoto (2048 bits, radius 2) to any of the 300-compound training set's actives. Training compounds are not in the evaluation set, but their analogs can be."""),
code(r"""from scipy.stats import mannwhitneyu
DS = ["score_boltz2", "score_boltzina", "score_gnina", "score_vina"]; rows = []; SIM = {}
for t, r in RR.items():
    S, G = r["S"], r["G"]
    for c in DS:
        try:
            o = oriented_percentiles(S, c)
            if o is None: continue
            pc, auc = o; v = lambda m: pc[m].dropna()
            pa, la, al = v(G["promoted actives"]), v(G["lost actives"]), v(S.label == 1)
            rows.append(dict(target=SHORT[t], score=c, coverage=f"{100*S[c].notna().mean():.0f}%", AUROC=auc, n_promoted=len(pa), med_promoted=pa.median(), n_lost=len(la), med_lost=la.median(), med_all_actives=al.median(),
                             p_prom_vs_all=mannwhitneyu(pa, al).pvalue if len(pa) > 2 else np.nan))
        except Exception as e:
            print(f"{SHORT[t]} {c}: {type(e).__name__}: {e}")
TD = pd.DataFrame(rows)
if len(TD): display(TD.set_index(["target", "score"]).round(3)); print(f"{len(TD)} Mann-Whitney tests (promoted actives vs all held-out actives); Bonferroni threshold for 0.05: {0.05/len(TD):.4f}. Tests within a target are not independent and targets with few promoted actives have little power.")
for t in PEND: print(f"{SHORT[t]}: {PEND[t]}")"""),
code(r"""rows = []
for t, r in RR.items():
    try:
        S, G = r["S"], r["G"]; tt = sim_table(t); tr = train_ids(t, 300)
        if tr is None: print(f"{SHORT[t]}: pending: no train_ids_top300.txt"); continue
        tra = [i for i in tr if bool(tt.target_active_v2.get(i, False))]; ref = [f for f in fps(tt["neut-smiles"].reindex(tra).values) if f is not None]
        grp = dict(G); grp["all held-out actives"] = S.label == 1; grp["random inactives (500)"] = S.index.isin(S[S.label == 0].sample(n=min(500, int((S.label == 0).sum())), random_state=0).index)
        SIM[t] = {}
        for g in ["promoted actives", "lost actives", "kept actives", "all held-out actives", "new false positives", "demoted false positives", "random inactives (500)"]:
            m = grp[g] if isinstance(grp[g], pd.Series) else pd.Series(grp[g], index=S.index); v = max_sim(S[m.values].smiles.values, ref); SIM[t][g] = v
            rows.append(dict(target=SHORT[t], group=g, n=len(v), median_maxTanimoto=np.nanmedian(v) if len(v) else np.nan, pct_ge_0_5=100 * np.nanmean(v >= 0.5) if len(v) else np.nan, pct_ge_0_6=100 * np.nanmean(v >= 0.6) if len(v) else np.nan))
        SIM[t]["n_train_actives"] = len(ref)
    except Exception as e:
        print(f"{SHORT[t]}: similarity failed ({type(e).__name__}: {e}); skipped")
TS = pd.DataFrame(rows)
if len(TS): display(TS.set_index(["target", "group"]).round(2))
for t in SIM: print(f"{SHORT[t]}: {SIM[t]['n_train_actives']} training actives used as reference")
fig, AX = grid(); b = np.linspace(0, 1, 26)
for t, ax in AX.items():
    if t not in SIM: placeholder(ax, t); ax.set_title(f"{SHORT[t]}: pending", fontsize=10); continue
    for g, c in [("promoted actives", "#1baf7a"), ("lost actives", "#e08a1e"), ("all held-out actives", GREY)]:
        v = SIM[t][g][~np.isnan(SIM[t][g])]
        if len(v): ax.hist(v, bins=b, density=True, alpha=0.5, color=c, label=f"{g} (n={len(v)})")
    ax.set_xlabel("highest Tanimoto to a training active"); ax.set_ylabel("density"); ax.set_title(f"{SHORT[t]}: similarity to training actives", fontsize=10); ax.legend(fontsize=7)
plt.tight_layout(); plt.show()
ST = []
for t in SIM:
    for a_, b_ in [("promoted actives", "lost actives"), ("promoted actives", "all held-out actives"), ("new false positives", "random inactives (500)")]:
        x, z = SIM[t][a_][~np.isnan(SIM[t][a_])], SIM[t][b_][~np.isnan(SIM[t][b_])]
        ST.append(dict(target=SHORT[t], comparison=f"{a_} vs {b_}", n=f"{len(x)} / {len(z)}", median=f"{np.median(x):.2f} / {np.median(z):.2f}" if len(x) and len(z) else "", MW_p=mannwhitneyu(x, z).pvalue if len(x) > 2 and len(z) > 2 else np.nan))
if ST: display(pd.DataFrame(ST).set_index(["target", "comparison"]).round(4)); print(f"{len(ST)} Mann-Whitney tests; Bonferroni threshold for 0.05: {0.05/len(ST):.4f}")"""),

md("""### The molecules: promoted and lost actives, with Tanimoto similarity to the nearest training active
For each filled target, up to 8 promoted and 8 lost actives (largest rank change first). Each active is drawn **next to the training active it most resembles**: the legend of the first molecule gives its compound id, its rank under No-FT then under head-FT, and the **ECFP4 Tanimoto** (2048 bits, radius 2) to the nearest of the 300-compound training set's actives; the legend of the second molecule marks the nearest training active. A scatter then plots that Tanimoto against the rank change for every promoted and lost active of the target."""),
code(r"""from rdkit import Chem, DataStructs
from rdkit.Chem import Draw
def nearest_train(smiles, ref_fps):
    out = []
    for f in fps(smiles):
        if f is None: out.append((np.nan, -1)); continue
        sims = DataStructs.BulkTanimotoSimilarity(f, ref_fps); j = int(np.argmax(sims)); out.append((float(sims[j]), j))
    return out
REFS = {}
for t, r in RR.items():
    try:
        tt = sim_table(t); tr = train_ids(t, 300)
        if tr is None: print(f"{SHORT[t]}: pending: no train_ids_top300.txt"); continue
        tra = [i for i in tr if bool(tt.target_active_v2.get(i, False))]; smi = list(tt["neut-smiles"].reindex(tra).values)
        keep = [(s, f) for s, f in zip(smi, fps(smi)) if f is not None]; REFS[t] = ([k[0] for k in keep], [k[1] for k in keep])
    except Exception as e: print(f"{SHORT[t]}: could not build the training-active reference ({type(e).__name__}: {e})")
def draw_pairs(df, ref, n, title):
    d = df.head(n); nn = nearest_train(d.smiles.values, ref[1]); mols, leg = [], []
    for (_, r), (sim, j) in zip(d.iterrows(), nn):
        m = Chem.MolFromSmiles(r.smiles); k = Chem.MolFromSmiles(ref[0][j]) if j >= 0 else None
        if m is None or k is None: continue
        mols += [m, k]; leg += [f"{str(r.id).split('_')[-1]}  rank {int(r.rank_noft):,} -> {int(r.rank_ft):,}  Tanimoto {sim:.2f}", "nearest training active"]
    if not mols: print(title, ": nothing to draw"); return
    print(title); display(Draw.MolsToGridImage(mols, molsPerRow=4, subImgSize=(260, 200), legends=leg))
SC = {}
for t, r in RR.items():
    if t not in REFS: continue
    S, G = r["S"], r["G"]; ref = REFS[t]
    draw_pairs(S[G["promoted actives"]].sort_values("rshift", ascending=False), ref, 8, f"{SHORT[t]}: promoted actives ({int(G['promoted actives'].sum())} in total), each with its nearest training active")
    draw_pairs(S[G["lost actives"]].sort_values("rshift"), ref, 8, f"{SHORT[t]}: lost actives ({int(G['lost actives'].sum())} in total), each with its nearest training active")
    SC[t] = {g: (max_sim(S[G[g]].smiles.values, ref[1]), S[G[g]].rshift.values) for g in ("promoted actives", "lost actives")}
for t in PEND: print(f"{SHORT[t]}: {PEND[t]}")
if SC:
    fig, AX = grid()
    for t, ax in AX.items():
        if t not in SC: placeholder(ax, t); ax.set_title(f"{SHORT[t]}: pending", fontsize=10); continue
        for g, c in (("promoted actives", "#1baf7a"), ("lost actives", "#e08a1e")):
            sim, sh = SC[t][g]; ax.scatter(sim, sh, s=14, color=c, alpha=0.7, label=f"{g} (n={len(sim)})")
        ax.axhline(0, color="black", lw=0.6); ax.set_xlabel("Tanimoto to the nearest training active"); ax.set_ylabel("rank shift (positive = moved up under head-FT)"); ax.set_title(f"{SHORT[t]}: similarity against rank change", fontsize=10); ax.legend(fontsize=7)
    plt.tight_layout(); plt.show()"""),

md("""## 4. Per-target summary and cross-target panel

Average precision (AP) and EF1% are given for our runs next to the paper. Ours for head-FT is the mean +/- SD over the 5 seeds (AP) or the 5-seed ensemble (EF1%, top-1% counts). Targets are the unit for the cross-target claims; the sign test counts targets, not compounds."""),
code(r"""from sklearn.metrics import average_precision_score as AP
rows = []
for t in TARGETS:
    pb, pf = paper_row(t), paper_row(t, 300)
    row = dict(target=SHORT[t], status="filled" if t in D else PEND[t], paper_AP_noft=pb.auprc if pb is not None else np.nan, paper_AP_ft300=pf.auprc if pf is not None else np.nan,
               paper_EF1_noft=pb.ef_1pct if pb is not None else np.nan, paper_EF1_ft300=pf.ef_1pct if pf is not None else np.nan)
    if t in D:
        d = D[t]; y = d["y"]; k = kof(0.01, d["N"]); rt = d["y"].mean(); aps = [AP(y, s) for s in d["per"]["FT N=300"]]; ef = lambda s: conf_k(s, y, k)["TP"] / k / rt
        row.update(n_eval=d["N"], n_active=d["P"], AP_noft=AP(y, d["ens"]["No-FT"]), AP_ft300=f"{np.mean(aps):.3f} +/- {np.std(aps, ddof=1):.3f}", AP_ratio=np.mean(aps) / AP(y, d["ens"]["No-FT"]),
                   EF1_noft=ef(d["ens"]["No-FT"]), EF1_ft300=ef(d["ens"]["FT N=300"]))
        if t in R2:
            r = R2[t]; m = r["matched"].set_index("level")
            row.update(TP1_noft=r["TP_a"], TP1_ft300=r["TP_b"], dTP1=r["dTP"], dTP1_CI=f"[{r['dTP_ci'][0]:+.0f}, {r['dTP_ci'][1]:+.0f}]", FP1_noft=r["FP_a"], FP1_ft300=r["FP_b"],
                       thr_verdict=verdict(r["thr_a"], r["thr_b"]), dFP_at_sens50=m.loc["sensitivity 50%", "dFP"])
        if t in RR: row.update(promoted=int(RR[t]["G"]["promoted actives"].sum()), lost=int(RR[t]["G"]["lost actives"].sum()))
    rows.append(row)
T4 = pd.DataFrame(rows).set_index("target"); pd.set_option("display.width", 250, "display.max_columns", 40); display(T4.round(3))"""),
code(r"""fig, axs = plt.subplots(1, 3, figsize=(19, 5)); X = np.arange(len(TARGETS)); lab = [SHORT[t] for t in TARGETS]
ax = axs[0]   # EF1% ours vs paper
for i, t in enumerate(TARGETS):
    if t in D:
        ax.bar(i - 0.2, T4.loc[SHORT[t], "EF1_noft"], 0.38, color=GREY, label="ours No-FT" if i == 0 else None); ax.bar(i + 0.2, T4.loc[SHORT[t], "EF1_ft300"], 0.38, color=OUR, label="ours head-FT N=300 (ensemble)" if i == 0 else None)
    ax.scatter(i - 0.2, T4.loc[SHORT[t], "paper_EF1_noft"], marker="D", color=THEIR, zorder=4, s=36, label="paper" if i == 0 else None); ax.scatter(i + 0.2, T4.loc[SHORT[t], "paper_EF1_ft300"], marker="D", color=THEIR, zorder=4, s=36)
ax.set_xticks(X); ax.set_xticklabels(lab, fontsize=8); ax.set_ylabel("enrichment factor at 1%"); ax.set_title("EF1%: ours against the paper (paper-only where pending)", fontsize=10); ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)
ax = axs[1]   # delta TP at top 1%
for i, t in enumerate(TARGETS):
    if t in R2:
        r = R2[t]; ax.bar(i, r["dTP"], 0.6, color=OUR, yerr=[[r["dTP"] - r["dTP_ci"][0]], [r["dTP_ci"][1] - r["dTP"]]], capsize=3, ecolor=BLK, label="ours: ensemble, 95% bootstrap CI" if i == 0 or (i > 0 and not any(x in R2 for x in TARGETS[:i])) else None)
    ax.scatter(i, paper_tp(t, 300) - paper_tp(t), marker="D", color=THEIR, zorder=4, s=36, label="paper (from EF1%)" if i == 0 else None)
ax.axhline(0, color=BLK, lw=1); ax.set_xticks(X); ax.set_xticklabels(lab, fontsize=8); ax.set_ylabel("more actives in top 1%: head-FT N=300 minus No-FT"); ax.set_title("gain in actives found at the top-1% cutoff", fontsize=10); ax.legend(fontsize=7)
ax = axs[2]   # delta FP at matched sensitivity 50%
for i, t in enumerate(TARGETS):
    if t in R2:
        m = R2[t]["matched"].set_index("level").loc["sensitivity 50%"]; c = m.dFP_ci; ax.bar(i, m.dFP, 0.6, color=OUR, yerr=[[m.dFP - c[0]], [c[1] - m.dFP]], capsize=3, ecolor=BLK)
    else: ax.text(i, 0.03, "pending", rotation=90, ha="center", va="bottom", fontsize=8, transform=ax.get_xaxis_transform())
ax.axhline(0, color=BLK, lw=1); ax.set_xticks(X); ax.set_xticklabels(lab, fontsize=8); ax.set_ylabel("false positives at 50% sensitivity: head-FT minus No-FT"); ax.set_title("false positives at 50% sensitivity (negative = head-FT better)", fontsize=10)
plt.tight_layout(); plt.show()"""),
code(r"""def sign(vals, name):
    vals = [v for v in vals if v != 0]; n = len(vals); k = sum(v > 0 for v in vals)
    if n == 0: return f"{name}: no filled target, no test possible.", None
    p = stats.binomtest(k, n, 0.5).pvalue
    return f"{name}: n targets = {n}, k = {k} in the stated direction; exact two-sided sign test p = {p:.3g} (smallest attainable p with n = {n} is {2*0.5**n:.3g}).", p
tests = [sign([R2[t]["dTP"] for t in R2], "head-FT finds more top-1% actives than No-FT"),
         sign([-R2[t]["matched"].set_index("level").loc["sensitivity 50%", "dFP"] for t in R2], "head-FT needs fewer false positives than No-FT to reach 50% sensitivity"),
         sign([int(RR[t]["G"]["promoted actives"].sum()) - int(RR[t]["G"]["lost actives"].sum()) for t in RR], "head-FT promotes more actives into the top 1% than it loses")]
for s, p in tests: print(s)
print(f"{len(tests)} sign tests were run; Bonferroni threshold for 0.05 is {0.05/len(tests):.4f}. A sign test ignores effect size and with few targets has little power; "
      f"in the paper's own table, head-FT N=300 beats No-FT in EF1% on {int(sum(paper_row(t, 300).ef_1pct > paper_row(t).ef_1pct for t in TARGETS))} of {len(TARGETS)} targets (fixed paper value).")"""),

md("## Conclusions\nEvery number below is computed from the data above; nothing is stated for targets that are pending."),
code(r"""L = []
L.append(f"Targets filled: {len(D)} of {len(TARGETS)} ({', '.join(SHORT[t] for t in D) or 'none'}). Pending: {', '.join(SHORT[t] + ' (' + PEND[t].replace('pending: ', '') + ')' for t in PEND) or 'none'}.")
for t, r in R2.items():
    ta, tb = r["thr_a"], r["thr_b"]; m = r["matched"].set_index("level"); d = D[t]
    L.append(f"\nTarget {SHORT[t]} ({d['N']:,} held-out compounds, {d['P']} active, training compounds excluded).")
    L.append(f"  Matched top 1% (k = {r['k']}): head-FT N=300 ensemble finds {r['TP_b']} actives against {r['TP_a']} for No-FT (difference {r['dTP']:+d}, 95% bootstrap CI [{r['dTP_ci'][0]:+.0f}, {r['dTP_ci'][1]:+.0f}]; the 5 single seeds find {min(r['seed_TP'])} to {max(r['seed_TP'])}), "
             f"false positives {r['FP_b']} against {r['FP_a']}, missed actives {d['P']-r['TP_b']} against {d['P']-r['TP_a']}. Paper (EF1% converted): {paper_tp(t, 300):.0f} against {paper_tp(t):.0f}.")
    L.append(f"  Default p > 0.5: No-FT flags {ta['flagged']} (finds {ta['TP']}, FP {ta['FP']}, misses {ta['FN']}); head-FT ensemble flags {tb['flagged']} (finds {tb['TP']}, FP {tb['FP']}, misses {tb['FN']}). Verdict: {verdict(ta, tb)}.")
    x = m.loc["sensitivity 50%"]; L.append(f"  At 50% sensitivity No-FT accepts {x.FP_noft} false positives and head-FT {x.FP_ft} (specificity {100*x.spec_noft:.2f}% against {100*x.spec_ft:.2f}%; difference CI [{x.dFP_ci[0]:+.0f}, {x.dFP_ci[1]:+.0f}]).")
    if t in RR:
        g = RR[t]["G"]; L.append(f"  Re-ranking at the top 1%: {int(g['promoted actives'].sum())} actives promoted, {int(g['lost actives'].sum())} lost, {int(g['kept actives'].sum())} kept; {int(g['demoted false positives'].sum())} false positives removed, {int(g['new false positives'].sum())} created.")
    if t in SIM:
        s_ = lambda g: np.nanmedian(SIM[t][g]) if len(SIM[t][g]) else np.nan
        L.append(f"  Similarity to the {SIM[t]['n_train_actives']} training actives (median highest Tanimoto): promoted actives {s_('promoted actives'):.2f}, lost {s_('lost actives'):.2f}, all held-out actives {s_('all held-out actives'):.2f}.")
L.append("")
for s, p in tests: L.append(s)
L.append("Limits: intervals reflect the evaluation-set sample and not the choice of training compounds (the seed range is shown separately); the paper reports one value per target and budget; "
         "the evaluation set is disjoint from the 300 training compounds but not from their chemical families; per-target intervals are not corrected for multiplicity.")
print("\n".join(L))"""),
]
nb = nbf.v4.new_notebook(); nb.cells = cells
nb.metadata = {"kernelspec": {"name": "boltzba", "display_name": "Python (boltzba)", "language": "python"}, "language_info": {"name": "python"}}
NB.write_text(nbf.writes(nb)); print("wrote", NB)

#!/usr/bin/env python3
"""Builds 01_replication.ipynb (self-filling, all 8 targets). Matplotlib only, black text, no bold, NO HTML."""
import nbformat as nbf
from pathlib import Path
NB = Path("/global/scratch/users/sergiomar10/boltzaff/notebooks/01_replication.ipynb")
md = nbf.v4.new_markdown_cell; code = nbf.v4.new_code_cell
PRE = r'''
import sys; sys.path.insert(0, "/global/scratch/users/sergiomar10/boltzaff/pipeline_local")
from bft_replication import *
import matplotlib.pyplot as plt
from IPython.display import display
style()
MT = {}
for t in TARGETS:
    try: d = metrics_table(t)
    except Exception as e: d = None; print(f"{SHORT[t]}: error while computing metrics: {e!r}")
    if d is None: print(f"pending: {SHORT[t]} - needs No-FT and all 15 head-FT arms scored (status {status().loc[SHORT[t], 'arms_scored']})")
    else: MT[t] = d
DONE = list(MT); K = len(DONE)
print("\n" + "=" * 50 + f"\n  n targets = {K} (of 8): {[SHORT[t] for t in DONE]}\n" + "=" * 50)
if K < 2: print("k < 2: cross-target tests (sign test, Wilcoxon, t-CI) cannot be computed yet; only per-target values are shown.")
elif K < 6: print(f"k = {K}: the two-sided sign test cannot reach p < 0.05 with fewer than 6 targets (minimum p = {2*0.5**K:.3f}); treat all cross-target p-values as descriptive.")
BUD = BUDGETS
def ours_base(t, m): d = MT[t]; return float(d[d.arm == "base"][m].iloc[0])
def ours_seeds(t, N, m): d = MT[t]; return d[(d.arm == "headft") & (d.n_train == N)][m].values
def paper_base(t, m): return float(PAPER_BASE.loc[t, MCOLS[m]])
def paper_ft(t, N, m): return float(PAPER_FT.loc[(t, N), MCOLS[m]])
'''
cells = [
 md("# 01 Replication of the head-FT result (Furui and Ohue, arXiv:2609.24302)\n\n"
    "Self-filling over the 8 targets: every target whose No-FT and 15 head-FT arms (3 budgets x 5 seeds) are fully scored is analysed; the rest print a pending line and appear as paper-only placeholders. "
    "Metrics (AP, EF@1%, BEDROC alpha=20) are computed here from the per-compound scores with the authors' `BoltzFT/evaluation/metrics.py`. "
    "Cross-target statistics use targets as the analysis unit, on per-target values averaged over the 5 seeds."),
 code(PRE),

 md("## 1. The paper's own numbers (all 8 targets, reference)\nGeometric mean over the 8 targets of head-FT / No-FT, computed from the released per-target values (trainer_control.csv). The paper states AP 2.14x and EF@1% 1.77x at N=300."),
 code(r'''rows = []
for m in MCOLS:
    for N in BUD:
        g = geo_stats([paper_ft(t, N, m) / paper_base(t, m) for t in TARGETS])
        rows.append({"metric": MNAME[m], "N": N, "paper_geo_mean_ratio": g["geo"], "95% t-CI": f"[{g['lo']:.2f}, {g['hi']:.2f}]", "improved": f"{g['improved']}/8", "sign_p": g["sign_p"], "wilcoxon_p": g["wilcoxon_p"]})
PAPER_T2 = pd.DataFrame(rows); display(PAPER_T2.style.format(precision=3))
PG = {(r.metric, r.N): r.paper_geo_mean_ratio for r in PAPER_T2.itertuples()}
print(f"paper reference at N=300 (8 targets): AP {PG[('AP',300)]:.2f}x, EF@1% {PG[('EF@1%',300)]:.2f}x, BEDROC {PG[('BEDROC(a=20)',300)]:.2f}x")'''),

 md("## 1b. The No-FT starting point for all 8 targets (available now)\n"
    "Three No-FT numbers per target: the **paper's** reported value; the **dataset's shipped Boltz-2 probability** scored on the paper's own eval subset (`selection/eval_subset.csv`, "
    "available for every target now, so this checks the labels, the eval-set definition and our metric code on all 8 targets); and **our own No-FT re-scoring** through the two-pass "
    "pipeline, which appears for a target as soon as its No-FT arm is fully scored (it does not wait for the head-FT arms). The shipped score is the paper group's own run, so agreement "
    "with the paper checks the evaluation, not our Boltz-2 runs; the third column is the real test of our pipeline."),
 code(r'''NT = nofT_table()
cols = ["paper_n_eval","shipped_n_eval","paper_n_active","shipped_n_active","paper_AP","shipped_AP","ours_AP","paper_EF1","shipped_EF1","ours_EF1","spearman_ours_vs_shipped"]
display(NT[[c for c in cols if c in NT]].round(3))
same = ((NT.paper_n_eval == NT.shipped_n_eval) & (NT.paper_n_active == NT.shipped_n_active)).sum()
print(f"eval-set size and active count identical to the paper for {same} of {len(NT)} targets")
dap = (NT.shipped_AP - NT.paper_AP).abs(); print(f"|shipped AP - paper AP|: median {dap.median():.4f}, max {dap.max():.4f} ({NT.index[dap.argmax()]})")
if NT.ours_AP.notna().any():
    o = NT[NT.ours_AP.notna()]
    for tg, r in o.iterrows(): print(f"our No-FT, target {tg}: AP {r.ours_AP:.3f} vs paper {r.paper_AP:.3f} (shipped {r.shipped_AP:.3f}); Spearman with the shipped score {r.spearman_ours_vs_shipped:.3f}")
else: print("pending: our own No-FT re-scoring is not complete for any target yet")
fig, ax = plt.subplots(figsize=(12, 4.6)); x = np.arange(len(NT)); w = 0.26
ax.bar(x - w, NT.paper_AP, w, color=THEIR, label="paper No-FT"); ax.bar(x, NT.shipped_AP, w, color="#bbbbbb", label="shipped Boltz-2 score on the paper's eval subset")
ax.bar(x + w, NT.ours_AP, w, color=OUR, label="our No-FT re-scoring (where complete)")
ax.plot(x, [RATE[t] for t in TARGETS], "k_", ms=22, mew=1.5, label="active rate (random ranking)")
ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(NT.index); ax.set_ylabel("average precision (log)"); ax.legend(fontsize=8)
ax.set_title("No-FT average precision per target: paper, shipped score, our re-scoring"); plt.tight_layout(); plt.show()'''),
 md("## 2. Consistency check of our metric computation\nOur metrics computed here are compared with `results/analysis/<t>/per_seed.csv` (the eval_headft_seeds.py output) where it exists. Differences above 1e-6 are reported."),
 code(r'''for t in TARGETS:
    if t not in MT: continue
    ps = per_seed(t)
    if ps is None: print(f"{SHORT[t]}: no per_seed.csv yet, nothing to compare"); continue
    a = MT[t].merge(ps, on=["arm", "n_train", "seed"], suffixes=("", "_ref"))
    if len(a) != len(MT[t]): print(f"{SHORT[t]}: per_seed.csv has {len(a)} matching rows out of {len(MT[t])}")
    mm = {m: float((a[m] - a[m + "_ref"]).abs().max()) for m in MCOLS}
    bad = {m: v for m, v in mm.items() if v > 1e-6}
    print(f"{SHORT[t]}: max abs difference vs per_seed.csv " + ", ".join(f"{m} {v:.1e}" for m, v in mm.items()) + ("  -> MISMATCH" if bad else "  -> consistent"))'''),

 md("## 3. Table 2 for our runs: geometric-mean head-FT / No-FT ratios\nPer-target head-FT value = mean over the 5 seeds; ratio = that mean / No-FT value. Geometric mean across scored targets, 95% t-interval in log space, improved-target count with two-sided sign test, Wilcoxon signed-rank on per-target log ratios. "
    "The paper columns use the same subset of targets as ours (so the comparison is like for like), plus the 8-target paper value."),
 code(r'''rows = []
for m in MCOLS:
    for N in BUD:
        if not DONE: break
        ro = [np.mean(ours_seeds(t, N, m)) / ours_base(t, m) for t in DONE]
        rp = [paper_ft(t, N, m) / paper_base(t, m) for t in DONE]
        g, gp = geo_stats(ro), geo_stats(rp)
        rows.append({"metric": MNAME[m], "N": N, "k": g["k"], "ours_geo_ratio": g["geo"], "ours 95% t-CI": f"[{g['lo']:.2f}, {g['hi']:.2f}]" if g["k"] > 1 else "-",
                     "improved": f"{g['improved']}/{g['k']}", "sign_p": g["sign_p"], "wilcoxon_p": g["wilcoxon_p"],
                     "paper_geo_same_targets": gp["geo"], "paper_geo_8_targets": PG[(MNAME[m], N)]})
T2 = pd.DataFrame(rows)
if len(T2): display(T2.style.format(precision=3, na_rep="-"))
print(f"n targets = {K}")
if K >= 2:
    print(f"Multiple comparisons: {len(T2)} (metric x budget) tests are shown, so the Bonferroni threshold for a family-wise 0.05 is {0.05/len(T2):.4f}; "
          f"with k = {K} the smallest attainable two-sided sign-test p is {2*0.5**K:.4f}.")
    print("Metric-wise ratios are not independent (same compounds, same runs), so the tests are not a set of independent confirmations.")'''),
 code(r'''if DONE:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), sharey=False); off = {40: -0.22, 100: 0, 300: 0.22}; cols = {40: "#9ecae1", 100: "#4292c6", 300: "#08306b"}
    for ax, m in zip(axes, MCOLS):
        for N in BUD:
            for i, t in enumerate(DONE):
                s = ours_seeds(t, N, m) / ours_base(t, m)
                ax.scatter(np.full(len(s), i + off[N]) + np.linspace(-0.04, 0.04, len(s)), s, s=10, color=cols[N], alpha=0.5)
                ax.scatter(i + off[N], s.mean(), s=55, color=cols[N], label=f"ours N={N} (mean, dots = seeds)" if i == 0 else None, zorder=4)
                ax.scatter(i + off[N] + 0.07, paper_ft(t, N, m) / paper_base(t, m), s=50, facecolor="white", edgecolor=THEIR, lw=1.8, label="paper (single seed)" if (i == 0 and N == 40) else None, zorder=5)
        ax.axhline(1, color="black", lw=1); ax.set_yscale("log"); ax.set_xticks(range(len(DONE))); ax.set_xticklabels([SHORT[t] for t in DONE])
        ax.set_xlabel("target"); ax.set_ylabel(f"head-FT {MNAME[m]} / No-FT {MNAME[m]} (log)"); ax.set_title(f"per-target {MNAME[m]} ratio"); ax.legend(fontsize=7)
    plt.suptitle(f"head-FT / No-FT ratio per scored target (n targets = {K}); ours filled, paper hollow", y=1.02); plt.tight_layout(); plt.show()
else: print("pending: no scored target, nothing to draw")'''),

 md("## 4. AP per target on a log absolute axis (paper vs ours)\nNo-FT and head-FT N=40/100/300. Orange hollow = paper; blue = ours (marker = seed mean, bar = seed SD, small dots = the 5 seeds). Dashed line = the target's active rate, which is the AP of a random ranking. Unscored targets show the paper only."),
 code(r'''fig, axes = plt.subplots(2, 4, figsize=(19, 8.6))
for ax, t in zip(axes.ravel(), TARGETS):
    xs = np.arange(4); labs = ["No-FT"] + [f"N={b}" for b in BUD]
    pv = [paper_base(t, "ap")] + [paper_ft(t, b, "ap") for b in BUD]
    ax.scatter(xs - 0.16, pv, s=70, facecolor="white", edgecolor=THEIR, lw=2, zorder=4, label="paper")
    txt = f"paper N=300 / No-FT: x{pv[3]/pv[0]:.2f}"
    if t in MT:
        b0 = ours_base(t, "ap"); means = [b0]; sds = [0.0]; ax.scatter([0.16], [b0], s=70, color=OUR, zorder=4)
        for i, b in enumerate(BUD):
            s = ours_seeds(t, b, "ap"); means.append(s.mean()); sds.append(s.std(ddof=1))
            ax.scatter(np.full(len(s), i + 1.16) + np.linspace(-0.05, 0.05, len(s)), s, s=14, color=OUR, alpha=0.5, zorder=3)
        ax.errorbar(xs + 0.16, means, yerr=sds, fmt="o", color=OUR, ms=7, capsize=3, zorder=5, label="ours")
        txt += f"\nours: x{means[3]/means[0]:.2f}"
    else: ax.text(0.5, 0.05, "pending: ours not scored yet", transform=ax.transAxes, ha="center", fontsize=8)
    ax.axhline(RATE[t], ls="--", color="black", lw=0.9); ax.text(3.45, RATE[t], f" active rate {100*RATE[t]:.2f}%", va="bottom", ha="right", fontsize=7)
    ax.set_yscale("log"); ax.set_xticks(xs); ax.set_xticklabels(labs, fontsize=8); ax.set_xlim(-0.5, 3.5)
    ax.set_title(f"{SHORT[t]}\n{txt}", fontsize=9); ax.set_ylabel("average precision (log)")
    if t == TARGETS[0]: ax.legend(fontsize=8, loc="upper left")
fig.suptitle("average precision per target: No-FT and head-FT, paper vs ours (dashed = active rate)", fontsize=12); plt.tight_layout(); plt.show()'''),

 md("## 5. AP as a multiple of chance, and lift per target\nAP divided by the active rate (1 = random). The table lists all 8 targets with paper and our values side by side."),
 code(r'''rows = []
for t in TARGETS:
    r = dict(target=SHORT[t], active_rate_pct=100 * RATE[t], paper_NoFT_AP=paper_base(t, "ap"), paper_N300_AP=paper_ft(t, 300, "ap"),
             paper_NoFT_x_chance=paper_base(t, "ap") / RATE[t], paper_N300_x_chance=paper_ft(t, 300, "ap") / RATE[t], paper_lift=paper_ft(t, 300, "ap") / paper_base(t, "ap"))
    if t in MT:
        s = ours_seeds(t, 300, "ap"); r.update(ours_NoFT_AP=ours_base(t, "ap"), ours_N300_AP=s.mean(), ours_NoFT_x_chance=ours_base(t, "ap") / RATE[t], ours_N300_x_chance=s.mean() / RATE[t], ours_lift=s.mean() / ours_base(t, "ap"))
    rows.append(r)
LIFT = pd.DataFrame(rows).set_index("target"); display(LIFT.style.format(precision=3, na_rep="pending"))
fig, axes = plt.subplots(1, 2, figsize=(16, 4.6)); x = np.arange(8); w = 0.2
ax = axes[0]
ax.bar(x - 1.5*w, LIFT.paper_NoFT_x_chance, w, color=NEU, label="paper No-FT"); ax.bar(x - 0.5*w, LIFT.paper_N300_x_chance, w, color=THEIR, label="paper head-FT N=300")
if "ours_NoFT_x_chance" in LIFT:
    ax.bar(x + 0.5*w, LIFT.ours_NoFT_x_chance, w, color="#9ecae1", label="ours No-FT"); ax.bar(x + 1.5*w, LIFT.ours_N300_x_chance, w, color=OUR, label="ours head-FT N=300")
ax.axhline(1, color="black", ls="--", lw=1); ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(LIFT.index); ax.set_xlabel("target"); ax.set_ylabel("AP / active rate (log)"); ax.set_title("AP as a multiple of chance"); ax.legend(fontsize=7, ncol=2)
ax = axes[1]
ax.scatter(x - 0.12, LIFT.paper_lift, s=60, facecolor="white", edgecolor=THEIR, lw=2, label="paper N=300 (single seed)")
if "ours_lift" in LIFT: ax.scatter(x + 0.12, LIFT.ours_lift, s=60, color=OUR, label="ours N=300 (seed mean)")
ax.axhline(1, color="black", lw=1); ax.axhline(PG[("AP", 300)], color=THEIR, ls=":", label=f"paper 8-target geo-mean {PG[('AP',300)]:.2f}x")
ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(LIFT.index); ax.set_xlabel("target"); ax.set_ylabel("head-FT AP / No-FT AP (log)"); ax.set_title("AP lift at N=300"); ax.legend(fontsize=7)
plt.tight_layout(); plt.show()'''),

 md("## 6. Multi-seed spread, and how far the paper's single seed is from our seed mean\nPer target and budget: mean, SD, min and max of AP over our 5 seeds. `paper_minus_ours_mean` is the paper's (single-seed) AP minus our seed mean, and `in_SDs` expresses it in units of our seed SD (a rough guide only: the paper's run also differs in inputs, so this is not a pure seed-noise comparison)."),
 code(r'''rows = []
for t in DONE:
    b = ours_base(t, "ap"); rows.append(dict(target=SHORT[t], N=0, n_seeds=1, mean=b, sd=np.nan, min=b, max=b, paper=paper_base(t, "ap"), paper_minus_ours_mean=paper_base(t, "ap") - b, in_SDs=np.nan))
    for N in BUD:
        s = ours_seeds(t, N, "ap"); p = paper_ft(t, N, "ap")
        rows.append(dict(target=SHORT[t], N=N, n_seeds=len(s), mean=s.mean(), sd=s.std(ddof=1), min=s.min(), max=s.max(), paper=p, paper_minus_ours_mean=p - s.mean(), in_SDs=(p - s.mean()) / s.std(ddof=1)))
SP = pd.DataFrame(rows)
if len(SP):
    display(SP.style.format(precision=4, na_rep="-").hide(axis="index"))
    h = SP[SP.N > 0]
    print(f"head-FT arms: paper value lies inside our [min, max] over seeds in {int(((h.paper >= h['min']) & (h.paper <= h['max'])).sum())}/{len(h)} (target, budget) cells; "
          f"median relative seed SD (SD/mean) = {100*(h.sd/h['mean']).median():.1f}% ; median |paper - ours| / ours = {100*(h.paper_minus_ours_mean.abs()/h['mean']).median():.1f}%.")
    sdall = []
    for t in DONE:
        for N in BUD: sdall.append(ours_seeds(t, N, "ap").std(ddof=1) / ours_seeds(t, N, "ap").mean())
    print(f"seed-to-seed relative SD of AP ranges {100*min(sdall):.1f}% to {100*max(sdall):.1f}% over scored (target, budget) cells.")
else: print("pending: no scored target")'''),

 md("## 7. Is the paper's headline reproduced?\nAll numbers below are computed above. The comparison is like for like only on the same target subset; the paper's 8-target value is shown for reference."),
 code(r'''if K == 0:
    print("No scored target yet: headline cannot be assessed.")
else:
    tl = ", ".join(SHORT[t] for t in DONE)
    for m in ["ap", "ef1"]:
        r = T2[(T2.metric == MNAME[m]) & (T2.N == 300)].iloc[0]
        line = (f"{MNAME[m]} at N=300 over k = {K} scored target(s) [{tl}]: ours geo-mean ratio {r.ours_geo_ratio:.2f}x"
                + (f" (95% t-CI {r['ours 95% t-CI']})" if K > 1 else "") + f"; paper on the same targets {r.paper_geo_same_targets:.2f}x; paper on all 8 targets {r.paper_geo_8_targets:.2f}x.")
        print(line)
        if K > 1:
            lo, hi = [float(v) for v in r["ours 95% t-CI"].strip("[]").split(",")]
            print(f"   paper same-target value {'lies inside' if lo <= r.paper_geo_same_targets <= hi else 'lies outside'} our CI; paper 8-target value {'lies inside' if lo <= r.paper_geo_8_targets <= hi else 'lies outside'} our CI. "
                  f"improved {r.improved}, sign p = {r.sign_p:.3f}, Wilcoxon p = {r.wilcoxon_p:.3f}.")
    if K < 8: print(f"\nOnly {K}/8 targets are scored, so this is a partial replication; the cross-target claim needs all 8 and the conclusion will change as targets finish.")
    if K < 6: print(f"With k = {K} no cross-target test can reach p < 0.05 (minimum sign-test p = {2*0.5**K:.3f}); the ratios are descriptive.")
    else: print(f"Bonferroni over the {len(T2)} (metric x budget) rows: threshold {0.05/len(T2):.4f}.")
    # per-target direction
    for t in DONE:
        print(f"{SHORT[t]}: ours AP ratio N=300 {np.mean(ours_seeds(t,300,'ap'))/ours_base(t,'ap'):.2f}x vs paper {paper_ft(t,300,'ap')/paper_base(t,'ap'):.2f}x; "
              f"No-FT AP ours {ours_base(t,'ap'):.3f} vs paper {paper_base(t,'ap'):.3f}")'''),

 md("## 8. What differs from the paper\n"
    "- **Same evaluation set per target**: the eval compounds are the paper's (the common eval set, n_eval and n_active match trainer_control.csv where checked above), so No-FT values should agree closely; small differences come from our independently reconstructed protein sequences / MSAs and poses, which are not bit-identical to the authors'.\n"
    "- **Seeds**: the paper reports one training seed (seed 0); we train 5 seeds per budget and report the seed mean and spread, so our per-target head-FT values are less noisy than the paper's single-seed numbers.\n"
    "- **Training set**: the N highest-ranked actives per target (top-N), as in the paper's recipe; head-only fine-tuning of the affinity module.\n"
    "- **Statistics**: cross-target tests use targets as the unit and seed-averaged values; with few scored targets they are descriptive (see the k printed at the top)."),
]
nb = nbf.v4.new_notebook(); nb.cells = cells
nb.metadata = {"kernelspec": {"name": "boltzba", "display_name": "Python (boltzba)"}, "language_info": {"name": "python"}}
NB.write_text(nbf.writes(nb)); print("wrote", NB)

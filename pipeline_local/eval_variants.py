#!/usr/bin/env python3
"""Training-data-composition experiment evaluation (single target, multi-seed).

The 14 conditions (conds.txt) vary how the fine-tuning training set is chosen at a fixed label
budget: `top_N{40,100,300,600,1000}` is the paper's default (highest-ranked compounds), and
`{random,balanced,hardneg,diverse,stratified,actdiv}_N{300,600}` are alternative selection
strategies at the same budget. Eval set = ft_inputs_variants (rank>1000), scored per arm in
headft_variants_scores/<cond>_seed<seed>/chunk_*.csv.

Produces:
  1. variants_per_seed.csv : (cond, strategy, n_train, seed) -> AP, EF@1%, BEDROC
  2. variants_spread.csv   : (cond) -> seed mean/sd/min/max
  3. variants_compare.csv  : at each budget N, each strategy vs `top` (same N):
                             seed-mean ratio + within-condition spread (single target, so the
                             cross-target geo-mean/Wilcoxon of Table 2 does not apply; we report
                             the seed-mean ratio and a paired-over-seeds t-test instead).
Reuses BoltzFT/evaluation/metrics.py; mirrors eval_headft_seeds.py's loading.
"""
from __future__ import annotations
import argparse, os, re, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

WORKDIR = Path(os.environ["WORKDIR"])
BOLTZFT = Path(os.environ["BOLTZFT"])
sys.path.insert(0, str(BOLTZFT / "evaluation"))
from metrics import all_metrics  # noqa: E402

SCORE_COLUMN = "affinity_probability_binary"


def load_arm(scores_dir: Path, expected_chunks: int) -> pd.DataFrame | None:
    chunks = sorted(scores_dir.glob("chunk_*.csv"))
    if not chunks or len(chunks) != expected_chunks:
        return None
    frame = pd.concat([pd.read_csv(p) for p in chunks], ignore_index=True)
    frame["sample_id"] = frame["sample_id"].astype(str)
    # variant eval_chunks contain 385 overlapping ids (prep artifact); dedup keeping first.
    frame = frame.drop_duplicates("sample_id")
    return frame[["sample_id", SCORE_COLUMN]]


def parse_cond(cond: str) -> tuple[str, int]:
    """'balanced_N300' -> ('balanced', 300); 'top_N40' -> ('top', 40)."""
    m = re.match(r"^(.*)_N(\d+)$", cond)
    if not m:
        raise ValueError(f"unparseable cond: {cond}")
    return m.group(1), int(m.group(2))


def metrics_for(frame, eval_ids, labels) -> dict:
    m = frame.set_index("sample_id").reindex(eval_ids)
    if m[SCORE_COLUMN].isna().any():
        raise ValueError(f"{int(m[SCORE_COLUMN].isna().sum())} eval ids unscored")
    y = pd.Series(eval_ids).map(labels).to_numpy(dtype=int)
    s = m[SCORE_COLUMN].to_numpy(dtype=float)
    a = all_metrics(y, s)
    return {"ap": a["auprc"], "ef1": a["ef_1pct"], "bedroc": a["bedroc"],
            "n_eval": a["n_eval"], "n_active": a["n_active"]}


def collect(target: str, seeds) -> tuple[pd.DataFrame, list[str]]:
    vin = WORKDIR / "runs" / target / "ft_inputs_variants"
    sroot = WORKDIR / "runs" / target / "headft_variants_scores"
    n_chunks = len(list((vin / "eval_chunks").glob("chunk_*.txt")))
    eval_ids = [l.strip() for l in (vin / "eval_ids.txt").read_text().splitlines() if l.strip()]
    labels = (pd.read_csv(vin / "ml_table.csv", usecols=["complex_id", "is_binder"],
                          dtype={"complex_id": str}).set_index("complex_id")["is_binder"].astype(int))
    conds = [c.strip() for c in (vin / "conds.txt").read_text().splitlines() if c.strip()]
    rows, incomplete = [], []
    for cond in conds:
        strat, n = parse_cond(cond)
        for sd in seeds:
            arm = load_arm(sroot / f"{cond}_seed{sd}", n_chunks)
            if arm is None:
                incomplete.append(f"{cond}_seed{sd}")
                continue
            mm = metrics_for(arm, eval_ids, labels)
            rows.append({"cond": cond, "strategy": strat, "n_train": n, "seed": sd, **mm})
    return pd.DataFrame(rows), incomplete


def spread_table(df) -> pd.DataFrame:
    out = []
    for cond, g in df.groupby("cond"):
        row = {"cond": cond, "strategy": g.strategy.iloc[0], "n_train": int(g.n_train.iloc[0]),
               "n_seeds": len(g)}
        for k in ["ap", "ef1", "bedroc"]:
            row[f"{k}_mean"] = g[k].mean()
            row[f"{k}_sd"] = g[k].std(ddof=1) if len(g) > 1 else 0.0
            row[f"{k}_min"] = g[k].min(); row[f"{k}_max"] = g[k].max()
        out.append(row)
    return pd.DataFrame(out).sort_values(["n_train", "strategy"]).reset_index(drop=True)


def compare_table(df) -> pd.DataFrame:
    """Each strategy vs `top` at the same budget N, paired over seeds (single target)."""
    rows = []
    for n, gN in df.groupby("n_train"):
        if "top" not in set(gN.strategy):
            continue
        top = gN[gN.strategy == "top"].set_index("seed")
        for strat, gS in gN.groupby("strategy"):
            if strat == "top":
                continue
            s = gS.set_index("seed")
            seeds = sorted(set(top.index) & set(s.index))
            if not seeds:
                continue
            for metric in ["ap", "ef1", "bedroc"]:
                tv = top.loc[seeds, metric].to_numpy(dtype=float)
                sv = s.loc[seeds, metric].to_numpy(dtype=float)
                ratio = float(np.exp(np.mean(np.log(sv / tv))))  # seed geo-mean of paired ratio
                # paired t-test over seeds on log ratios (within-condition, single target)
                if len(seeds) > 1 and np.all(np.isfinite(np.log(sv / tv))):
                    p = float(stats.ttest_rel(np.log(sv), np.log(tv)).pvalue)
                else:
                    p = float("nan")
                rows.append({"n_train": int(n), "strategy": strat, "metric": metric.upper(),
                             "top_mean": float(tv.mean()), "strat_mean": float(sv.mean()),
                             "ratio_vs_top": ratio, "n_seeds": len(seeds), "paired_p": p})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", default="588689")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()
    out = args.out_dir or (WORKDIR / "analysis" / args.target)
    out.mkdir(parents=True, exist_ok=True)

    df, incomplete = collect(args.target, args.seeds)
    if incomplete:
        print(f"[note] {len(incomplete)} arms incomplete (skipped): {incomplete[:8]}"
              + (" ..." if len(incomplete) > 8 else ""))
    if df.empty:
        print("No complete variant arms yet.")
        return
    df.to_csv(out / "variants_per_seed.csv", index=False)
    sp = spread_table(df); sp.to_csv(out / "variants_spread.csv", index=False)
    cmp = compare_table(df); cmp.to_csv(out / "variants_compare.csv", index=False)

    pd.set_option("display.width", 170, "display.max_columns", 30)
    print("\n=== per-condition spread (seed mean +/- sd) ===")
    print(sp.round(4).to_string(index=False))
    print("\n=== composition vs top at same budget (ratio>1 = strategy beats naive top-N) ===")
    print(cmp.round(4).to_string(index=False))
    print(f"\nWrote variants_per_seed.csv, variants_spread.csv, variants_compare.csv -> {out}")


if __name__ == "__main__":
    main()

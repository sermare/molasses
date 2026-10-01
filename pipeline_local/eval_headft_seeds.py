#!/usr/bin/env python3
"""Multi-seed head-FT evaluation.

Produces:
  1. per_seed.csv  : (target, n_train, seed) -> AP, EF@1%, BEDROC on the common eval set
  2. spread.csv    : (target, n_train)       -> seed mean/sd/min/max (within-condition spread)
  3. table2.csv    : re-derived Table 2 from SEED-AVERAGED per-target values
                     (head-FT/No-FT geo-mean ratio, sign test, Wilcoxon, 95% CIs)

Reuses BoltzFT/evaluation/metrics.py and mirrors eval_headft_cached.py's loading.
"""
from __future__ import annotations
import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
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
    if frame["sample_id"].duplicated().any():
        raise ValueError(f"Duplicate sample_id in {scores_dir}")
    return frame[["sample_id", SCORE_COLUMN]]


def labels_for(target: str) -> pd.Series:
    ml = WORKDIR / "runs" / target / "ft_inputs_full" / "ml_table.csv"
    t = pd.read_csv(ml, usecols=["complex_id", "is_binder"], dtype={"complex_id": str})
    return t.set_index("complex_id")["is_binder"].astype(int)


def eval_ids_for(target: str) -> list[str]:
    p = WORKDIR / "runs" / target / "ft_inputs_full" / "eval_ids.txt"
    return [l.strip() for l in p.read_text().splitlines() if l.strip()]


def metrics_for(frame: pd.DataFrame, eval_ids, labels) -> dict:
    m = frame.set_index("sample_id").reindex(eval_ids)
    if m[SCORE_COLUMN].isna().any():
        raise ValueError(f"{int(m[SCORE_COLUMN].isna().sum())} eval ids unscored")
    y = pd.Series(eval_ids).map(labels).to_numpy(dtype=int)
    s = m[SCORE_COLUMN].to_numpy(dtype=float)
    a = all_metrics(y, s)
    return {"ap": a["auprc"], "ef1": a["ef_1pct"], "bedroc": a["bedroc"],
            "n_eval": a["n_eval"], "n_active": a["n_active"]}


def collect(targets, sizes, seeds) -> pd.DataFrame:
    rows = []
    for target in targets:
        run = WORKDIR / "runs" / target / "headft_affcache"
        n_chunks = len(list((WORKDIR / "runs" / target / "ft_inputs_full" / "eval_chunks").glob("chunk_*.txt")))
        labels = labels_for(target)
        eval_ids = eval_ids_for(target)

        base = load_arm(run / "base" / "scores", n_chunks)
        if base is None:
            print(f"[{target}] base arm incomplete -> skip target")
            continue
        bm = metrics_for(base, eval_ids, labels)
        rows.append({"target": target, "arm": "base", "n_train": 0, "seed": -1, **bm})

        for n in sizes:
            for sd in seeds:
                arm = load_arm(run / f"lightning_top{n}_seed{sd}" / "scores", n_chunks)
                if arm is None:
                    continue
                mm = metrics_for(arm, eval_ids, labels)
                rows.append({"target": target, "arm": "headft", "n_train": n, "seed": sd, **mm})
    return pd.DataFrame(rows)


def spread_table(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (target, n), g in df[df.arm == "headft"].groupby(["target", "n_train"]):
        row = {"target": target, "n_train": n, "n_seeds": len(g)}
        for k in ["ap", "ef1", "bedroc"]:
            row[f"{k}_mean"] = g[k].mean()
            row[f"{k}_sd"] = g[k].std(ddof=1) if len(g) > 1 else 0.0
            row[f"{k}_min"] = g[k].min()
            row[f"{k}_max"] = g[k].max()
        out.append(row)
    base = df[df.arm == "base"].set_index("target")
    for target in base.index:
        out.append({"target": target, "n_train": 0, "n_seeds": 1,
                    **{f"{k}_mean": base.loc[target, k] for k in ["ap", "ef1", "bedroc"]},
                    **{f"{k}_sd": 0.0 for k in ["ap", "ef1", "bedroc"]}})
    return pd.DataFrame(out).sort_values(["target", "n_train"]).reset_index(drop=True)


def table2(df: pd.DataFrame, sizes) -> pd.DataFrame:
    """Seed-average per-target, then aggregate across targets like the paper."""
    base = df[df.arm == "base"].set_index("target")[["ap", "ef1", "bedroc"]]
    # seed-averaged per-target head-FT values
    seed_avg = (df[df.arm == "headft"]
                .groupby(["target", "n_train"])[["ap", "ef1", "bedroc"]].mean())
    rows = []
    for n in sizes:
        for metric in ["ap", "ef1", "bedroc"]:
            try:
                ft = seed_avg.xs(n, level="n_train")[metric]
            except KeyError:
                continue
            common = ft.index.intersection(base.index)
            ft_v = ft.loc[common].to_numpy()
            bs_v = base.loc[common, metric].to_numpy()
            ratios = ft_v / bs_v
            logr = np.log(ratios)
            geo = float(np.exp(logr.mean()))
            n_t = len(ratios)
            # 95% t-CI of the geo-mean ratio in log space
            if n_t > 1:
                se = logr.std(ddof=1) / np.sqrt(n_t)
                tcrit = stats.t.ppf(0.975, n_t - 1)
                ci = (float(np.exp(logr.mean() - tcrit * se)), float(np.exp(logr.mean() + tcrit * se)))
                # sign test: ties (ratio<=1) count as non-improvement
                wins = int((ratios > 1).sum())
                sign_p = float(stats.binomtest(wins, n_t, 0.5, alternative="two-sided").pvalue)
                try:
                    wilcox_p = float(stats.wilcoxon(logr, alternative="two-sided").pvalue)
                except ValueError:
                    wilcox_p = float("nan")
            else:
                ci = (float("nan"), float("nan"))
                wins = int((ratios > 1).sum())
                sign_p = wilcox_p = float("nan")
            rows.append({"n_train": n, "metric": metric.upper(),
                         "base_mean": float(bs_v.mean()), "ft_mean": float(ft_v.mean()),
                         "geo_mean_ratio": geo, "ci_low": ci[0], "ci_high": ci[1],
                         "improved": f"{wins}/{n_t}", "sign_p": sign_p, "wilcoxon_p": wilcox_p})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--targets", nargs="+", required=True)
    ap.add_argument("--sizes", type=int, nargs="+", default=[40, 100, 300])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out-dir", type=Path, default=WORKDIR / "analysis")
    args = ap.parse_args()

    df = collect(args.targets, args.sizes, args.seeds)
    if df.empty:
        print("Nothing to evaluate.")
        return
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_dir / "per_seed.csv", index=False)
    sp = spread_table(df)
    sp.to_csv(args.out_dir / "spread.csv", index=False)
    t2 = table2(df, args.sizes)
    t2.to_csv(args.out_dir / "table2.csv", index=False)

    pd.set_option("display.width", 160, "display.max_columns", 30)
    print("\n=== within-condition spread (seed mean +/- sd) ===")
    print(sp.to_string(index=False))
    print("\n=== Table 2 (seed-averaged per-target) ===")
    print(t2.to_string(index=False))
    print(f"\nWrote per_seed.csv, spread.csv, table2.csv -> {args.out_dir}")


if __name__ == "__main__":
    main()

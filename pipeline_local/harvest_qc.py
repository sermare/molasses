#!/usr/bin/env python3
"""Harvest Boltz-2 co-folding confidence metrics for every folded 588689 compound,
merge with activity labels + the cached scores, and cache to qc.csv for the notebooks."""
import json, re
from pathlib import Path
import pandas as pd

ROOT = Path("/global/scratch/users/sergiomar10/boltzaff")
R = ROOT / "results/runs/588689"
OUT = R / "qc.csv"

rows = []
for jf in R.glob("outputs_chunks_full/chunk_*/boltz_results_*/predictions/*/confidence_*_model_0.json"):
    m = re.search(r"confidence_588689_(\d+)_model_0\.json", jf.name)
    if not m:
        continue
    try:
        d = json.loads(jf.read_text())
    except Exception:
        continue
    rows.append(dict(
        CID=int(m.group(1)),
        confidence=d.get("confidence_score"), ptm=d.get("ptm"), iptm=d.get("iptm"),
        ligand_iptm=d.get("ligand_iptm"), complex_plddt=d.get("complex_plddt"),
        complex_pde=d.get("complex_pde")))
qc = pd.DataFrame(rows).drop_duplicates("CID")

# merge activity label + cached comparator scores from the raw distributed CSV
raw = pd.read_csv(ROOT / "data/mf-pcba_test/588689.csv")
raw["CID"] = raw["CID"].astype(int)
raw["Active"] = raw["target_active_v2"].astype(bool).astype(int)
keep = ["CID", "Active", "score_boltz2", "score_boltzina", "score_gnina", "score_vina"]
qc = qc.merge(raw[keep], on="CID", how="left")
OUT.parent.mkdir(parents=True, exist_ok=True)
qc.to_csv(OUT, index=False)

n, na = len(qc), int(qc.Active.sum())
full_rate = raw["Active"].mean() * 100
print(f"folded compounds with confidence: {n}")
print(f"actives in folded subset: {na} ({na/n*100:.2f}%)  |  full-library active rate: {full_rate:.2f}%")
print(f"-> {OUT}")

#!/usr/bin/env python3
"""Collect the affinity outputs of the decoy-protein rescore (arms real / shuffled / other) into one CSV:
results/runs/588689/decoy/scores.csv with columns arm, id, prob (affinity_probability_binary), value (affinity_pred_value), prob1, prob2, value1, value2."""
import json, glob, sys
from pathlib import Path
import pandas as pd
D=Path("/global/scratch/users/sergiomar10/boltzaff/results/runs/588689/decoy"); rows=[]
for arm in ("real","shuffled","other"):
    for jf in glob.glob(str(D/"outputs"/arm/"chunk_*"/"boltz_results_*"/"predictions"/"*"/"affinity_*.json")):
        try: d=json.load(open(jf))
        except Exception: continue
        rid=Path(jf).parent.name
        rows.append(dict(arm=arm,id=rid,prob=d.get("affinity_probability_binary"),value=d.get("affinity_pred_value"),
                         prob1=d.get("affinity_probability_binary1"),prob2=d.get("affinity_probability_binary2"),value1=d.get("affinity_pred_value1"),value2=d.get("affinity_pred_value2")))
df=pd.DataFrame(rows); df.to_csv(D/"scores.csv",index=False)
print("collected",len(df),"scores:",df.groupby("arm").size().to_dict() if len(df) else {}, "| columns:",list(df.columns))

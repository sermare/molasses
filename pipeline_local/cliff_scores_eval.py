#!/usr/bin/env python3
"""Per-compound scores for the WHOLE eval set (No-FT pipeline, 5-seed head-FT N=300, dataset Boltz-2 score, labels)
so the cliff notebook can ask how much of the top-of-ranking false-positive mass is cliff siblings."""
import glob, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT=Path("/global/scratch/users/sergiomar10/boltzaff"); T=sys.argv[1] if len(sys.argv)>1 else "588689"; R=ROOT/"results/runs"/T
def arm(d):
    f=pd.concat([pd.read_csv(p) for p in sorted(glob.glob(str(d)+"/chunk_*.csv"))],ignore_index=True); f["sample_id"]=f.sample_id.astype(str)
    return f.drop_duplicates("sample_id").set_index("sample_id").affinity_probability_binary
ev=[l.strip() for l in open(R/"ft_inputs_full/eval_ids.txt") if l.strip()]
d=pd.read_csv(ROOT/f"data/mf-pcba_test/{T}.csv"); d["id"]=f"{T}_"+d.CID.astype(str); d=d.set_index("id").reindex(ev)
out=pd.DataFrame({"id":ev,"label":d.target_active_v2.values,"score_boltz2":d.score_boltz2.values,"smiles":d["neut-smiles"].values})
out["base_noft"]=arm(R/"headft_affcache/base/scores").reindex(ev).values
out["ft300"]=np.mean([arm(R/f"headft_affcache/lightning_top300_seed{s}/scores").reindex(ev).values for s in range(5)],axis=0)
p=ROOT/"results/analysis"/T/"cliffs"/"scores_eval.csv"; out.to_csv(p,index=False); print("wrote",p,len(out),"rows; missing:",int(out.isna().sum().sum()))

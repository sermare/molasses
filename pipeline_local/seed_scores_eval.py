#!/usr/bin/env python3
"""Per-seed head-FT (N=300) and No-FT scores for the WHOLE eval set: probability of each of the 5 seeds, No-FT probability,
and the two-head disagreement |p_head1 - p_head2| for No-FT and for each seed. -> results/analysis/<T>/cliffs/scores_seeds_eval.csv"""
import glob, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT=Path("/global/scratch/users/sergiomar10/boltzaff"); T=sys.argv[1] if len(sys.argv)>1 else "588689"; R=ROOT/"results/runs"/T
def arm(d):
    f=pd.concat([pd.read_csv(p) for p in sorted(glob.glob(str(d)+"/chunk_*.csv"))],ignore_index=True); f["sample_id"]=f.sample_id.astype(str)
    return f.drop_duplicates("sample_id").set_index("sample_id")
ev=[l.strip() for l in open(R/"ft_inputs_full/eval_ids.txt") if l.strip()]
d=pd.read_csv(ROOT/f"data/mf-pcba_test/{T}.csv"); d["id"]=f"{T}_"+d.CID.astype(str); d=d.set_index("id").reindex(ev)
out=pd.DataFrame({"id":ev,"label":d.target_active_v2.astype(int).values})
b=arm(R/"headft_affcache/base/scores").reindex(ev)
out["noft_p"]=b.affinity_probability_binary.values; out["noft_dis"]=(b.affinity_probability_binary1-b.affinity_probability_binary2).abs().values
for s in range(5):
    a=arm(R/f"headft_affcache/lightning_top300_seed{s}/scores").reindex(ev)
    out[f"ft_p{s}"]=a.affinity_probability_binary.values; out[f"ft_dis{s}"]=(a.affinity_probability_binary1-a.affinity_probability_binary2).abs().values
p=ROOT/"results/analysis"/T/"cliffs"/"scores_seeds_eval.csv"; out.to_csv(p,index=False); print("wrote",p,out.shape,"missing:",int(out.isna().sum().sum()))

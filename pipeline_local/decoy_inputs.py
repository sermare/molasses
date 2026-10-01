#!/usr/bin/env python3
"""Decoy-protein rescore inputs for target 588689: does the Boltz-2 affinity head still separate actives from inactives
when the protein is NOT the true target? Subset = all held-out (eval) actives + random inactives. Three arms, identical protocol
(standard boltz2 co-fold + affinity, 5 affinity samples, same ligands):
  real     the true 588689 protein + its MSA (control; reproduces the standard pipeline)
  shuffled the 588689 sequence randomly permuted (same composition, no fold information), single-sequence (msa: empty)
  other    an unrelated real protein (target 493091, its own MSA)
Writes YAML chunk dirs under results/runs/588689/decoy/inputs/<arm>/chunk_XX and decoy/tasks.tsv (arm, chunk_dir, out_dir)."""
import random, yaml, sys
from pathlib import Path
import pandas as pd
ROOT=Path("/global/scratch/users/sergiomar10/boltzaff"); T="588689"; R=ROOT/"results/runs"/T; D=R/"decoy"
N_INACT=int(sys.argv[1]) if len(sys.argv)>1 else 1204; CHUNK=50
def seq_from_msa(p):
    with open(p) as f: f.readline(); return f.readline().split(",")[1].strip() if "," in f.readline() else None
# robust query-sequence read: boltz csv = 'key,sequence' header then first row is the query
def query_seq(p):
    import csv
    rows=list(csv.reader(open(p))); return rows[1][1]
real_seq=query_seq(ROOT/"data/588689_msa.csv"); other_msa=ROOT/"data/540297-493091_msa.csv"; other_seq=query_seq(other_msa)
rnd=random.Random(0); sh=list(real_seq); rnd.shuffle(sh); shuf_seq="".join(sh)
ev=[l.strip() for l in open(R/"ft_inputs_full/eval_ids.txt") if l.strip()]
d=pd.read_csv(ROOT/f"data/mf-pcba_test/{T}.csv"); d["id"]=f"{T}_"+d.CID.astype(str); d=d.set_index("id").reindex(ev)
act=d[d.target_active_v2==1]; ina=d[d.target_active_v2==0].sample(n=N_INACT,random_state=0); sub=pd.concat([act,ina]).sample(frac=1,random_state=1)
D.mkdir(parents=True,exist_ok=True)
sub[["CID","target_active_v2","neut-smiles"]].reset_index().rename(columns={"index":"id"}).to_csv(D/"subset.csv",index=False)
print(f"subset: {len(sub)} compounds ({len(act)} active, {len(ina)} inactive); real len {len(real_seq)}, other len {len(other_seq)}, shuffled len {len(shuf_seq)}")
ARMS={"real":(real_seq,str(ROOT/"data/588689_msa.csv")),"shuffled":(shuf_seq,"empty"),"other":(other_seq,str(other_msa))}
tasks=[]
for arm,(seq,msa) in ARMS.items():
    ids=list(sub.index)
    for c in range(0,len(ids),CHUNK):
        cd=D/"inputs"/arm/f"chunk_{c//CHUNK:02d}"; cd.mkdir(parents=True,exist_ok=True)
        for cid in ids[c:c+CHUNK]:
            doc={"version":1,"sequences":[{"protein":{"id":["A"],"sequence":seq,"msa":msa}},{"ligand":{"id":["B"],"smiles":str(sub.loc[cid,"neut-smiles"])}}],"properties":[{"affinity":{"binder":"B"}}]}
            (cd/f"{cid}.yaml").write_text(yaml.safe_dump(doc,sort_keys=False))
        tasks.append((arm,str(cd),str(D/"outputs"/arm/f"chunk_{c//CHUNK:02d}")))
pd.DataFrame(tasks).to_csv(D/"tasks.tsv",sep="\t",header=False,index=True)   # idx starts at 0 -> use idx+1 as array id
print(f"{len(tasks)} tasks ({len(tasks)//3} chunks per arm) -> {D/'tasks.tsv'}")

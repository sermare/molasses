#!/usr/bin/env python3
"""Run the (cached) affinity heads of one arm on the latent-analysis sample and save the internal features.
    python pipeline_local/extract_latent.py --target 434954-2097 --arm noft|ft300s0|ft300s1 [--limit 20]
Saves results/analysis/<t>/latent/<arm>.npz with, for each compound and each of the two ensemble modules:
  g      the 384-dim features after affinity_out_mlp (what the paper analyses), g_raw the 128-dim pooled pair representation before the MLP,
  value  affinity_pred_value, logit the binary affinity logit.  Resumable (partial file every 500 compounds)."""
import argparse, os, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "BoltzFT/pipeline")); sys.path.insert(0, str(ROOT / "Boltz2_affinity/src"))
import affinity_cached_infer as aci
from boltz.data.crop_embeddings import load_affinity_cropped_inputs

ap = argparse.ArgumentParser(); ap.add_argument("--target", required=True); ap.add_argument("--arm", required=True); ap.add_argument("--limit", type=int, default=0); a = ap.parse_args()
R = ROOT / "results/runs" / a.target; OUT = ROOT / "results/analysis" / a.target / "latent"; OUT.mkdir(parents=True, exist_ok=True)
ids = pd.read_csv(OUT / "sample.csv").id.tolist()
if a.limit: ids = ids[:a.limit]
ck = ROOT / "boltz_cache/boltz2_aff.ckpt" if a.arm == "noft" else R / f"headft_full/top300_seed{a.arm.split('s')[-1]}/checkpoints/last.ckpt"
assert ck.exists(), ck
dev = torch.device("cuda"); model = aci.CachedAffinityHeads(str(ck), use_kernels=False).eval().to(dev)
index = aci.cache_index(R / "outputs_affcache")
part = OUT / f"{a.arm}.part.npz"; res = {k: [] for k in ("g", "g_raw", "value", "logit")}; done = []
if part.exists() and not a.limit:
    z = np.load(part, allow_pickle=True); done = list(z["ids"]); res = {k: list(z[k]) for k in res}; print(f"resuming after {len(done)} compounds", flush=True)
have = set(done)
def save(path):
    np.savez_compressed(path, ids=np.array(done), **{k: np.stack(v) for k, v in res.items()})
for n, i in enumerate(ids):
    if i in have: continue
    c = aci._device_batch(load_affinity_cropped_inputs(Path(index[i])), dev)
    feats = {k: c[k] for k in ("token_to_rep_atom", "token_pad_mask", "mol_type", "affinity_token_mask")}
    kw = dict(s_inputs=c["s_inputs"], z=c["z"], x_pred=c["coords"], feats=feats, multiplicity=1, use_kernels=False)
    with torch.no_grad(), torch.autocast("cuda", enabled=False):
        outs = [model.affinity_module1(**kw), model.affinity_module2(**kw)]
    res["g"].append(np.stack([o["affinity_g_after_affinity_out_mlp"].float().cpu().numpy().reshape(-1) for o in outs]))
    res["g_raw"].append(np.stack([o["affinity_g_raw"].float().cpu().numpy().reshape(-1) for o in outs]))
    res["value"].append(np.array([o["affinity_pred_value"].item() for o in outs])); res["logit"].append(np.array([o["affinity_logits_binary"].item() for o in outs]))
    done.append(i)
    if len(done) % 500 == 0: save(part); print(f"{len(done)}/{len(ids)}", flush=True)
save(OUT / f"{a.arm}{'_test' if a.limit else ''}.npz"); print("done", len(done), "->", OUT / f"{a.arm}.npz"); part.unlink(missing_ok=True)

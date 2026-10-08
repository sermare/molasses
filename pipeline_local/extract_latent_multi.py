#!/usr/bin/env python3
"""All arms in ONE pass over the sample: each compound's cached input (about 6 MB, read from a slow shared filesystem) is loaded once, by prefetching threads, and run through
No-FT and head-FT seeds 0-2 on the GPU. The I/O, not the head, is the cost, so this is about as fast as one arm of extract_latent.py.
    python pipeline_local/extract_latent_multi.py --target 434954-2097 [--limit 100]
Writes results/analysis/<t>/latent/<arm>.npz for arm in noft, ft300s0, ft300s1, ft300s2 (same schema as extract_latent.py: ids, g (n,2,384), g_raw (n,2,128), value (n,2), logit (n,2)).
Resumable (partial file every 300 compounds)."""
import argparse, sys, time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd, torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "BoltzFT/pipeline")); sys.path.insert(0, str(ROOT / "Boltz2_affinity/src")); sys.path.insert(0, str(ROOT / "pipeline_local"))
import affinity_cached_infer as aci
from boltz.data.crop_embeddings import load_affinity_cropped_inputs

ap = argparse.ArgumentParser(); ap.add_argument("--target", required=True); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--threads", type=int, default=12); ap.add_argument("--device", default="cuda"); ap.add_argument("--torch-threads", type=int, default=0); ap.add_argument("--shard", type=int, default=-1); ap.add_argument("--nshards", type=int, default=0); a = ap.parse_args()
R = ROOT / "results/runs" / a.target; OUT = ROOT / "results/analysis" / a.target / "latent"; OUT.mkdir(parents=True, exist_ok=True)
ids = pd.read_csv(OUT / "sample.csv").id.tolist()
if a.limit: ids = ids[:a.limit]
SHARD = None
if a.shard >= 0:                                   # short-job mode: this task handles one contiguous block of the sample and writes shards/shard_XXX.npz (merged later by bft_latent.merge_shards)
    per = -(-len(ids) // a.nshards); ids = ids[a.shard * per:(a.shard + 1) * per]; SHARD = OUT / "shards" / f"shard_{a.shard:03d}.npz"; SHARD.parent.mkdir(exist_ok=True)
    if SHARD.exists(): print("shard already done"); sys.exit(0)
ARMS = {"noft": ROOT / "boltz_cache/boltz2_aff.ckpt", **{f"ft300s{s}": R / f"headft_full/top300_seed{s}/checkpoints/last.ckpt" for s in range(3)}}
for k, v in ARMS.items(): assert v.exists(), v
dev = torch.device(a.device)
if a.torch_threads: torch.set_num_threads(a.torch_threads)
models = {k: aci.CachedAffinityHeads(str(v), use_kernels=False).eval().to(dev) for k, v in ARMS.items()}
index = aci.cache_index(R / "outputs_affcache")
part = OUT / (f"multi.part{a.shard}.npz" if SHARD else "multi.part.npz"); KEYS = ("g", "g_raw", "value", "logit"); res = {arm: {k: [] for k in KEYS} for arm in ARMS}; done = []
if part.exists() and not a.limit:
    z = np.load(part, allow_pickle=True); done = list(z["ids"]); res = {arm: {k: list(z[f"{arm}__{k}"]) for k in KEYS} for arm in ARMS}; print(f"resuming after {len(done)} compounds", flush=True)
have = set(done); todo = [i for i in ids if i not in have]
def save(path_for):
    for arm in ARMS: np.savez_compressed(path_for(arm), ids=np.array(done), **{k: np.stack(v) for k, v in res[arm].items()})
def save_part():
    np.savez_compressed(part, ids=np.array(done), **{f"{arm}__{k}": np.stack(v) for arm in ARMS for k, v in res[arm].items()})
def load(i): return i, load_affinity_cropped_inputs(Path(index[i]))
t0 = time.time(); ex = ThreadPoolExecutor(a.threads); q = deque(); it = iter(todo)
for _ in range(a.threads * 3):
    i = next(it, None)
    if i is None: break
    q.append(ex.submit(load, i))
while q:
    i, cb = q.popleft().result(); nxt = next(it, None)
    if nxt is not None: q.append(ex.submit(load, nxt))
    c = aci._device_batch(cb, dev); feats = {k: c[k] for k in ("token_to_rep_atom", "token_pad_mask", "mol_type", "affinity_token_mask")}
    kw = dict(s_inputs=c["s_inputs"], z=c["z"], x_pred=c["coords"], feats=feats, multiplicity=1, use_kernels=False)
    with torch.no_grad(), torch.autocast("cuda", enabled=False):
        for arm, m in models.items():
            outs = [m.affinity_module1(**kw), m.affinity_module2(**kw)]
            res[arm]["g"].append(np.stack([o["affinity_g_after_affinity_out_mlp"].float().cpu().numpy().reshape(-1) for o in outs]))
            res[arm]["g_raw"].append(np.stack([o["affinity_g_raw"].float().cpu().numpy().reshape(-1) for o in outs]))
            res[arm]["value"].append(np.array([o["affinity_pred_value"].item() for o in outs])); res[arm]["logit"].append(np.array([o["affinity_logits_binary"].item() for o in outs]))
    done.append(i)
    if len(done) % 300 == 0: save_part(); print(f"{len(done)}/{len(ids)}  {(time.time() - t0) / max(len(done) - len(have), 1):.2f} s per compound", flush=True)
if SHARD:
    tmp = SHARD.with_suffix(".tmp.npz"); np.savez_compressed(tmp, ids=np.array(done), **{f"{arm}__{k}": np.stack(v) for arm in ARMS for k, v in res[arm].items()}); tmp.rename(SHARD)
else: save(lambda arm: OUT / f"{arm}{'_test' if a.limit else ''}.npz")
part.unlink(missing_ok=True); print(f"done {len(done)} compounds in {(time.time() - t0) / 60:.1f} min", flush=True)

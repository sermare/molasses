#!/usr/bin/env python3
"""Docking site of every compound in the latent-analysis sample: align each Boltz-2 pose's protein CA atoms to the target's reference frame (results/runs/<t>/pose_density/ca_colored.npz),
take the ligand heavy-atom centroid in that frame, and cluster the centroids with k-means (k = 2..5 chosen by silhouette; sites are numbered by size, 1 = largest).
    python pipeline_local/pose_sites.py <target>      -> results/analysis/<t>/latent/pose_sites.csv  (id, x, y, z, site, dist_to_median_centroid)
Reuses the Kabsch / CIF parsing of pose_centroids.py. A pose whose protein has a different number of residues than the reference is skipped (rare)."""
import sys, os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import pose_centroids as pc, bft_common as bc, bft_latent as lat


def cif_for(t, chunk_of, i):
    ch = chunk_of[i]; return bc.RUNS / t / "outputs_chunks_full" / ch / f"boltz_results_{ch}" / "predictions" / i / f"{i}_model_0.cif"


def run(t):
    D = lat.make_sample(t); R = bc.RUNS / t; ref = np.load(R / "pose_density/ca_colored.npz")["ca"]; INP = R / "inputs_chunks_full"; chunk_of = {}
    for ch in sorted(c for c in os.listdir(INP) if len(c) == 9 and c.startswith("chunk_")):
        for y in os.listdir(INP / ch):
            if y.endswith(".yaml"): chunk_of[y[:-5]] = ch
    def one(i):
        try:
            P, L = pc.coords(str(cif_for(t, chunk_of, i)))
            if len(P) != len(ref) or len(L) == 0: return i, None
            Rm, tr = pc.kabsch(P, ref); La = (Rm @ L.T).T + tr; return i, La.mean(0)
        except Exception: return i, None
    with ThreadPoolExecutor(32) as ex: res = list(ex.map(one, D.id.tolist()))
    C = pd.DataFrame([dict(id=i, x=c[0], y=c[1], z=c[2]) for i, c in res if c is not None])
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    X = C[["x", "y", "z"]].values; best = None
    for k in range(2, 6):
        km = KMeans(k, n_init=10, random_state=0).fit(X); sc = silhouette_score(X, km.labels_, sample_size=min(len(X), 3000), random_state=0)
        if best is None or sc > best[0]: best = (sc, k, km.labels_)
    sc, k, lab = best; order = pd.Series(lab).value_counts().index.tolist(); remap = {o: n + 1 for n, o in enumerate(order)}
    C["site"] = [remap[l] for l in lab]; C["dist_to_median_centroid"] = np.linalg.norm(X - np.median(X, 0), axis=1)
    out = bc.AN / t / "latent" / "pose_sites.csv"; C.to_csv(out, index=False)
    print(f"{t}: {len(C)} of {len(D)} poses placed; k={k} (silhouette {sc:.2f}); site sizes {C.site.value_counts().sort_index().to_dict()}; spread of centroids (A, std x/y/z) {X.std(0).round(1).tolist()}")


if __name__ == "__main__":
    run(sys.argv[1])

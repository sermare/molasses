"""Structure-confidence helpers for 07_structure_confidence: per-residue pLDDT sampled over the library's Boltz-2 poses, confidence JSON metrics,
and secondary structure (DSSP via PyMOL) of the reference structure. Everything is cached under results/analysis/<t>/ and returns None when a target has no poses yet."""
import json, random, subprocess, glob, os
from pathlib import Path
import numpy as np, pandas as pd
import sys; sys.path.insert(0, "/global/scratch/users/sergiomar10/boltzaff/pipeline_local")
from bft_common import ROOT, RUNS, AN, TARGETS, SHORT

PYMOL = "/clusterfs/nilah/sergio/miniconda3/envs/pymolenv/bin/python"


def n_residues(t):
    p = RUNS / t / "pose_density/residue_density.csv"
    return len(pd.read_csv(p)) if p.exists() else None


def sample_poses(t, n=400, seed=0, per_chunk=10):
    """Random sample of up to n predictions across chunks: returns dict(plddt=[n, n_res], lig_plddt=[n], metrics=DataFrame) or None."""
    cp = AN / t / "structure_conf.npz"; mp = AN / t / "structure_conf_metrics.csv"
    nres = n_residues(t)
    if nres is None: return None
    if cp.exists() and mp.exists():
        z = np.load(cp); m = pd.read_csv(mp)
        if z["plddt"].shape[1] == nres: return dict(plddt=z["plddt"], lig_plddt=z["lig_plddt"], metrics=m)
    chunks = sorted(glob.glob(str(RUNS / t / "outputs_chunks_full/chunk_[0-9][0-9][0-9]")))
    if not chunks: return None
    rng = random.Random(seed); rng.shuffle(chunks)
    pl, lig, rows = [], [], []
    for c in chunks:
        pd_ = glob.glob(c + "/boltz_results_*/predictions")
        if not pd_: continue
        ids = os.listdir(pd_[0]); rng.shuffle(ids)
        got = 0
        for i in ids:
            f = f"{pd_[0]}/{i}/plddt_{i}_model_0.npz"; j = f"{pd_[0]}/{i}/confidence_{i}_model_0.json"
            if not (os.path.exists(f) and os.path.exists(j)): continue
            try:
                a = np.load(f)["plddt"]; d = json.load(open(j))
            except Exception: continue
            if len(a) <= nres: continue
            pl.append(a[:nres]); lig.append(float(a[nres:].mean()))
            rows.append(dict(id=i, **{k: d[k] for k in ("confidence_score", "ptm", "iptm", "ligand_iptm", "complex_plddt", "complex_pde")}))
            got += 1
            if got >= per_chunk: break
        if len(pl) >= n: break
    if not pl: return None
    P = np.array(pl); L = np.array(lig); M = pd.DataFrame(rows)
    cp.parent.mkdir(parents=True, exist_ok=True); np.savez(cp, plddt=P, lig_plddt=L); M.to_csv(mp, index=False)
    return dict(plddt=P, lig_plddt=L, metrics=M)


def dssp(t):
    """Per-residue secondary structure string (H helix, S sheet, L loop) of the reference structure, via PyMOL dss. Cached."""
    cp = AN / t / "dssp.txt"; pdb = RUNS / t / "pose_density/density_colored.pdb"
    if not pdb.exists(): return None
    if cp.exists() and cp.stat().st_mtime >= pdb.stat().st_mtime: return cp.read_text().strip()
    code = ("import pymol; pymol.finish_launching(['pymol','-qc']); from pymol import cmd\n"
            f"cmd.load('{pdb}','m'); cmd.select('p','m and polymer and name CA'); cmd.dss('p')\n"
            "ss=''.join(a.ss if a.ss in ('H','S') else 'L' for a in cmd.get_model('p').atom)\n"
            "print('SSOUT ' + ss, flush=True)\n")
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh: fh.write(code); script = fh.name
    r = subprocess.run([PYMOL, script], capture_output=True, text=True, timeout=600); os.unlink(script)
    s = [l[6:] for l in r.stdout.splitlines() if l.startswith("SSOUT ")]
    if not s: return None
    cp.parent.mkdir(parents=True, exist_ok=True); cp.write_text(s[-1]); return s[-1]


def write_plddt_pdb(t, S):
    """density_colored.pdb with the B-factor replaced by the sample-mean per-residue pLDDT (0-100) for the protein chain: results/analysis/<t>/plddt_colored.pdb."""
    src = RUNS / t / "pose_density/density_colored.pdb"; out = AN / t / "plddt_colored.pdb"
    if not src.exists() or S is None: return None
    mean = S["plddt"].mean(axis=0) * 100; lines = []; seen = {}
    for l in src.read_text().splitlines():
        if l.startswith("ATOM") and l[21] == "A":
            key = (l[22:27]); seen.setdefault(key, len(seen)); k = seen[key]
            l = l[:60] + f"{mean[k] if k < len(mean) else 0:6.2f}" + l[66:]
        lines.append(l)
    out.write_text("\n".join(lines) + "\n"); return out

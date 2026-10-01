#!/usr/bin/env python3
"""Ray-traced PyMOL renders of the pose-density map on the real protein (run with the pymolenv python, headless):

    /clusterfs/nilah/sergio/miniconda3/envs/pymolenv/bin/python render_density_pymol.py <target> [outdir]

Input : results/runs/<t>/pose_density/density_colored.pdb  (the reference Boltz-2 prediction: protein chain A with B-factor = percent of the library's
                                                            poses whose ligand contacts that residue (0-100); chain B 'LIG' = the ligand of that ONE
                                                            reference prediction, a single compound's pose, shown only to mark the pocket)
        results/runs/<t>/pose_density/residue_density.csv
Output: <outdir>/{overview,pocket,back}.png  (default results/analysis/<t>/render/), 1400x1050, ray traced, white background.
  overview : cartoon + surface coloured by density (white = no contact, red = most contacted), reference ligand as sticks
  pocket   : zoom on the pocket, the 10 most contacted residues as sticks with labels
  back     : the same surface seen from the opposite side
  cutaway  : same camera as overview with the protein in front of the ligand clipped away (shows buried pockets)
A render is skipped when its png is newer than the input (self-filling)."""
import sys, os, csv
from pathlib import Path
ROOT = Path("/global/scratch/users/sergiomar10/boltzaff")
t = sys.argv[1]
out = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "results/analysis" / t / "render"
pdb = ROOT / "results/runs" / t / "pose_density/density_colored.pdb"; rd = ROOT / "results/runs" / t / "pose_density/residue_density.csv"
if not pdb.exists(): sys.exit(f"pending: {pdb} not found")
out.mkdir(parents=True, exist_ok=True)
names = ["overview", "pocket", "back", "cutaway"]
need = {n: not ((out / f"{n}.png").exists() and (out / f"{n}.png").stat().st_mtime > pdb.stat().st_mtime) for n in names}
if not any(need.values()): print("up to date:", out); sys.exit(0)

import pymol
pymol.pymol_argv = ["pymol", "-qc"]; pymol.finish_launching(["pymol", "-qc"])
from pymol import cmd
cmd.reinitialize(); cmd.bg_color("white"); cmd.load(str(pdb), "mol")
cmd.select("prot", "mol and polymer"); cmd.select("lig", "mol and resn LIG")
# density colouring from the B-factor column
bmax = max(a.b for a in cmd.get_model("prot").atom) or 1.0
cmd.spectrum("b", "white_yellow_red", "prot", minimum=0, maximum=0.6 * bmax)   # saturate so moderate contact is visible
cmd.hide("everything"); cmd.show("cartoon", "prot"); cmd.set("cartoon_fancy_helices", 1); cmd.set("cartoon_smooth_loops", 1)
cmd.show("sticks", "lig"); cmd.color("tv_green", "lig and elem C"); cmd.set("stick_radius", 0.22, "lig")
cmd.set("max_threads", 4); cmd.set("ray_opaque_background", 1); cmd.set("antialias", 2); cmd.set("ray_trace_mode", 1); cmd.set("ray_trace_gain", 0.4)
cmd.set("ambient", 0.45); cmd.set("specular", 0.25); cmd.set("depth_cue", 0); cmd.set("ray_shadow", 1); cmd.set("light_count", 2)
cmd.set("surface_quality", 1); cmd.set("transparency", 0.25); cmd.set("cartoon_transparency", 0.0)
W, H = 1400, 1050

# --- overview: surface coloured by density + cartoon, ligand sticks
cmd.show("surface", "prot"); cmd.set("surface_color", "default")
def face_pocket():
    """Look at the protein from the pocket side: the camera z axis is the vector protein centre -> reference ligand
    (PyMOL view layout: camera coordinate k of a model point x is sum_i v[3i+k] * x[i])."""
    import numpy as np
    cmd.orient("prot"); v = list(cmd.get_view())
    com = np.array(cmd.centerofmass("prot")); lg = np.array(cmd.centerofmass("lig")); d = lg - com; d /= np.linalg.norm(d)
    y0 = np.array([v[1], v[4], v[7]]); e1 = y0 - d * (y0 @ d); e1 /= np.linalg.norm(e1); e0 = np.cross(e1, d)
    for i in range(3): v[3 * i], v[3 * i + 1], v[3 * i + 2] = e0[i], e1[i], d[i]
    v[12:15] = list(com); cmd.set_view(v)
face_pocket()
if need["overview"]: cmd.png(str(out / "overview.png"), width=W, height=H, dpi=150, ray=1)
# --- back view
cmd.turn("y", 180)
if need["back"]: cmd.png(str(out / "back.png"), width=W, height=H, dpi=150, ray=1)
cmd.turn("y", 180)

# --- cutaway: clip away everything in front of the ligand plane
if need["cutaway"]:
    cmd.set("two_sided_lighting", 1); cmd.set("ray_shadow", 0); cmd.set("backface_cull", 0); cmd.clip("atoms", 1.0, "lig"); cmd.clip("far", -60); cmd.set("transparency", 0.0); cmd.set("cartoon_transparency", 0.0)
    cmd.zoom("lig", 22, clip=0) if False else None
    cmd.png(str(out / "cutaway.png"), width=W, height=H, dpi=150, ray=1)
    cmd.clip("slab", 400)       # reset the slab for the pocket view
    cmd.set("transparency", 0.25)

# --- pocket: the most contacted residues as sticks, labelled
top = []
if rd.exists():
    rows = list(csv.DictReader(open(rd)))
    rows.sort(key=lambda r: -float(r["soft_density"])); top = rows[:10]
sel = " or ".join(f"(mol and polymer and resi {r['res_num']})" for r in top) or "none"
cmd.select("hot", sel)
cmd.create("hotsticks", "hot"); cmd.hide("everything", "hotsticks"); cmd.show("sticks", "hotsticks"); cmd.set("stick_radius", 0.16, "hotsticks")
cmd.color("tv_orange", "hotsticks and elem C"); cmd.color("blue", "hotsticks and elem N"); cmd.color("red", "hotsticks and elem O"); cmd.color("yellow", "hotsticks and elem S")
cmd.set("label_size", 18); cmd.set("label_color", "black"); cmd.set("label_font_id", 5); cmd.set("label_outline_color", "white"); cmd.set("label_position", (0, 4.5, 7))
for r in top[:6]: cmd.label(f"mol and polymer and resi {r['res_num']} and name CA", f"'{r['res_name'].capitalize()}{r['res_num']}'")
cmd.set("transparency", 0.8); cmd.set("cartoon_transparency", 0.0); cmd.set("cartoon_tube_radius", 0.35); cmd.set("cartoon_loop_radius", 0.3)
cmd.show("cartoon", "prot"); cmd.set("stick_radius", 0.2, "lig")
cmd.zoom("lig or hot", 3); cmd.clip("slab", 40, "lig")
if need["pocket"]: cmd.png(str(out / "pocket.png"), width=W, height=H, dpi=150, ray=1)
cmd.quit()
print("rendered", t, "->", out)

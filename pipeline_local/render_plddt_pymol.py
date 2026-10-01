#!/usr/bin/env python3
"""Ray-traced PyMOL render of the reference structure coloured by Boltz-2 pLDDT (AlphaFold palette), run with the pymolenv python:
    pymolenv/bin/python render_plddt_pymol.py <target>
Input : results/analysis/<t>/plddt_colored.pdb (B-factor = sample-mean per-residue pLDDT, 0-100; written by bft_structure.write_plddt_pdb)
Output: results/analysis/<t>/render/{plddt,plddt_side}.png.  Palette: >90 dark blue, 70-90 light blue, 50-70 yellow, <50 orange."""
import sys
from pathlib import Path
ROOT = Path("/global/scratch/users/sergiomar10/boltzaff"); t = sys.argv[1]
pdb = ROOT / "results/analysis" / t / "plddt_colored.pdb"; out = ROOT / "results/analysis" / t / "render"
if not pdb.exists(): sys.exit(f"pending: {pdb} not found")
out.mkdir(parents=True, exist_ok=True)
if all((out / f"{n}.png").exists() and (out / f"{n}.png").stat().st_mtime > pdb.stat().st_mtime for n in ("plddt", "plddt_side")): print("up to date"); sys.exit(0)
import numpy as np, pymol
pymol.finish_launching(["pymol", "-qc"]); from pymol import cmd
cmd.bg_color("white"); cmd.load(str(pdb), "m"); cmd.select("prot", "m and polymer"); cmd.select("lig", "m and resn LIG")
cmd.set_color("pl_vh", [0 / 255, 83 / 255, 214 / 255]); cmd.set_color("pl_h", [101 / 255, 203 / 255, 243 / 255]); cmd.set_color("pl_l", [255 / 255, 219 / 255, 19 / 255]); cmd.set_color("pl_vl", [255 / 255, 125 / 255, 69 / 255])
cmd.color("pl_vh", "prot and b > 90"); cmd.color("pl_h", "prot and b < 90 and b > 70"); cmd.color("pl_l", "prot and b < 70 and b > 50"); cmd.color("pl_vl", "prot and b < 50")
cmd.hide("everything"); cmd.show("cartoon", "prot"); cmd.show("sticks", "lig"); cmd.color("gray40", "lig and elem C"); cmd.set("stick_radius", 0.22, "lig")
cmd.set("cartoon_fancy_helices", 1); cmd.set("cartoon_smooth_loops", 1); cmd.set("cartoon_loop_radius", 0.3)
for k, v in dict(max_threads=4, ray_opaque_background=1, antialias=2, ray_trace_mode=1, ray_trace_gain=0.4, ambient=0.45, specular=0.25, depth_cue=0, ray_shadow=1).items(): cmd.set(k, v)
W, H = 1400, 1050
cmd.orient("prot"); cmd.png(str(out / "plddt.png"), width=W, height=H, dpi=150, ray=1)
cmd.turn("y", 90); cmd.png(str(out / "plddt_side.png"), width=W, height=H, dpi=150, ray=1)
cmd.quit(); print("rendered pLDDT", t)

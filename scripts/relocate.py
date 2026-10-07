#!/usr/bin/env python3
"""Adapt a fresh clone of this repository to your own machine and Slurm site. Standard library only.

The code was developed on one cluster (UC Berkeley Savio) and refers to that cluster's paths and Slurm settings in many files. Run this once
after cloning; it rewrites those references in the tracked text files (shell, Slurm, Python, notebook CODE cells; never notebook outputs,
the README, the licence or the patches):

    python scripts/relocate.py --conda-sh ~/miniconda3/etc/profile.d/conda.sh \\
        --account myproject --gpu-partition gpu --cpu-partition cpu --qos normal --exclude ""      # or --exclude node1,node2
    python scripts/relocate.py --check           # exit 1 if any original site setting is still present
    python scripts/relocate.py ... --dry-run     # report what would change

What is replaced (the values were all hard-coded in the original):
  /global/scratch/users/sergiomar10/boltzaff        -> --root (default: the directory that contains this scripts/ folder)
  /clusterfs/nilah/sergio/miniconda3[/etc/...]     -> the conda base / --conda-sh
  /clusterfs/nilah/sergio/RBX1/weights             -> <root>/boltz_cache
  co_nilah, savio3_gpu, savio3_htc, savio_lowprio  -> --account, --gpu-partition, --cpu-partition, --qos
  --exclude node lists (bad nodes on that cluster) -> --exclude (empty removes the option)
Submit Slurm jobs from the repository root: log paths in the #SBATCH headers are absolute after relocation.
"""
import argparse, json, os, re, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
ORIG = dict(root="/global/scratch/users/sergiomar10/boltzaff", conda_sh="/clusterfs/nilah/sergio/miniconda3/etc/profile.d/conda.sh",
            conda_base="/clusterfs/nilah/sergio/miniconda3", weights="/clusterfs/nilah/sergio/RBX1/weights", account="co_nilah", gpu="savio3_gpu",
            cpu="savio3_htc", qos="savio_lowprio")
EXCLUDES = ["n0143.savio3,n0144.savio3,n0215.savio3,n0176.savio3", "n0143.savio3,n0144.savio3,n0215.savio3", "n0143.savio3,n0144.savio3"]
TEXT_SUFFIX = {".sh", ".sbatch", ".py", ".ipynb", ".yml", ".yaml", ".cfg", ".txt", ""}
SKIP_NAMES = {"README.md", "LICENSE"}
SKIP_DIRS = {"patches", "envs", "inputs", "tests", ".github"}      # tests/ reference the original values on purpose; envs/ and inputs/ are data


def tracked(root: Path):
    try:
        out = subprocess.run(["git", "ls-files"], cwd=root, capture_output=True, text=True, check=True).stdout.split("\n")
        return [root / f for f in out if f]
    except Exception:
        return [p for p in root.rglob("*") if p.is_file() and ".git" not in p.parts]


def wanted(p: Path, root: Path) -> bool:
    rel = p.relative_to(root)
    if not p.is_file() or p.suffix not in TEXT_SUFFIX or p.name in SKIP_NAMES: return False
    return rel.parts[0] not in SKIP_DIRS and rel.as_posix() != "scripts/relocate.py"


def build_rules(a):
    base = str(Path(a.conda_sh).resolve().parents[2]) if a.conda_sh else None
    rules = [(ORIG["conda_sh"], a.conda_sh), (ORIG["conda_base"], base), (ORIG["weights"], f"{a.root}/boltz_cache"), (ORIG["root"], a.root)]
    rules += [(ORIG["account"], a.account), (ORIG["gpu"], a.gpu_partition), (ORIG["cpu"], a.cpu_partition), (ORIG["qos"], a.qos)]
    return [(o, n) for o, n in rules if n is not None], a.exclude


def rewrite(text: str, rules, exclude):
    n = 0
    for lst in EXCLUDES if exclude is not None else []:             # node exclusions first; None leaves them untouched
        if lst in text:
            if exclude:
                c = text.count(lst); text = text.replace(lst, exclude); n += c
            else:
                text, c = re.subn(r"^#SBATCH --exclude=" + re.escape(lst) + r"[^\n]*\n", "", text, flags=re.M); n += c
                text, c = re.subn(r"[ \t]*--exclude=" + re.escape(lst) + r"[ \t]*(\\\n)?", " ", text); n += c
    for o, nw in rules:
        if o in text: n += text.count(o); text = text.replace(o, nw)
    return text, n


def process_file(p: Path, rules, exclude, write: bool):
    if p.suffix == ".ipynb":
        nb = json.loads(p.read_text()); n = 0
        for c in nb.get("cells", []):
            if c.get("cell_type") != "code": continue
            src = "".join(c["source"]); new, k = rewrite(src, rules, exclude)
            if k: c["source"] = new.splitlines(keepends=True); n += k
        if n and write: p.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
        return n
    try: text = p.read_text()
    except UnicodeDecodeError: return 0
    new, n = rewrite(text, rules, exclude)
    if n and write: p.write_text(new)
    return n


def remaining(root: Path):
    """Original site settings still present in the files this script would process."""
    pats = [ORIG[k] for k in ("root", "conda_base", "weights", "account", "gpu", "cpu", "qos")] + EXCLUDES[:1]
    out = {}
    for p in tracked(root):
        if not wanted(p, root): continue
        if p.suffix == ".ipynb":
            nb = json.loads(p.read_text()); text = "\n".join("".join(c["source"]) for c in nb.get("cells", []) if c.get("cell_type") == "code")
        else:
            try: text = p.read_text()
            except UnicodeDecodeError: continue
        hit = [x for x in pats if x in text]
        if hit: out[str(p.relative_to(root))] = hit
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", type=Path, default=HERE, help="the checkout to rewrite (default: this one)")
    ap.add_argument("--root", default=None, help="absolute path the checkout will live at (default: --repo)")
    ap.add_argument("--conda-sh", default=None, help="path to <conda base>/etc/profile.d/conda.sh")
    ap.add_argument("--account", default=None); ap.add_argument("--gpu-partition", default=None); ap.add_argument("--cpu-partition", default=None)
    ap.add_argument("--qos", default=None); ap.add_argument("--exclude", default=None, help="nodes to exclude, comma separated; '' removes the option")
    ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--check", action="store_true")
    a = ap.parse_args(); repo = a.repo.resolve(); a.root = str(Path(a.root).resolve()) if a.root else str(repo)
    if a.check:
        left = remaining(repo)
        for f, h in left.items(): print(f"{f}: {', '.join(h)}")
        print(f"{len(left)} file(s) still contain the original site settings"); sys.exit(1 if left else 0)
    rules, exclude = build_rules(a); total = files = 0
    for p in tracked(repo):
        if not wanted(p, repo): continue
        n = process_file(p, rules, exclude, write=not a.dry_run)
        if n: files += 1; total += n
    print(f"{'would change' if a.dry_run else 'changed'} {total} reference(s) in {files} file(s) under {repo}")


if __name__ == "__main__":
    main()

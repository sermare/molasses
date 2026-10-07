"""scripts/relocate.py: adapting a clone to another machine and Slurm site must leave no original site setting behind and break nothing."""
import json, re, shutil, subprocess, sys
import pytest

ORIG_TOKENS = ["/global/scratch/users/sergiomar10/boltzaff", "/clusterfs/nilah/sergio/miniconda3", "co_nilah", "savio3_gpu", "savio3_htc", "savio_lowprio"]


@pytest.fixture(scope="module")
def clone(repo, tracked_files, tmp_path_factory):
    """A copy of the code (not data) of this repository in a temp folder."""
    dst = tmp_path_factory.mktemp("clone")
    for f in tracked_files:
        rel = f.relative_to(repo)
        if rel.parts[0] in ("slurm", "pipeline_local", "scripts", "tests") or rel.as_posix() in ("env.sh",) or (rel.parts[0] == "notebooks" and f.suffix in (".ipynb", ".py") and len(rel.parts) == 2):
            (dst / rel).parent.mkdir(parents=True, exist_ok=True); shutil.copy2(f, dst / rel)
    (dst / "scripts").mkdir(exist_ok=True)
    for f in (repo / "scripts").glob("*"):                                       # always include the tool under test, tracked or not
        if f.is_file(): shutil.copy2(f, dst / "scripts" / f.name)
    return dst


def run(clone, *args):
    return subprocess.run([sys.executable, str(clone / "scripts/relocate.py"), "--repo", str(clone), *args], capture_output=True, text=True)


def test_original_repo_reports_the_site_settings(clone):
    r = run(clone, "--check"); assert r.returncode == 1 and "slurm/" in r.stdout


def test_dry_run_changes_nothing(clone):
    before = (clone / "slurm/pass1_embed.sbatch").read_text()
    r = run(clone, "--dry-run", "--account", "x"); assert r.returncode == 0 and "would change" in r.stdout
    assert (clone / "slurm/pass1_embed.sbatch").read_text() == before


def test_relocation_removes_every_original_setting_and_breaks_nothing(clone):
    nb = clone / "notebooks/02_uncertainty.ipynb"; outputs_before = [c.get("outputs") for c in json.loads(nb.read_text())["cells"]] if nb.exists() else None
    r = run(clone, "--root", "/opt/boltzaff", "--conda-sh", "/opt/conda/etc/profile.d/conda.sh", "--account", "myproj", "--gpu-partition", "gpu", "--cpu-partition", "cpu", "--qos", "normal", "--exclude", "")
    assert r.returncode == 0, r.stderr
    assert run(clone, "--check").returncode == 0                                    # nothing original left in the processed files
    p1 = (clone / "slurm/pass1_embed.sbatch").read_text()
    assert "#SBATCH --account=myproj" in p1 and "#SBATCH --partition=gpu" in p1 and "#SBATCH --qos=normal" in p1
    assert "--exclude" not in p1 and "/opt/boltzaff/slurm/logs/" in p1              # empty --exclude removes the option; log paths follow --root
    assert "source /opt/conda/etc/profile.d/conda.sh" in p1
    for f in clone.rglob("*"):
        if f.suffix == ".py" and "tests" not in f.parts: compile(f.read_text(), str(f), "exec")
        if f.suffix in (".sh", ".sbatch"): assert subprocess.run(["bash", "-n", str(f)], capture_output=True).returncode == 0, f.name
        if f.suffix == ".ipynb": json.loads(f.read_text())
    if outputs_before is not None:                                                  # notebook outputs are never rewritten
        assert [c.get("outputs") for c in json.loads(nb.read_text())["cells"]] == outputs_before
    for f in ["env.sh"]: assert "/opt/boltzaff" in (clone / f).read_text()


def test_relocation_is_idempotent(clone):
    snap = {f: f.read_bytes() for f in clone.rglob("*") if f.is_file() and f.suffix in (".sh", ".sbatch", ".py")}
    r = run(clone, "--root", "/opt/boltzaff", "--conda-sh", "/opt/conda/etc/profile.d/conda.sh", "--account", "myproj", "--gpu-partition", "gpu", "--cpu-partition", "cpu", "--qos", "normal", "--exclude", "")
    assert r.returncode == 0 and all(f.read_bytes() == b for f, b in snap.items())


def test_custom_exclude_list_replaces_the_original(repo, tracked_files, tmp_path):
    f = repo / "slurm/pass1_embed.sbatch"; dst = tmp_path / "slurm"; dst.mkdir(); shutil.copy2(f, dst / f.name)
    subprocess.run([sys.executable, str(repo / "scripts/relocate.py"), "--repo", str(tmp_path), "--exclude", "badnode1,badnode2"], check=True, capture_output=True)
    t = (dst / f.name).read_text(); assert "#SBATCH --exclude=badnode1,badnode2" in t and "n0176" not in t

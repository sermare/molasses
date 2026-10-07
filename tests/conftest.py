"""Shared test setup. The tests run on any machine (no GPU, no Slurm, no data):
- pure statistics and file-hygiene tests need only numpy/pandas/scipy/scikit-learn/nbformat;
- tests marked `needs_boltzft` also need a checkout of ohuelab/BoltzFT (set BOLTZFT=/path/to/BoltzFT, or clone it next to this repo as ./BoltzFT);
  they are skipped, not failed, when it is missing."""
import os, sys, subprocess
from pathlib import Path
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "pipeline_local"))
_bf = os.environ.get("BOLTZFT") or (str(REPO / "BoltzFT") if (REPO / "BoltzFT" / "evaluation" / "metrics.py").exists() else "")
HAVE_BOLTZFT = bool(_bf) and (Path(_bf) / "evaluation" / "metrics.py").exists()
if HAVE_BOLTZFT:
    os.environ["BOLTZFT"] = _bf
    os.environ.setdefault("WORKDIR", str(REPO / "results"))


def pytest_configure(config):
    config.addinivalue_line("markers", "needs_boltzft: needs a checkout of ohuelab/BoltzFT (BOLTZFT env var)")


def pytest_collection_modifyitems(config, items):
    if HAVE_BOLTZFT: return
    skip = pytest.mark.skip(reason="BoltzFT checkout not found (set BOLTZFT or clone ohuelab/BoltzFT into ./BoltzFT)")
    for it in items:
        if "needs_boltzft" in it.keywords: it.add_marker(skip)


@pytest.fixture(scope="session")
def repo():
    return REPO


@pytest.fixture(scope="session")
def tracked_files(repo):
    """Files tracked by git (falls back to walking the tree when this is not a git checkout)."""
    try:
        out = subprocess.run(["git", "ls-files"], cwd=repo, capture_output=True, text=True, check=True).stdout.split("\n")
        return [repo / f for f in out if f]
    except Exception:
        return [p for p in repo.rglob("*") if p.is_file() and ".git" not in p.parts]

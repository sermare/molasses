"""Cheap guards that keep the repository runnable and tidy: everything compiles, notebooks are valid, no HTML, README figures exist."""
import json, re, subprocess
import pytest


def test_all_python_files_compile(tracked_files):
    bad = []
    for f in tracked_files:
        if f.suffix == ".py":
            try: compile(f.read_text(), str(f), "exec")
            except SyntaxError as e: bad.append(f"{f.name}: {e.msg} (line {e.lineno})")
    assert not bad, bad


def test_shell_scripts_parse(tracked_files):
    bad = []
    for f in tracked_files:
        if f.suffix in (".sh", ".sbatch"):
            r = subprocess.run(["bash", "-n", str(f)], capture_output=True, text=True)
            if r.returncode: bad.append(f"{f.name}: {r.stderr.strip()[:80]}")
    assert not bad, bad


def test_notebooks_are_valid_nbformat(tracked_files):
    nbformat = pytest.importorskip("nbformat")
    for f in tracked_files:
        if f.suffix == ".ipynb":
            nb = nbformat.read(str(f), as_version=4); nbformat.validate(nb)


def test_notebooks_have_no_stored_errors(tracked_files):
    for f in tracked_files:
        if f.suffix == ".ipynb" and f.parent.name == "notebooks":
            nb = json.loads(f.read_text())
            errs = [i for i, c in enumerate(nb["cells"]) for o in c.get("outputs", []) if o.get("output_type") == "error"]
            assert not errs, f"{f.name}: error output in cells {errs}"


def test_no_html_deliverables(tracked_files):
    assert not [f.name for f in tracked_files if f.suffix in (".html", ".htm")]       # project rule: notebooks only, never HTML


def test_no_huge_files(tracked_files):
    big = [(f.name, f.stat().st_size) for f in tracked_files if f.exists() and f.stat().st_size > 12_000_000]
    assert not big, big


def test_readme_images_and_links_exist(repo):
    text = (repo / "README.md").read_text()
    for rel in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text) + re.findall(r"\]\((notebooks/[^)#]+)\)", text):
        if rel.startswith("http"): continue
        assert (repo / rel).exists(), f"README references missing file {rel}"


def test_readme_progress_block_is_intact(repo):
    t = (repo / "README.md").read_text()
    assert "<!-- PROGRESS:START -->" in t and "<!-- PROGRESS:END -->" in t and t.index("PROGRESS:START") < t.index("PROGRESS:END")


def test_every_finding_has_a_figure_or_a_table(repo):
    t = (repo / "README.md").read_text(); body = t[t.index("## Findings"):t.index("## Pipeline progress")]
    sections = re.split(r"^### (\d+)\. ", body, flags=re.M)[1:]
    for n, sec in zip(sections[::2], sections[1::2]):
        has_fig = re.search(rf"finding_{n}_\w+\.png", sec); has_table = re.search(r"^\|---", sec, flags=re.M)
        assert has_fig or has_table, f"finding {n} has neither a figure nor a table"

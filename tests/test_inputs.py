"""The per-target inputs that ship with the repository (constructs and exact MSAs) and the script that builds the rest."""
import gzip, subprocess, sys
import pytest

TARGETS = {"588689": 275, "504329": 219, "1053173-743445": 141, "493248-485317": 139, "434954-2097": 420, "540297-493091": 180, "463203-2650": 345, "624273-588549": 580}


@pytest.mark.parametrize("t,length", TARGETS.items())
def test_construct_length_matches_the_readme_table(repo, t, length):
    assert len((repo / "inputs/constructs" / f"{t}_seq.txt").read_text().strip()) == length


@pytest.mark.parametrize("t", TARGETS)
def test_msa_header_and_query_row(repo, t):
    rows = gzip.open(repo / "inputs/msa" / f"{t}_msa.csv.gz", "rt").read().split("\n")
    assert rows[0] == "key,sequence"
    assert rows[1].split(",", 1)[1] == (repo / "inputs/constructs" / f"{t}_seq.txt").read_text().strip()    # row 1 is the query itself
    assert len([r for r in rows[1:] if r]) >= 200


def test_setup_targets_uses_the_shipped_constructs(repo):
    import setup_targets as st
    assert set(st.SEQS) == set(TARGETS)
    for t, (_, seq) in st.SEQS.items():
        assert seq == (repo / "inputs/constructs" / f"{t}_seq.txt").read_text().strip(), t


def test_setup_targets_regenerates_the_inputs_byte_for_byte(repo, tmp_path):
    """Local-only: needs data/mf-pcba_test/*.csv (from the Boltzina release) and the data/<target>_results.csv we used."""
    if not (repo / "data/mf-pcba_test/588689.csv").exists() or not (repo / "data/588689_results.csv").exists():
        pytest.skip("MF-PCBA data not present")
    subprocess.run([sys.executable, str(repo / "pipeline_local/setup_targets.py"), "--out-dir", str(tmp_path)], check=True, capture_output=True)
    for t in TARGETS:
        assert (tmp_path / f"{t}_results.csv").read_bytes() == (repo / f"data/{t}_results.csv").read_bytes(), t
        assert (tmp_path / f"{t}_seq.txt").read_bytes() == (repo / f"data/{t}_seq.txt").read_bytes(), t

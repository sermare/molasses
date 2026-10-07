"""The screening metrics of the authors' evaluation code (BoltzFT/evaluation/metrics.py), which every number in this repo goes through."""
import numpy as np, pytest
pytestmark = pytest.mark.needs_boltzft


@pytest.fixture(scope="module")
def m():
    import metrics   # BoltzFT/evaluation is put on sys.path by tests/conftest.py via the eval script, so add it here too
    return metrics


@pytest.fixture(autouse=True, scope="module")
def _path():
    import os, sys
    p = os.path.join(os.environ["BOLTZFT"], "evaluation")
    if p not in sys.path: sys.path.insert(0, p)


def data(n=10_000, n_act=100, seed=0):
    rng = np.random.default_rng(seed); y = np.zeros(n, int); y[:n_act] = 1
    return y, rng


def test_perfect_ranking(m):
    y, _ = data(); s = y.astype(float) + np.linspace(0, 1e-3, len(y))[::-1] * 0
    r = m.all_metrics(y, s)
    assert r["auprc"] == pytest.approx(1.0) and r["auroc"] == pytest.approx(1.0)
    assert r["bedroc"] == pytest.approx(1.0, abs=1e-6)
    assert r["ef_1pct"] == pytest.approx(1.0 / y.mean())     # all 100 compounds of the top 1% are active: EF = 1 / prevalence


def test_random_ranking_is_near_prevalence(m):
    y, rng = data(n=50_000, n_act=500); aps = [m.all_metrics(y, rng.random(len(y)))["auprc"] for _ in range(5)]
    assert np.mean(aps) == pytest.approx(y.mean(), rel=0.35)
    assert m.all_metrics(y, rng.random(len(y)))["auroc"] == pytest.approx(0.5, abs=0.05)


def test_worst_ranking_has_zero_early_enrichment(m):
    y, _ = data(); s = -y.astype(float)     # actives at the very bottom
    r = m.all_metrics(y, s)
    assert r["ef_1pct"] == 0 and r["hits_at_20"] == 0 and r["hits_at_100"] == 0 and r["bedroc"] < 0.01


def test_enrichment_factor_hand_example(m):
    y = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 1]); s = np.arange(10, 0, -1, dtype=float)   # actives at ranks 1 and 10
    assert m.enrichment_factor(y, s, 0.1) == pytest.approx(1.0 / 0.2)       # top 1 of 10 is active: hit rate 1 against base 0.2
    assert m.enrichment_factor(y, s, 0.5) == pytest.approx((1 / 5) / 0.2)  # top 5 contain one active
    assert m.recall_at_k(y, s, 1) == pytest.approx(0.5)


def test_no_actives_gives_nan_not_a_crash(m):
    y = np.zeros(100, int); r = m.all_metrics(y, np.random.default_rng(0).random(100))
    assert np.isnan(r["auprc"]) and np.isnan(r["bedroc"]) and np.isnan(r["ef_1pct"])

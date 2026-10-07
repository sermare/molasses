"""Rank-correlation helpers used for the uncertainty-vs-similarity analysis (notebook 02, Part 8b) and the continuous-readout analysis."""
import numpy as np, pytest
import bft_uncertainty_sim as us
import bft_continuous as bc


def test_rho_is_rank_based_and_bounded():
    rng = np.random.default_rng(0); a = rng.normal(size=500)
    assert bc.rho(a, a) == pytest.approx(1.0) and bc.rho(a, -a) == pytest.approx(-1.0)
    assert bc.rho(a, np.exp(3 * a)) == pytest.approx(1.0)                 # any monotone transform leaves it at 1
    assert abs(bc.rho(a, rng.normal(size=500))) < 0.15


def test_rho_matches_scipy():
    from scipy.stats import spearmanr
    rng = np.random.default_rng(1); a = rng.normal(size=300); b = a + rng.normal(size=300)
    assert bc.rho(a, b) == pytest.approx(spearmanr(a, b)[0], abs=1e-9)


def test_partial_spearman_removes_a_shared_cause():
    rng = np.random.default_rng(2); s = rng.normal(size=20_000)           # the score drives both uncertainty and similarity
    u = s + 0.3 * rng.normal(size=s.size); tc = s + 0.3 * rng.normal(size=s.size)
    assert bc.rho(u, tc) > 0.8                                           # raw correlation is large ...
    assert abs(us.partial_spearman(u, tc, s)) < 0.05                     # ... and vanishes once the score is removed


def test_partial_spearman_keeps_a_direct_effect():
    rng = np.random.default_rng(3); s = rng.normal(size=20_000); tc = rng.normal(size=s.size)
    u = s - 0.8 * tc + 0.3 * rng.normal(size=s.size)                     # uncertainty falls with similarity at fixed score
    assert us.partial_spearman(u, tc, s) < -0.5


def test_partial_spearman_ignores_nans_and_monotone_transforms():
    rng = np.random.default_rng(4); s = rng.normal(size=5000); tc = rng.normal(size=5000); u = tc + s + rng.normal(size=5000)
    base = us.partial_spearman(u, tc, s)
    u2 = u.copy(); u2[:100] = np.nan
    assert us.partial_spearman(u2, tc, s) == pytest.approx(base, abs=0.05)
    assert us.partial_spearman(np.exp(u / 3), tc, s) == pytest.approx(base, abs=1e-9)


def test_boot_partial_interval_contains_the_estimate_and_excludes_zero_for_a_real_effect():
    rng = np.random.default_rng(5); s = rng.normal(size=4000); tc = rng.normal(size=4000); u = -0.5 * tc + rng.normal(size=4000)
    est, (lo, hi) = us.boot_partial(u, tc, s, B=100)
    assert lo <= est <= hi and hi < 0

"""The cross-target statistics of eval_headft_seeds.py (Table 2 and the seed spread), checked on synthetic per-seed results."""
import numpy as np, pandas as pd, pytest
pytestmark = pytest.mark.needs_boltzft


@pytest.fixture(scope="module")
def ev():
    import eval_headft_seeds as e
    return e


def frame(ratios, seeds=5, sizes=(40, 100, 300), base_ap=0.1, jitter=0.0, seed=0):
    """base AP 0.1 for each target; head-FT AP = ratio x base (same for every size) with optional seed jitter."""
    rng = np.random.default_rng(seed); rows = []
    for i, r in enumerate(ratios):
        t = f"t{i}"; rows.append(dict(target=t, arm="base", n_train=0, seed=-1, ap=base_ap, ef1=10.0, bedroc=0.4))
        for n in sizes:
            for s in range(seeds):
                f = 1 + jitter * rng.standard_normal()
                rows.append(dict(target=t, arm="headft", n_train=n, seed=s, ap=base_ap * r * f, ef1=10.0 * r * f, bedroc=0.4 * r * f))
    return pd.DataFrame(rows)


def test_geometric_mean_and_sign_test_with_six_targets(ev):
    ratios = [2.0, 3.0, 1.5, 1.4, 2.3, 1.6]; t = ev.table2(frame(ratios), (300,)); ap = t[t.metric == "AP"].iloc[0]
    assert ap.geo_mean_ratio == pytest.approx(np.exp(np.mean(np.log(ratios))))
    assert ap.improved == "6/6"
    assert ap.sign_p == pytest.approx(0.5 ** 5)          # two-sided sign test, 6 of 6: the minimum p-value with six targets (0.03125)
    assert ap.ci_low < ap.geo_mean_ratio < ap.ci_high


def test_minimum_sign_p_with_five_targets_is_0_0625(ev):
    t = ev.table2(frame([2, 2, 2, 2, 2.0]), (300,)); assert t[t.metric == "AP"].iloc[0].sign_p == pytest.approx(0.0625)


def test_ties_and_losses_are_not_improvements(ev):
    t = ev.table2(frame([2.0, 1.0, 0.5, 3.0]), (300,)); r = t[t.metric == "AP"].iloc[0]
    assert r.improved == "2/4"                            # a ratio of exactly 1 is a non-improvement
    assert r.geo_mean_ratio == pytest.approx(np.exp(np.mean(np.log([2, 1, .5, 3]))))


def test_single_target_has_no_interval(ev):
    r = ev.table2(frame([2.0]), (300,)); r = r[r.metric == "AP"].iloc[0]
    assert np.isnan(r.ci_low) and np.isnan(r.sign_p) and r.improved == "1/1"


def test_seed_averaging_happens_before_the_ratio(ev):
    df = frame([2.0, 2.0, 2.0], jitter=0.2, seed=3); t = ev.table2(df, (300,)); ap = t[t.metric == "AP"].iloc[0]
    expected = df[(df.arm == "headft") & (df.n_train == 300)].groupby("target").ap.mean().mean() / 0.1
    assert ap.ft_mean / ap.base_mean == pytest.approx(expected)


def test_spread_table_counts_seeds_and_sd(ev):
    df = frame([2.0, 3.0], seeds=5, jitter=0.1, seed=1); sp = ev.spread_table(df)
    h = sp[(sp.target == "t0") & (sp.n_train == 300)].iloc[0]
    vals = df[(df.target == "t0") & (df.n_train == 300)].ap
    assert h.n_seeds == 5 and h.ap_mean == pytest.approx(vals.mean()) and h.ap_sd == pytest.approx(vals.std(ddof=1))
    assert h.ap_min == pytest.approx(vals.min()) and h.ap_max == pytest.approx(vals.max())
    assert sp[(sp.target == "t0") & (sp.n_train == 0)].iloc[0].n_seeds == 1      # the No-FT row

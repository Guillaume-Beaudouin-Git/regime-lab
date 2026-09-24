"""Tests for `regime_lab.extensions.longhist` — synthetic data only."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regime_lab.extensions import longhist as lh
from regime_lab.features import crosssection


def _panel(n: int = 400, k: int = 12, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    common = rng.normal(0, 1, (n, 1))
    values = 0.6 * common + rng.normal(0, 1, (n, k))
    index = pd.bdate_range("2000-01-03", periods=n)
    return pd.DataFrame(values, index=index, columns=[f"i{j}" for j in range(k)])


def test_cross_section_matches_repository_on_complete_panels():
    panel = _panel()
    pd.testing.assert_series_equal(lh.average_correlation(panel),
                                   crosssection.average_correlation(panel), check_names=False)
    pd.testing.assert_series_equal(lh.absorption_ratio(panel, window=126),
                                   crosssection.absorption_ratio(panel, window=126),
                                   check_names=False)


def test_cross_section_survives_a_missing_industry():
    panel = _panel()
    panel.iloc[:200, 0] = np.nan
    assert crosssection.average_correlation(panel).iloc[100:200].isna().all()
    corr = lh.average_correlation(panel)
    assert corr.iloc[100:].notna().all()
    assert lh.absorption_ratio(panel, window=126).iloc[130:].notna().all()


def test_breadth_counts_populated_columns_only():
    panel = pd.DataFrame({"a": [1.0] * 70, "b": [-1.0] * 70, "c": [np.nan] * 70},
                         index=pd.bdate_range("2000-01-03", periods=70))
    out = lh.breadth(panel)
    assert out.iloc[:62].isna().all()
    assert out.iloc[-1] == pytest.approx(0.5)


def test_market_returns_and_short_rate():
    index = pd.bdate_range("2000-01-03", periods=45)
    ff3 = pd.DataFrame({"ff_mkt_rf": 1.0, "ff_rf": 0.01}, index=index)
    r = lh.market_returns(ff3)
    assert r["total"].iloc[0] == pytest.approx(0.0101)
    assert r["excess"].iloc[0] == pytest.approx(0.01)
    rate = lh.short_rate(r["cash"])
    january = index[index.month == 1]
    assert rate.loc[january[0]] == pytest.approx(len(january) * 0.0001 * 12 * 100)


def test_monthly_asof_enters_on_publication():
    frame = pd.DataFrame({
        "series_id": ["fred_aaa", "fred_aaa"],
        "period": pd.to_datetime(["2000-01-01", "2000-02-01"]),
        "available_at": pd.to_datetime(["2000-02-08", "2000-03-08"]),
        "value": [5.0, 6.0],
    })
    index = pd.bdate_range("2000-02-01", "2000-03-31")
    out = lh.monthly_asof(frame, "fred_aaa", index)
    assert out.loc["2000-02-07"] != out.loc["2000-02-07"]
    assert out.loc["2000-02-08"] == 5.0
    assert out.loc["2000-03-07"] == 5.0
    assert out.loc["2000-03-08"] == 6.0


def test_asymmetric_exit_releases_only_after_k_sessions():
    rng = np.random.default_rng(1)
    index = pd.bdate_range("2000-01-03", periods=300)
    vol = np.r_[np.full(150, 0.03), np.full(150, 0.005)]
    returns = pd.Series(rng.normal(0, 1, 300) * vol, index=index)
    state = pd.Series(lh.STRESS, index=index)
    state.iloc[:10] = np.nan
    out = lh.asymmetric_exit(state, returns, k=10)
    assert out.iloc[:10].isna().all()
    rv21 = lh.realised_vol(returns, 21)
    rv63 = lh.realised_vol(returns, 63)
    below = (rv21 < rv63) & rv63.notna()
    released = below.astype(float).rolling(10).min().eq(1.0)
    expected = pd.Series(np.where(released, lh.CALM, lh.STRESS), index=index)
    pd.testing.assert_series_equal(out.iloc[10:], expected.iloc[10:], check_names=False)
    assert (out.iloc[150:] == lh.CALM).any()
    calm = pd.Series(lh.CALM, index=index)
    assert (lh.asymmetric_exit(calm, returns) == lh.CALM).all()


def test_bear_market_uses_calendar_months():
    index = pd.bdate_range("2000-01-03", "2003-12-31")
    price = pd.Series(np.linspace(100, 50, len(index)), index=index)
    bear = lh.bear_market(price)
    assert bear.loc[:"2001-12-31"].isna().all()
    assert (bear.loc["2002-01-07":] == 1.0).all()
    rising = pd.Series(np.linspace(50, 100, len(index)), index=index)
    assert (lh.bear_market(rising).dropna() == 0.0).all()


def test_panic_state_needs_bear_and_high_variance():
    index = pd.bdate_range("2000-01-03", "2004-12-31")
    rng = np.random.default_rng(2)
    vol = np.select([index < pd.Timestamp("2002-01-01"), index < pd.Timestamp("2003-06-01")],
                    [0.02, 0.005], 0.04)
    returns = pd.Series(rng.normal(0, 1, len(index)) * vol, index=index)
    falling = pd.Series(np.linspace(100, 40, len(index)), index=index)
    label = lh.panic_state(falling, returns)
    assert label.loc["2002-06-03":"2003-05-01"].eq(lh.CALM).all()
    assert label.loc["2004-06-01":].eq(lh.STRESS).all()
    rising = pd.Series(np.linspace(40, 100, len(index)), index=index)
    assert lh.panic_state(rising, returns).dropna().eq(lh.CALM).all()


def test_recessions_reference_and_detection():
    months = pd.date_range("2000-01-01", "2002-12-01", freq="MS")
    usrec = pd.Series(0, index=months)
    usrec.loc["2001-03-01":"2001-11-01"] = 1
    episodes = lh.recessions(usrec)
    assert episodes == [(pd.Timestamp("2001-03-01"), pd.Timestamp("2001-11-30"))]
    index = pd.bdate_range("2000-01-03", "2002-12-31")
    ref = lh.daily_reference(usrec, index)
    assert ref.loc["2001-03-01"] == 1.0 and ref.loc["2001-02-28"] == 0.0
    label = pd.Series(lh.CALM, index=index)
    label.loc["2001-02-15":"2001-06-29"] = lh.STRESS
    table = lh.detection(label, episodes)
    assert bool(table.loc[0, "detected"])
    assert table.loc[0, "latency_days"] == -14.0
    assert lh.detection(pd.Series(lh.CALM, index=index), episodes).loc[0, "latency_days"] != \
        lh.detection(pd.Series(lh.CALM, index=index), episodes).loc[0, "latency_days"]


def test_transitions_and_kappas():
    index = pd.bdate_range("2000-01-03", periods=600)
    label = pd.Series(np.r_[np.ones(200), np.zeros(200), np.ones(200)], index=index)
    years = (index[-1] - index[0]).days / 365.25
    assert lh.transitions_per_year(label) == pytest.approx(2 / years)
    ref = 1.0 - label
    assert lh.nber_kappa(label, ref) == pytest.approx(1.0)
    assert lh.kappa(label, label) == pytest.approx(1.0)
    same = lh.kappa_difference_ci(label, label, ref, block=50, draws=50)
    assert same["delta"] == 0.0 and same["low"] == 0.0 and same["high"] == 0.0


def test_sharpe_refuses_non_finite_and_folds_split_evenly():
    x = pd.Series([0.01, -0.005, 0.002, np.nan])
    assert np.isnan(lh.sharpe(x))
    index = pd.bdate_range("2000-01-03", periods=500)
    rng = np.random.default_rng(3)
    a = pd.Series(rng.normal(0.001, 0.01, 500), index=index)
    deltas = lh.fold_deltas(a, a * 0.5)
    assert len(deltas) == 5
    assert np.allclose(deltas, 0.0)


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ((0.30, 0.20, 3.0, 2.39, 0.99, 5, {"m": 0.1}), "USEFUL"),
        ((0.30, 0.20, 3.0, 2.39, 0.99, 5, {"m": 0.35}), "NOT SHOWN"),
        ((0.30, 0.20, 2.0, 2.39, 0.99, 5, {"m": 0.1}), "NOT SHOWN"),
        ((0.30, 0.20, 3.0, 2.39, 0.99, 2, {"m": 0.1}), "NOT SHOWN"),
        ((0.30, 0.20, 3.0, 2.39, 0.90, 5, {"m": 0.1}), "NOT SHOWN"),
        ((0.10, 0.20, 3.0, 2.39, 0.99, 5, {"m": 0.0}), "UNDERPOWERED"),
        ((-0.10, 0.20, -1.0, 2.39, 0.20, 1, {"m": 0.0}), "NOT USEFUL"),
        ((-0.30, 0.20, -3.0, 2.39, 0.01, 0, {"m": 0.0}), "HARMFUL"),
        ((-0.30, 0.20, -1.0, 2.39, 0.01, 0, {"m": 0.0}), "NOT USEFUL"),
        ((0.30, 0.20, np.nan, 2.39, 0.99, 5, {"m": 0.1}), "NOT FINITE"),
        ((0.30, 0.20, 3.0, 2.39, 0.99, 5, {"m": np.nan}), "NOT FINITE"),
    ],
)
def test_verdict(args, expected):
    assert lh.verdict(*args).startswith(expected)


def test_fast_dp_is_bit_identical_to_the_reference():
    from jumpmodels import jump as reference

    ref = getattr(reference, "_longhist_reference_dp", reference.dp)
    rng = np.random.default_rng(4)
    for penalty in (0.0, 1.0, 30.0, 300.0):
        pen = reference.jump_penalty_to_mx(penalty, 2)
        loss = rng.gamma(2.0, 1.0, (3000, 2))
        loss[100:110] = 1.0
        loss[500, 1] = np.nan
        loss[700:720, 0] = np.inf
        values = lh.fast_dp(loss, pen, return_value_mx=True)
        assert np.array_equal(values, ref(loss, pen, return_value_mx=True))
        path, best = lh.fast_dp(loss, pen)
        ref_path, ref_best = ref(loss, pen)
        assert np.array_equal(path, ref_path)
        assert best == ref_best


def test_fast_dp_leaves_a_sparse_jump_fit_unchanged():
    from jumpmodels import jump as reference
    from jumpmodels.sparse_jump import SparseJumpModel

    rng = np.random.default_rng(5)
    regime = np.repeat([0, 1, 0, 1], 150)
    x = rng.normal(0, 1, (600, 6)) + regime[:, None] * np.array([2.0, 1.0, 0, 0, 0, 0])
    frame = pd.DataFrame(x)
    original = reference.dp
    try:
        reference.dp = getattr(reference, "_longhist_reference_dp", original)
        slow = SparseJumpModel(n_components=2, max_feats=3.0, jump_penalty=10.0).fit(frame)
        slow_labels, slow_online = slow.labels_.to_numpy(), slow.predict_online(frame).to_numpy()
        lh.use_fast_dp()
        fast = SparseJumpModel(n_components=2, max_feats=3.0, jump_penalty=10.0).fit(frame)
        assert np.array_equal(slow_labels, fast.labels_.to_numpy())
        assert np.array_equal(slow_online, fast.predict_online(frame).to_numpy())
        assert np.array_equal(slow.feat_weights, fast.feat_weights)
    finally:
        reference.dp = original

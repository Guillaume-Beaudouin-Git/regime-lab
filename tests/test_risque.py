"""Construction tests for the risk-forecast study (`regime_lab/extensions/risque.py`).

Synthetic data only: nothing here reads a state, a price file or a result.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regime_lab.extensions import risque


def _returns(n: int = 900, seed: int = 0) -> pd.Series:
    """A return series with slow volatility clustering, so a HAR has something to find."""
    rng = np.random.default_rng(seed)
    vol = 0.01 * np.exp(0.5 * np.sin(np.arange(n) / 60.0) + 0.1 * rng.normal(size=n))
    index = pd.bdate_range("2005-01-03", periods=n)
    return pd.Series(vol * rng.normal(size=n), index=index)


def _design(r: pd.Series, horizon: int = 5):
    y = np.log(risque.forward_variance(r, horizon)).to_numpy()
    x = np.column_stack([np.ones(len(r)), risque.har_components(r, (5, 22)).to_numpy()])
    rows = np.isfinite(y) & np.isfinite(x).all(axis=1)
    return y, x, rows


def test_forward_variance_sums_the_next_sessions_only():
    r = pd.Series(np.arange(1.0, 11.0), index=pd.bdate_range("2020-01-01", periods=10))
    fwd = risque.forward_variance(r, horizon=3)
    assert fwd.iloc[0] == pytest.approx(2.0**2 + 3.0**2 + 4.0**2)
    assert fwd.iloc[6] == pytest.approx(8.0**2 + 9.0**2 + 10.0**2)
    assert fwd.iloc[7:].isna().all()


def test_forward_variance_is_missing_when_a_return_is():
    r = pd.Series([0.01, 0.02, np.nan, 0.01, 0.03, 0.02, 0.01])
    fwd = risque.forward_variance(r, horizon=2)
    assert np.isnan(fwd.iloc[0]) and np.isnan(fwd.iloc[1])
    assert np.isfinite(fwd.iloc[2])


def test_log_returns_refuse_a_non_positive_price():
    p = pd.Series([10.0, 11.0, -5.0, 12.0, 13.0])
    r = risque.log_returns(p)
    assert np.isnan(r.iloc[2]) and np.isnan(r.iloc[3])
    assert r.iloc[4] == pytest.approx(np.log(13.0 / 12.0))


def test_har_components_are_strict_and_refuse_a_zero_window():
    r = pd.Series([0.0] * 6 + [0.01, -0.02, 0.01, np.nan, 0.02, 0.01, 0.0])
    comps = risque.har_components(r, windows=(3,))
    assert np.isnan(comps["har_3"].iloc[4])  # three stale sessions: no log
    assert np.isfinite(comps["har_3"].iloc[8])
    assert comps["har_3"].iloc[9:12].isna().all()  # a missing return poisons its windows
    assert comps["har_3"].iloc[8] == pytest.approx(np.log((0.01**2 + 0.02**2 + 0.01**2) / 3))


def test_expanding_forecast_matches_a_slow_refit():
    r = _returns()
    h = 5
    y, x, rows = _design(r, h)
    m = risque.moments(y, x, rows, horizon=h)
    fast = risque.forecast(m, min_rows=100)
    for t in (150, 400, 700, 880):
        train = np.flatnonzero(rows[: t - h + 1])
        if len(train) < 100:
            assert np.isnan(fast[t])
            continue
        beta, *_ = np.linalg.lstsq(x[train], y[train], rcond=None)
        resid = y[train] - x[train] @ beta
        s2 = resid @ resid / (len(train) - x.shape[1])
        assert fast[t] == pytest.approx(np.exp(x[t] @ beta + 0.5 * s2), rel=1e-8)


def test_forecast_never_uses_a_return_after_the_forecast_date():
    r = _returns()
    h = 5
    cut = 600
    y, x, rows = _design(r, h)
    before = risque.forecast(risque.moments(y, x, rows, horizon=h), min_rows=100)
    shocked = r.copy()
    shocked.iloc[cut + 1:] *= 7.0
    y2, x2, rows2 = _design(shocked, h)
    after = risque.forecast(risque.moments(y2, x2, rows2, horizon=h), min_rows=100)
    np.testing.assert_allclose(before[: cut + 1], after[: cut + 1], rtol=1e-12, equal_nan=True)
    assert not np.allclose(before[cut + 30:], after[cut + 30:], equal_nan=True)


def test_dummy_forecast_equals_the_full_design_and_abstains_on_a_thin_bucket():
    r = _returns()
    h = 5
    y, x, rows = _design(r, h)
    n = len(y)
    dummy = ((np.arange(n) // 90) % 3 == 0).astype(float)
    m = risque.moments(y, x, rows, horizon=h)
    fast = risque.forecast_with_dummy(m, dummy, min_rows=100, min_bucket=20)
    full = risque.forecast(risque.moments(y, np.column_stack([x, dummy]), rows, horizon=h),
                           min_rows=100)
    ok = np.isfinite(full)
    np.testing.assert_allclose(fast[ok], full[ok], rtol=1e-9)

    late = np.zeros(n)
    late[700:] = 1.0  # the stress bucket is empty until session 700
    base = risque.forecast(m, min_rows=100)
    thin = risque.forecast_with_dummy(m, late, min_rows=100, min_bucket=20)
    np.testing.assert_allclose(thin[:700], base[:700], equal_nan=True)


def test_dummy_must_be_binary_and_present_on_training_rows():
    r = _returns(300)
    y, x, rows = _design(r)
    m = risque.moments(y, x, rows, horizon=5)
    with pytest.raises(ValueError):
        risque.forecast_with_dummy(m, np.full(len(y), 0.5))
    with pytest.raises(ValueError):
        risque.forecast_with_dummy(m, np.full(len(y), np.nan))


def test_moments_refuse_a_non_finite_training_row():
    y = np.array([1.0, np.nan, 2.0])
    x = np.ones((3, 1))
    with pytest.raises(ValueError):
        risque.moments(y, x, np.array([True, True, True]), horizon=1)


def test_qlike_is_zero_at_the_truth_and_positive_elsewhere():
    s = pd.Series([1.0, 2.0, 3.0, 0.0])
    h = pd.Series([1.0, 1.0, 6.0, 1.0])
    loss = risque.qlike(s, h)
    assert loss.iloc[0] == pytest.approx(0.0)
    assert (loss.iloc[1:3] > 0).all()
    assert np.isnan(loss.iloc[3])
    arr = risque.qlike(np.array([2.0]), np.array([2.0]))
    assert arr[0] == pytest.approx(0.0)


def test_qlike_punishes_under_prediction_more_than_over_prediction():
    under = risque.qlike(np.array([2.0]), np.array([1.0]))[0]
    over = risque.qlike(np.array([1.0]), np.array([2.0]))[0]
    assert under > over


def test_blinded_se_ignores_the_mean_and_refuses_a_nan():
    rng = np.random.default_rng(1)
    a = rng.normal(size=(400, 2))
    se = risque.blinded_mean_se(a, mean_block=10, draws=200, seed=3)
    shifted = risque.blinded_mean_se(a + np.array([5.0, -2.0]), mean_block=10, draws=200, seed=3)
    np.testing.assert_allclose(se, shifted, rtol=1e-9)
    assert (se > 0).all()
    bad = a.copy()
    bad[3, 0] = np.nan
    with pytest.raises(ValueError):
        risque.blinded_mean_se(bad, mean_block=10, draws=10)


def test_hac_mean_t_sign_and_short_sample():
    x = np.r_[np.full(50, 1.0), np.full(50, 1.2)]
    assert risque.hac_mean_t(x, lags=5) > 0
    assert np.isnan(risque.hac_mean_t(np.ones(10), lags=5))


def test_holm_rejects_in_order_and_stops():
    reject = risque.holm_reject([0.001, 0.04, 0.012, np.nan], alpha=0.05)
    # m = 3: 0.001 <= 0.05/3, 0.012 <= 0.05/2, 0.04 <= 0.05/1
    assert reject.tolist() == [True, True, True, False]
    assert risque.holm_reject([0.03, 0.04], alpha=0.05).tolist() == [False, False]


def test_participation_ratio_bounds():
    assert risque.participation_ratio(np.eye(5)) == pytest.approx(5.0)
    assert risque.participation_ratio(np.ones((5, 5))) == pytest.approx(1.0)


def test_fold_bounds_are_contiguous_and_cover():
    bounds = risque.fold_bounds(103, 5)
    assert bounds[0][0] == 0 and bounds[-1][1] == 103
    assert all(a[1] == b[0] for a, b in zip(bounds[:-1], bounds[1:], strict=True))
    with pytest.raises(ValueError):
        risque.fold_bounds(3, 5)


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ((0.5, 0.3, 2.0, 0.99, [0.1, 0.2], 4), "USEFUL"),
        ((0.5, 0.3, 2.0, 0.99, [0.6, 0.2], 4), "NOT SHOWN"),
        ((0.5, 0.3, 2.0, 0.99, [0.1, 0.2], 2), "NOT SHOWN"),
        ((0.5, 0.3, 2.0, 0.90, [0.1, 0.2], 5), "NOT SHOWN"),
        ((0.1, 0.3, 2.0, 0.99, [0.0], 5), "UNDERPOWERED"),
        ((-0.1, 0.3, -1.0, 0.2, [0.0], 1), "NOT USEFUL"),
        ((-0.5, 0.3, -3.0, 0.01, [0.0], 0), "HARMFUL"),
        ((-0.5, 0.3, -3.0, 0.40, [0.0], 0), "NOT USEFUL"),
        ((0.5, 0.3, np.nan, 0.99, [0.1], 5), "NOT FINITE"),
    ],
)
def test_verdict(args, expected):
    delta, mde, t, pct, controls, folds = args
    assert risque.verdict(delta, mde, t, pct, controls, folds).startswith(expected)


def test_trend_weights_use_yesterdays_forecast():
    index = pd.bdate_range("2010-01-01", periods=300)
    prices = pd.DataFrame({"a": np.linspace(100, 200, 300), "b": np.linspace(200, 100, 300)},
                          index=index)
    sigma = pd.DataFrame({"a": np.linspace(0.1, 0.3, 300), "b": 0.2}, index=index)
    w = risque.trend_raw_weights(prices, sigma)
    t = 280
    assert w["a"].iloc[t] == pytest.approx(0.10 / sigma["a"].iloc[t - 1])
    assert w["b"].iloc[t] == pytest.approx(-0.10 / 0.2)
    assert (w.iloc[:253] == 0).all().all()


def test_risk_parity_weights_sum_to_one_and_are_lagged():
    index = pd.bdate_range("2010-01-01", periods=5)
    sigma = pd.DataFrame({"a": [0.1, 0.1, 0.2, 0.1, 0.1], "b": 0.2, "c": np.nan}, index=index)
    w = risque.risk_parity_raw_weights(sigma)
    assert w.iloc[0].sum() == 0.0
    np.testing.assert_allclose(w.iloc[1:].sum(axis=1), 1.0)
    assert w["a"].iloc[3] == pytest.approx(0.5)  # sigma equal on the day before
    assert w["a"].iloc[1] == pytest.approx(2.0 / 3.0)


def test_targeted_book_applies_the_lagged_multiplier():
    rng = np.random.default_rng(4)
    index = pd.bdate_range("2010-01-01", periods=200)
    returns = pd.DataFrame(rng.normal(0, 0.01, (200, 2)), index=index, columns=["a", "b"])
    raw = pd.DataFrame(0.5, index=index, columns=["a", "b"])
    book, weights = risque.targeted_book(raw, returns)
    unscaled = (raw * returns).sum(axis=1)
    sigma = unscaled.rolling(63).std() * np.sqrt(252)
    t = 150
    assert weights["a"].iloc[t] == pytest.approx(0.5 * min(0.10 / sigma.iloc[t - 1], 3.0))
    assert book.iloc[t] == pytest.approx((weights.iloc[t] * returns.iloc[t]).sum())

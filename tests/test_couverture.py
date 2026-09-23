"""Construction tests for the hedge-choice study (`regime_lab/extensions/couverture.py`).

Synthetic data only: nothing here reads a state, a price file or a result.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from regime_lab.extensions import couverture as cv


def _index(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2000-01-03", periods=n)


def test_rolling_correlation_drops_a_holiday_instead_of_blanking_a_window():
    rng = np.random.default_rng(0)
    idx = _index(300)
    a = pd.Series(rng.normal(size=300), index=idx)
    b = pd.Series(rng.normal(size=300), index=idx)
    b.iloc[150] = np.nan
    corr = cv.rolling_correlation(a, b, window=20)
    assert len(corr) == 299 - 19
    assert corr.index[0] == idx[19]
    kept = pd.concat([a, b], axis=1).dropna()
    expected = np.corrcoef(kept.iloc[140:160, 0], kept.iloc[140:160, 1])[0, 1]
    assert corr.loc[kept.index[159]] == pytest.approx(expected, rel=1e-10)


def test_hysteresis_flips_only_after_k_sessions():
    flag = pd.Series([1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1], index=_index(11)).astype(bool)
    out = cv.hysteresis(flag, 3)
    assert out.tolist() == [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1]


def test_hysteresis_is_causal():
    rng = np.random.default_rng(1)
    flag = pd.Series(rng.random(400) > 0.5, index=_index(400))
    base = cv.hysteresis(flag, 5)
    shocked = flag.copy()
    shocked.iloc[250:] = ~shocked.iloc[250:]
    assert cv.hysteresis(shocked, 5).iloc[:250].equals(base.iloc[:250])


def test_hysteresis_rejects_k_below_one():
    with pytest.raises(ValueError):
        cv.hysteresis(pd.Series([True, False]), 0)


def test_correlation_regime_reads_the_sign():
    rng = np.random.default_rng(2)
    n = 400
    idx = _index(n)
    common = rng.normal(size=n)
    eq = pd.Series(common + 0.3 * rng.normal(size=n), index=idx)
    sign = np.where(np.arange(n) < 200, 1.0, -1.0)
    bond = pd.Series(sign * common + 0.3 * rng.normal(size=n), index=idx)
    label = cv.correlation_regime(eq, bond, window=20, k=5)
    assert label.loc[idx[150]] == cv.POSITIVE
    assert label.loc[idx[350]] == cv.NEGATIVE


def test_asof_carries_the_last_label_forward():
    labels = pd.Series([1.0, 0.0], index=pd.DatetimeIndex(["2020-01-02", "2020-01-06"]))
    idx = pd.bdate_range("2020-01-01", "2020-01-07")
    out = cv.asof(labels, idx)
    assert np.isnan(out.iloc[0])
    assert out.loc["2020-01-03"] == 1.0
    assert out.loc["2020-01-06"] == 0.0


def test_vix_rule_is_causal_and_calm_below_the_median():
    rng = np.random.default_rng(3)
    vix = pd.Series(15 + rng.normal(0, 3, 600), index=_index(600))
    rule = cv.vix_rule(vix, min_periods=100)
    assert rule.iloc[:99].isna().all()
    t = 300
    median = vix.iloc[: t + 1].median()
    assert rule.iloc[t] == float(vix.iloc[t] < median)
    shocked = vix.copy()
    shocked.iloc[400:] += 50
    assert cv.vix_rule(shocked, min_periods=100).iloc[:400].equals(rule.iloc[:400])


def test_transitions_per_year_counts_changes():
    idx = pd.bdate_range("2000-01-03", periods=11)
    label = pd.Series([0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1], index=idx, dtype=float)
    years = (idx[-1] - idx[0]).days / 365.25
    assert cv.transitions_per_year(label) == pytest.approx(3 / years)


def test_held_weights_permanent_pocket_and_switch():
    idx = _index(4)
    w = {k: pd.Series([1.0, 2.0, 3.0, 4.0], index=idx) for k in ("eq", "bd", "gd")}
    label = pd.Series([0.0, 1.0, 0.0, 1.0], index=idx)
    perm = cv.held_weights(w["eq"], w["bd"], w["gd"], 1.0 - label, equity_on=0.5, pocket_on=0.5)
    assert perm["EQ"].tolist() == [0.5, 1.0, 1.5, 2.0]
    assert perm["BOND"].tolist() == [0.5, 0.0, 1.5, 0.0]
    assert perm["GOLD"].tolist() == [0.0, 1.0, 0.0, 2.0]
    stress = pd.Series([0.0, 0.0, 1.0, 1.0], index=idx)
    sw = cv.held_weights(w["eq"], w["bd"], w["gd"], 1.0 - label, equity_on=1.0 - stress,
                         pocket_on=stress)
    assert sw["EQ"].tolist() == [1.0, 2.0, 0.0, 0.0]
    assert sw["BOND"].tolist() == [0.0, 0.0, 3.0, 0.0]
    assert sw["GOLD"].tolist() == [0.0, 0.0, 0.0, 4.0]
    mix = cv.held_weights(w["eq"], w["bd"], w["gd"], 0.5, equity_on=0.5, pocket_on=0.5)
    assert mix["BOND"].tolist() == mix["GOLD"].tolist() == [0.25, 0.5, 0.75, 1.0]


def test_book_returns_turnover_and_refuses_to_read_nan_as_zero():
    idx = _index(3)
    returns = pd.DataFrame({"EQ": [0.01, 0.02, -0.01], "BOND": [0.0, 0.01, 0.02],
                            "GOLD": [0.0, 0.0, 0.0]}, index=idx)
    held = pd.DataFrame({"EQ": [1.0, 1.0, 0.0], "BOND": [0.0, 1.0, 1.0],
                         "GOLD": [0.0, 0.0, 0.0]}, index=idx)
    r, turnover = cv.book(returns, held)
    assert r.tolist() == pytest.approx([0.01, 0.03, 0.02])
    assert turnover == pytest.approx(np.mean([1.0, 1.0]) * 252)
    held.iloc[1, 2] = np.nan
    r2, _ = cv.book(returns, held)
    assert np.isnan(r2.iloc[1])


def test_forward_and_trailing_variance_windows():
    idx = _index(50)
    r = pd.Series(np.arange(1, 51, dtype=float) / 100, index=idx)
    fwd = cv.forward_log_variance(r, horizon=3)
    assert fwd.iloc[10] == pytest.approx(np.log((r.iloc[11:14] ** 2).sum()))
    assert fwd.iloc[-3:].isna().all()
    trail = cv.trailing_log_variance(r, window=5, horizon=3)
    assert trail.iloc[10] == pytest.approx(np.log((r.iloc[6:11] ** 2).mean() * 3))


def test_forward_variance_does_not_see_the_present():
    rng = np.random.default_rng(4)
    r = pd.Series(rng.normal(0, 0.01, 100), index=_index(100))
    shocked = r.copy()
    shocked.iloc[40] = 1.0
    a, b = cv.forward_log_variance(r), cv.forward_log_variance(shocked)
    assert a.iloc[40] == pytest.approx(b.iloc[40], rel=1e-9)
    assert b.iloc[39] > a.iloc[39] + 1.0


def test_expanding_rank_is_causal():
    x = pd.Series([3.0, 1.0, 2.0, 5.0, 0.0], index=_index(5))
    out = cv.expanding_rank(x, min_periods=2)
    assert np.isnan(out.iloc[0])
    assert out.iloc[2] == pytest.approx(2 / 3)
    assert out.iloc[4] == pytest.approx(1 / 5)


def _regression_data(n: int, beta: float, seed: int) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    controls = rng.normal(size=(n, 2))
    regressor = (rng.random(n) > 0.6).astype(float)
    y = 0.5 + controls @ np.array([0.3, -0.2]) + beta * regressor + rng.normal(size=n)
    return y, controls, regressor


def test_incremental_r2_matches_the_fit():
    y, controls, regressor = _regression_data(2000, 0.3, 5)
    fit = cv.incremental_fit(y, controls, regressor, lags=5)
    assert cv.incremental_r2(y, controls, regressor) == pytest.approx(fit["incremental"])
    assert fit["coef"] == pytest.approx(0.3, abs=0.12)
    assert fit["t"] > 3


def test_regression_mde_matches_the_hac_standard_error_under_the_null():
    y, controls, regressor = _regression_data(5000, 0.0, 6)
    mde = cv.regression_mde(y, controls, regressor, lags=10, alpha=0.05)
    x = np.column_stack([np.ones(len(y)), controls, regressor])
    full = sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": 10})
    assert mde["se"] == pytest.approx(float(full.bse[-1]), rel=0.02)
    assert mde["beta"] == pytest.approx(2.8016 * mde["se"], rel=1e-3)


def test_regression_mde_is_blind_to_the_effect():
    y0, controls, regressor = _regression_data(3000, 0.0, 7)
    shifted = y0 + 0.5 * regressor
    a = cv.regression_mde(y0, controls, regressor, lags=5, alpha=0.01)
    b = cv.regression_mde(shifted, controls, regressor, lags=5, alpha=0.01)
    for key in ("se", "beta", "incremental_r2"):
        assert a[key] == pytest.approx(b[key], rel=1e-9)


def test_out_of_sample_gain_is_causal_and_detects_a_real_regressor():
    y, controls, regressor = _regression_data(3000, 0.6, 8)
    out = cv.out_of_sample_gain(y, controls, regressor, min_train=1000, refit=250,
                                horizon=21, lags=5)
    assert out["n"] == 2000
    assert out["gain"] > 0.02
    assert out["cw_t"] > 3
    noise = np.random.default_rng(9).random(3000) > 0.5
    null = cv.out_of_sample_gain(y, controls, noise.astype(float), min_train=1000, refit=250)
    assert null["gain"] < 0.01

def test_out_of_sample_forecasts_never_use_an_unrealised_target():
    y, controls, regressor = _regression_data(3000, 0.6, 10)
    kw = {"min_train": 1000, "refit": 250, "horizon": 21}
    pb, pf = cv.out_of_sample_forecasts(y, controls, regressor, **kw)
    assert np.isnan(pb[:1000]).all() and np.isfinite(pb[1000:]).all()
    # the first block is forecast at row 1000: targets of rows 980 on are not yet known
    shocked = y.copy()
    shocked[1000 - 21 + 1:] += 100.0
    sb, sf = cv.out_of_sample_forecasts(shocked, controls, regressor, **kw)
    assert np.allclose(pb[1000:1250], sb[1000:1250])
    assert np.allclose(pf[1000:1250], sf[1000:1250])
    assert not np.allclose(pb[1250:1500], sb[1250:1500])
    # the last known target does move them
    moved = y.copy()
    moved[1000 - 21] += 100.0
    mb, _ = cv.out_of_sample_forecasts(moved, controls, regressor, **kw)
    assert not np.allclose(pb[1000:1250], mb[1000:1250])


def test_participation_ratio_bounds():
    assert cv.participation_ratio(np.eye(3)) == pytest.approx(3.0)
    assert cv.participation_ratio(np.ones((3, 3))) == pytest.approx(1.0)

"""Construction tests for the per-factor jump models (`regime_lab/extensions/factorsjm.py`).

Synthetic data only: nothing here reads a factor, a state or a result.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regime_lab.extensions import factorsjm as fj


def _returns(n: int = 600, seed: int = 0, scale: float = 0.01) -> pd.Series:
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2000-01-03", periods=n)
    return pd.Series(rng.normal(0.0, scale, n), index=index)


# --------------------------------------------------------------------------- features
def test_rsi_is_bounded_and_saturates():
    r = _returns()
    out = fj.rsi(r, 21).dropna()
    assert out.between(0, 100).all()
    up = pd.Series(0.01, index=r.index)
    assert fj.rsi(up, 8).dropna().eq(100.0).all()


def test_stochastic_k_is_one_hundred_at_a_new_high():
    r = pd.Series(0.001, index=_returns().index)
    out = fj.stochastic_k(r, 21).dropna()
    assert np.allclose(out, 100.0)
    assert fj.stochastic_k(_returns(), 21).dropna().between(0, 100).all()


def test_macd_is_positive_in_an_uptrend():
    r = pd.Series(0.002, index=_returns().index)
    assert (fj.macd(r, 8, 21).dropna() > 0).all()


def test_ewm_beta_recovers_a_constant_beta():
    m = _returns(seed=1)
    a = 0.4 * m + _returns(seed=2, scale=1e-5)
    beta = fj.ewm_beta(a, m).dropna()
    assert beta.iloc[100:].sub(0.4).abs().max() < 0.01


def test_active_features_columns_and_causality():
    a, m = _returns(seed=3), _returns(seed=4)
    with_beta = fj.active_features(a, m)
    without = fj.active_features(a)
    assert with_beta.shape[1] == 13 and without.shape[1] == 12
    assert "beta_mkt" in with_beta and "beta_mkt" not in without
    shocked = a.copy()
    shocked.iloc[400:] *= 10
    again = fj.active_features(shocked, m)
    pd.testing.assert_frame_equal(with_beta.iloc[:400], again.iloc[:400])


def test_market_features_use_yields_known_at_the_session():
    m = _returns(seed=5)
    days = pd.date_range(m.index.min() - pd.Timedelta(days=10), m.index.max(), freq="D")
    y1 = pd.Series(np.linspace(1, 2, len(days)), index=days)
    y10 = y1 + 1.0
    out = fj.market_features(m, y1, y10)
    assert list(out.columns) == ["mkt_ret_ewm_21", "mkt_logvol_diff_ewm_21",
                                 "y1_diff_ewm_21", "slope_diff_ewm_21"]
    assert out["slope_diff_ewm_21"].dropna().abs().max() < 1e-12
    late = y1.copy()
    late.loc[m.index[300]:] += 5
    again = fj.market_features(m, late, late + 1.0)
    pd.testing.assert_frame_equal(out.iloc[:300], again.iloc[:300])


# --------------------------------------------------------------------------- windows
def test_training_window_floor_and_cap():
    index = pd.bdate_range("1960-01-01", "1990-01-01")
    assert fj.training_window(index, pd.Timestamp("1967-01-02")) is None
    win = fj.training_window(index, pd.Timestamp("1985-01-02"))
    assert win.max() < pd.Timestamp("1985-01-02")
    assert win.min() >= pd.Timestamp("1973-01-02")
    first = fj.training_window(index, pd.Timestamp("1968-01-03"))
    assert first.min() == index.min()


def test_refit_schedule_lands_on_sessions():
    index = pd.bdate_range("2000-01-03", "2003-12-31")
    refits = fj.refit_schedule(index, pd.Timestamp("2000-01-01"))
    assert len(refits) == 8
    assert refits.isin(index).all()
    assert (refits.month.isin([1, 7])).all()


# --------------------------------------------------------------------------- means, exposures
def test_state_means_fall_back_to_the_overall_mean():
    r = pd.Series([0.01, 0.03, -0.02, 0.00])
    both = fj.state_means(r, pd.Series([1, 1, 0, 0]))
    assert both[fj.BULL] == pytest.approx(0.02 * 252)
    assert both[fj.BEAR] == pytest.approx(-0.01 * 252)
    one = fj.state_means(r, pd.Series([1, 1, 1, 1]))
    assert one[fj.BEAR] == pytest.approx(r.mean() * 252)


def test_exposures():
    s = pd.Series([1.0, 0.0, np.nan])
    assert fj.onoff_exposure(s).tolist()[:2] == [1.0, 0.0]
    assert np.isnan(fj.onoff_exposure(s).iloc[2])
    mu = pd.Series([0.10, 0.025, -0.20, 0.0])
    assert fj.paper_exposure(mu).tolist() == [1.0, 0.5, -1.0, 0.0]


def test_long_short_returns_use_the_state_two_sessions_back():
    index = pd.bdate_range("2001-01-01", periods=6)
    path = pd.DataFrame({"state": [1, 1, 0, 0, 1, 1.0], "mu_bull_active": 0.10,
                         "mu_bear_active": -0.025}, index=index)
    active = pd.Series(0.01, index=index)
    out = fj.long_short_returns(path, active)
    assert out.iloc[:2].isna().all()
    assert out.iloc[2:].tolist() == pytest.approx([0.01, 0.01, -0.005, -0.005])


def test_trailing_return_rule():
    r = pd.Series([0.01] * 5 + [-0.03] * 5, index=pd.bdate_range("2001-01-01", periods=10))
    out = fj.trailing_return_rule(r, window=3)
    assert out.iloc[:2].isna().all()
    assert out.iloc[2:5].eq(1.0).all() and out.iloc[6:].eq(0.0).all()


# --------------------------------------------------------------------------- model
def _two_regimes(years: int = 11, seed: int = 7) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    rng = np.random.default_rng(seed)
    n = 252 * years
    index = pd.bdate_range("1990-01-01", periods=n)
    truth = np.zeros(n, dtype=int)
    block, state = 0, 1
    while block < n:
        length = int(rng.integers(150, 400))
        truth[block:block + length] = state
        block, state = block + length, 1 - state
    mean = np.where(truth == 1, 0.002, -0.002)
    active = pd.Series(mean + rng.normal(0, 0.003, n), index=index)
    feats = pd.DataFrame({
        "f1": np.where(truth == 1, 1.0, -1.0) + rng.normal(0, 0.5, n),
        "f2": np.where(truth == 1, -0.5, 0.5) + rng.normal(0, 0.5, n),
        "noise": rng.normal(0, 1, n),
    }, index=index)
    return feats, active, pd.Series(truth, index=index)


def test_walk_forward_is_out_of_sample_and_finds_the_regimes():
    feats, active, truth = _two_regimes()
    refits = fj.refit_schedule(feats.index, pd.Timestamp("1998-01-01"), months=12)
    path = fj.walk_forward(feats, active, refits, jump_penalty=20.0, max_feats=3.0)
    assert path.index.min() >= refits[0]
    assert set(path["state"].unique()) <= {0.0, 1.0}
    hit = (path["state"] == truth.reindex(path.index)).mean()
    assert hit > 0.9
    assert (path["mu_bull_active"] > path["mu_bear_active"]).all()


def test_walk_forward_does_not_see_the_future():
    feats, active, _ = _two_regimes()
    refits = fj.refit_schedule(feats.index, pd.Timestamp("1998-01-01"), months=12)
    full = fj.walk_forward(feats, active, refits, jump_penalty=20.0, max_feats=3.0)
    cut = refits[2]
    shocked_f = feats.copy()
    shocked_f.loc[cut:] = -shocked_f.loc[cut:] * 3
    shocked_a = active.copy()
    shocked_a.loc[cut:] = -shocked_a.loc[cut:]
    again = fj.walk_forward(shocked_f, shocked_a, refits, jump_penalty=20.0, max_feats=3.0)
    before = full.index < cut
    pd.testing.assert_frame_equal(full.loc[before], again.loc[again.index < cut])


def test_select_penalty_follows_the_validation_sharpe():
    index = pd.bdate_range("1990-01-01", "2001-12-31")
    rng = np.random.default_rng(3)
    active = pd.Series(rng.normal(0, 0.01, len(index)), index=index)
    good = pd.DataFrame({"state": 1.0, "refit": index[0], "mu_bull_active": 0.10,
                         "mu_bear_active": -0.10}, index=index)
    good["state"] = np.where(active.shift(-2).fillna(0) > 0, 1.0, 0.0)  # perfect foresight
    bad = good.copy()
    bad["state"] = 1.0 - good["state"]
    tune = [pd.Timestamp("1997-01-02"), pd.Timestamp("1999-01-04")]
    stitched, scores = fj.select_penalty({1.0: bad, 2.0: good}, active, tune)
    assert scores["chosen"].tolist() == [2.0, 2.0]
    assert stitched.index.min() == pd.Timestamp("1997-01-02")
    assert (stitched["penalty"] == 2.0).all()
    late = good.loc["1995-01-02":]
    _, scores2 = fj.select_penalty({1.0: late, 2.0: bad}, active, tune)
    assert scores2["sharpe_1"].isna().all()
    assert scores2["chosen"].tolist() == [2.0, 2.0]


def test_partition_means_carry_training_means():
    index = pd.bdate_range("1980-01-01", "1995-12-31")
    labels = pd.Series((np.arange(len(index)) // 50) % 2, index=index, dtype=float)
    returns = pd.Series(np.where(labels == 1, 0.001, -0.001), index=index)
    refits = fj.refit_schedule(index, pd.Timestamp("1989-01-01"), months=6)
    out = fj.partition_means(labels, returns, refits, index)
    assert np.allclose(out["mu_bull_target"], 0.252)
    assert np.allclose(out["mu_bear_target"], -0.252)
    pd.testing.assert_series_equal(out["state"], labels.reindex(out.index), check_names=False)


def test_book_returns_lag_and_nan():
    index = pd.bdate_range("2001-01-01", periods=6)
    sleeves = pd.DataFrame({"a": 0.01, "b": 0.02}, index=index)
    exposure = pd.DataFrame({"a": [1, 0, 1, 1, 1, 1.0], "b": [0, 1, 1, np.nan, 1, 1.0]},
                            index=index)
    out = fj.book_returns(sleeves, exposure)
    assert out.iloc[:2].isna().all()
    assert out.iloc[2] == pytest.approx(0.005)
    assert out.iloc[3] == pytest.approx(0.01)
    assert out.iloc[4] == pytest.approx(0.015)
    assert np.isnan(out.iloc[5])


# --------------------------------------------------------------------------- descriptive
def test_participation_ratio_and_kappa_and_transitions():
    assert fj.participation_ratio(np.eye(6)) == pytest.approx(6.0)
    assert fj.participation_ratio(np.ones((4, 4))) == pytest.approx(1.0)
    a = np.array([1, 1, 0, 0, 1, 0])
    assert fj.cohen_kappa(a, a) == pytest.approx(1.0)
    assert fj.cohen_kappa(a, 1 - a) == pytest.approx(-1.0)
    s = pd.Series([1.0] * 126 + [0.0] * 126 + [1.0] * 252)
    assert fj.transitions_per_year(s) == pytest.approx(2 / 2.0)

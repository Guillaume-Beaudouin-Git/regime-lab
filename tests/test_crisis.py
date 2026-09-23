"""Construction tests for the crisis objects (`regime_lab/extensions/crisis.py`).

Synthetic data only: nothing here reads a state, a price file or a result.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regime_lab.extensions import crisis


def _market(n: int = 200, seed: int = 0) -> tuple[pd.Series, pd.Series]:
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2010-01-01", periods=n)
    returns = pd.Series(rng.normal(0, 0.01, n), index=index)
    vix = pd.Series(15 + 5 * np.abs(np.sin(np.arange(n) / 9)) + rng.normal(0, 1, n), index=index)
    return returns, vix


def test_tranche_starts_at_zero_and_ends_at_its_payoff():
    returns, vix = _market()
    values = crisis.tranche_values(returns, vix, opened=10)
    strike = crisis.monthly_variance(vix).iloc[10]
    realised = (returns.iloc[11:32] ** 2).sum()
    assert values.iloc[0] == pytest.approx(0.0, abs=1e-15)
    assert values.iloc[-1] == pytest.approx(strike - realised, rel=1e-12)
    assert len(values) == crisis.TENOR + 1


def test_ladder_is_the_average_of_its_tranches():
    returns, vix = _market()
    ladder = crisis.variance_swap_ladder(returns, vix)
    scale = 1.0 / crisis.monthly_variance(pd.Series([20.0])).iloc[0]
    tranches = [crisis.tranche_values(returns, vix, i).diff() for i in range(40, 100)]
    for s in range(70, 90):
        day = returns.index[s]
        open_ = [t[day] for t, i in zip(tranches, range(40, 100), strict=True)
                 if i < s <= i + crisis.TENOR]
        assert len(open_) == crisis.TENOR
        assert ladder.iloc[s] == pytest.approx(scale * np.mean(open_), rel=1e-10)


def test_ladder_is_causal():
    returns, vix = _market()
    base = crisis.variance_swap_ladder(returns, vix)
    shocked_r, shocked_v = returns.copy(), vix.copy()
    shocked_r.iloc[120:] *= 5
    shocked_v.iloc[120:] += 30
    after = crisis.variance_swap_ladder(shocked_r, shocked_v)
    pd.testing.assert_series_equal(base.iloc[:120], after.iloc[:120])


def _futures(sessions: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    expiries = pd.DatetimeIndex([sessions[5], sessions[25], sessions[45], sessions[65],
                                 sessions[85]])
    settles = pd.DataFrame(20.0, index=sessions, columns=expiries)
    return settles, expiries


def test_constant_futures_return_nothing():
    sessions = pd.bdate_range("2012-01-02", periods=100)
    settles, _ = _futures(sessions)
    out = crisis.vix_futures_constant_maturity(settles, sessions)
    live = out.dropna()
    assert len(live) > 50
    assert np.allclose(live.to_numpy(), 0.0)


def test_front_weight_falls_to_one_over_period_and_rolls():
    sessions = pd.bdate_range("2012-01-02", periods=100)
    _, expiries = _futures(sessions)
    cal = crisis.expiry_calendar(expiries, sessions)
    one_period = cal.loc[sessions[5]:sessions[24]]
    assert one_period["w_front"].iloc[0] == pytest.approx(1.0)
    assert one_period["w_front"].iloc[1] == pytest.approx(19 / 20)
    assert one_period["w_front"].iloc[-1] == pytest.approx(1 / 20)
    assert (one_period["front"] == sessions[25]).all()
    assert cal.loc[sessions[25], "front"] == sessions[45]
    assert cal.loc[sessions[25], "w_front"] == pytest.approx(1.0)


def test_known_move_gives_the_weighted_return():
    sessions = pd.bdate_range("2012-01-02", periods=100)
    settles, expiries = _futures(sessions)
    t = sessions[15]
    settles.loc[sessions[16]:, expiries[1]] = 22.0
    settles.loc[sessions[16]:, expiries[2]] = 21.0
    out = crisis.vix_futures_constant_maturity(settles, sessions)
    w1 = crisis.expiry_calendar(expiries, sessions).loc[t, "w_front"]
    expected = (w1 * 22.0 + (1 - w1) * 21.0) / 20.0 - 1.0
    assert out[sessions[16]] == pytest.approx(expected)
    assert out[sessions[17]] == pytest.approx(0.0)


def test_missing_settle_is_nan_not_zero():
    sessions = pd.bdate_range("2012-01-02", periods=100)
    settles, expiries = _futures(sessions)
    settles.loc[sessions[30], expiries[2]] = np.nan
    out = crisis.vix_futures_constant_maturity(settles, sessions)
    assert np.isnan(out[sessions[30]])
    assert np.isnan(out[sessions[31]])


def test_vol_target_is_lagged_and_capped():
    returns, _ = _market(300)
    weight = crisis.vol_target_weight(returns)
    shocked = returns.copy()
    shocked.iloc[200:] *= 10
    pd.testing.assert_series_equal(weight.iloc[:201], crisis.vol_target_weight(shocked).iloc[:201])
    tiny = crisis.vol_target_weight(returns * 1e-4)
    assert tiny.dropna().max() == pytest.approx(crisis.MAX_LEVERAGE)


def test_couple_stops_or_halves_stress_only():
    index = pd.bdate_range("2012-01-02", periods=6)
    weight = pd.Series(2.0, index=index)
    state = pd.Series([1.0, 0.0, 0.0, 1.0, np.nan, 0.0], index=index)
    stop = crisis.couple(weight, state, "stop")
    half = crisis.couple(weight, state, "half")
    assert stop.tolist() == [2.0, 0.0, 0.0, 2.0, 2.0, 0.0]
    assert half.tolist() == [2.0, 1.0, 1.0, 2.0, 2.0, 1.0]
    with pytest.raises(ValueError):
        crisis.couple(weight, state, "switch")


def test_lagged_state_reads_the_previous_session():
    index = pd.bdate_range("2012-01-02", periods=5)
    labels = pd.Series([1.0, 0.0, 1.0], index=index[[0, 2, 3]])
    lagged = crisis.lagged_state(labels, index)
    assert np.isnan(lagged.iloc[0])
    assert lagged.iloc[1:].tolist() == [1.0, 1.0, 0.0, 1.0]


def test_tail_rule_is_causal_and_calls_the_tail_stress():
    returns, _ = _market(600, seed=3)
    rule = crisis.volatility_tail_rule(returns)
    shocked = returns.copy()
    shocked.iloc[500:] *= 8
    pd.testing.assert_series_equal(rule.iloc[:500], crisis.volatility_tail_rule(shocked).iloc[:500])
    live = rule.dropna()
    assert 0.05 < (live == 0).mean() < 0.40

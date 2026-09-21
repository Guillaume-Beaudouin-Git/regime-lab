"""M2's cost schedule and M3's volatility target must be causal and correctly priced.

The causality test is the one that matters. A volatility target reads the book's
own realised volatility, so it is the natural place for a look-ahead to enter: an
estimator that is not lagged uses the very session it is sizing. The test
perturbs the future and asserts the past does not move, which catches that
whatever the implementation looks like.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regime_lab.extensions.vehicle import (
    COST_SCHEDULE,
    NO_FUTURES_VEHICLE,
    charge_by_instrument,
    cost_bps,
    cost_class,
    turnover,
    unscaled_weights,
    vol_targeted_book,
)

SESSIONS = pd.bdate_range("2010-01-04", periods=900)


def _prices(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    columns = ["^GSPC", "EURUSD=X", "CL=F", "IEF", "LQD"]
    steps = rng.normal(0.0004, 0.011, size=(len(SESSIONS), len(columns)))
    return pd.DataFrame(100.0 * np.exp(np.cumsum(steps, axis=0)), index=SESSIONS, columns=columns)


def test_the_book_at_t_cannot_see_the_price_at_t_or_after():
    prices = _prices()
    split = SESSIONS[600]

    baseline = vol_targeted_book(prices)["returns"]
    perturbed_prices = prices.copy()
    perturbed_prices.loc[split:] *= 1.35
    perturbed = vol_targeted_book(perturbed_prices)["returns"]

    before = baseline.loc[:split].iloc[:-1]
    pd.testing.assert_series_equal(
        before, perturbed.loc[:split].iloc[:-1], obj="the past moved when the future changed"
    )
    assert not baseline.loc[split:].equals(perturbed.loc[split:]), "the test perturbed nothing"


def test_the_multiplier_is_lagged_a_session_behind_its_own_estimate():
    prices = _prices(seed=3)
    built = vol_targeted_book(prices)

    # Rebuilt from the same inputs rather than by dividing the weights back out,
    # which is undefined wherever the multiplier is zero.
    unscaled = (unscaled_weights(prices) * prices.pct_change()).sum(axis=1)
    sigma = unscaled.rolling(63).std() * np.sqrt(252)
    expected = (0.10 / sigma.replace(0.0, np.nan)).shift(1).clip(upper=3.0).fillna(0.0)
    pd.testing.assert_series_equal(
        built["multiplier"], expected, check_names=False, rtol=1e-10
    )

    # And the lag is load-bearing: the unlagged estimate is a different series.
    unlagged = (0.10 / sigma.replace(0.0, np.nan)).clip(upper=3.0).fillna(0.0)
    assert not built["multiplier"].equals(unlagged), "the multiplier is not lagged"


def test_m3_removes_the_gross_pinning_and_the_pinned_book_still_has_it():
    built = vol_targeted_book(_prices(seed=5))
    pinned_gross = built["pinned_weights"].abs().sum(axis=1)
    live = pinned_gross[pinned_gross > 0]
    assert np.allclose(live, 1.0), "the pinned construction must keep gross at 1.0"

    m3_gross = built["weights"].abs().sum(axis=1)
    assert m3_gross[m3_gross > 0].std() > 1e-6, "M3 gross must be free to move"


def test_every_instrument_is_priced_by_the_exposure_it_represents():
    assert cost_class("^GSPC") == "equity_index"
    assert cost_class("EURUSD=X") == "currency"
    assert cost_class("DX-Y.NYB") == "currency"
    assert cost_class("CL=F") == "commodity"
    assert cost_class("IEF") == "fixed_income"
    for name in NO_FUTURES_VEHICLE:
        assert cost_class(name) == "cash_vehicle", f"{name} has no futures vehicle"


def test_the_all_cash_reading_prices_everything_at_the_non_futures_rate():
    columns = pd.Index(["^GSPC", "CL=F", "LQD"])
    normal = cost_bps(columns)
    literal = cost_bps(columns, all_cash=True)
    assert normal.loc["^GSPC"] == COST_SCHEDULE["equity_index"]["headline"]
    assert (literal == COST_SCHEDULE["cash_vehicle"]["headline"]).all()
    assert literal.sum() > normal.sum(), "the literal reading must be the dearer one"


def test_cost_is_charged_on_each_instrument_and_scales_with_its_own_rate():
    index = pd.bdate_range("2020-01-01", periods=4)
    weights = pd.DataFrame(
        {"^GSPC": [0.0, 1.0, 1.0, 0.0], "LQD": [0.0, 1.0, 1.0, 0.0]}, index=index
    )
    returns = pd.Series(0.0, index=index, name="book")

    charged = charge_by_instrument(returns, weights, cost_bps(weights.columns))
    expected = -(COST_SCHEDULE["equity_index"]["headline"]
                 + COST_SCHEDULE["cash_vehicle"]["headline"]) / 10_000.0
    assert charged.loc[index[1]] == pytest.approx(expected), "one crossing of each leg"
    assert charged.loc[index[3]] == pytest.approx(expected), "the exit crosses again"
    assert charged.loc[index[2]] == pytest.approx(0.0), "holding a position costs nothing"


def test_turnover_counts_both_sides_and_annualises():
    index = pd.bdate_range("2020-01-01", periods=253)
    flipping = pd.DataFrame({"^GSPC": [0.0, 1.0] * 126 + [0.0]}, index=index)
    assert turnover(flipping) > 200.0, "a daily flip is a very high turnover"
    assert turnover(pd.DataFrame({"^GSPC": [1.0] * 253}, index=index)) == 0.0

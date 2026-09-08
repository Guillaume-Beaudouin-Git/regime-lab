"""The two base strategies, frozen before any regime is estimated.

Fixing them now is a protocol decision, not a convenience. Choosing a strategy
after seeing which regimes the model produces — specifically, after seeing the
sign of the mean/variance relation in each state — would select on the very
quantity the study tests. Both strategies are reported whatever the result.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Modified duration used to turn a yield series into a total return.
DURATION = {"bond_us_10y": 7.5, "bond_us_long": 17.0}


def bond_return(yields: pd.Series, series_id: str) -> pd.Series:
    """Approximate a bond total return from a constant-maturity yield.

    ``r = y / 252 - D * dy``, with yields in percent. The carry term is the
    accrual over one session; the second term is the price move implied by the
    yield change at modified duration ``D``. This is the standard research proxy
    when a total-return index is unavailable, and it is a proxy: it ignores
    convexity and roll-down, and it assumes the duration is constant.
    """
    y = yields.astype(float) / 100.0
    return y.shift(1) / 252.0 - DURATION[series_id] * y.diff()


def sixty_forty(panel: pd.DataFrame, *, rebalance: str = "M") -> pd.Series:
    """Sixty percent equity, forty percent ten-year bond, rebalanced monthly.

    Weights drift with performance between rebalances rather than being reset
    every session: a daily-rebalanced 60/40 is a different, and quietly more
    profitable, strategy than the one everybody means by 60/40.
    """
    equity = panel["eq_us_large"].pct_change()
    bond = bond_return(panel["bond_us_10y"], "bond_us_10y")

    legs = pd.concat([equity.rename("eq"), bond.rename("bd")], axis=1).dropna()
    initial = np.array([0.60, 0.40])
    period = legs.index.to_period(rebalance)

    out = pd.Series(index=legs.index, dtype="float64")
    weights = initial.copy()
    previous = period[0]

    for date, row, current in zip(legs.index, legs.to_numpy(), period, strict=True):
        if current != previous:
            weights = initial.copy()
            previous = current
        out.loc[date] = float(weights @ row)
        grown = weights * (1.0 + row)
        weights = grown / grown.sum()

    return out.rename("sixty_forty")


def equal_risk_momentum(
    panel: pd.DataFrame,
    assets: list[str],
    *,
    lookback: int = 252,
    skip: int = 21,
    vol_window: int = 63,
) -> pd.Series:
    """Twelve-minus-one momentum, sized inversely to volatility, long-short.

    Signal at t-1, traded at t. The skip month is the standard control for
    short-horizon reversal.
    """
    returns = panel[assets].pct_change()
    momentum = panel[assets].shift(skip) / panel[assets].shift(lookback) - 1.0
    signal = np.sign(momentum)

    vol = returns.rolling(vol_window).std()
    weight = signal / vol
    weight = weight.div(weight.abs().sum(axis=1), axis=0).fillna(0.0)

    return (weight.shift(1) * returns).sum(axis=1).rename("equal_risk_momentum")


def vol_target(
    returns: pd.Series, *, target: float, halflife: int = 20, cap: float = 3.0
) -> pd.Series:
    """Scale ``returns`` to an ex-ante volatility target.

    The scaling uses only information up to t-1: the exponentially weighted
    volatility is shifted before it sizes the position. ``target`` must be
    calibrated on training data alone — calibrating it on the evaluation sample
    is exactly the look-ahead that overturned the volatility-managed portfolio
    literature.
    """
    sigma = returns.ewm(halflife=halflife, min_periods=halflife).std() * np.sqrt(252)
    leverage = (target / sigma.shift(1)).clip(upper=cap)
    return (leverage * returns).rename(f"{returns.name}_vt")

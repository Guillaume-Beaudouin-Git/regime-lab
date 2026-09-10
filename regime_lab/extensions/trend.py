"""A trend book to condition, built plainly so the conditioning is what is tested.

Twelve-minus-one time-series momentum, each instrument scaled to a common
ex-ante volatility, equal risk across the book. No selection, no optimisation,
no cross-sectional ranking: every choice here would otherwise be a parameter
competing with the conditioner for credit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

VOL_TARGET = 0.10
MAX_LEVERAGE = 3.0


def tsmom_signal(prices: pd.DataFrame, *, lookback: int = 252, skip: int = 21) -> pd.DataFrame:
    """Sign of the return over the lookback, skipping the most recent month."""
    momentum = prices.shift(skip) / prices.shift(lookback) - 1.0
    return np.sign(momentum)


def book(
    prices: pd.DataFrame, *, lookback: int = 252, skip: int = 21, vol_window: int = 63
) -> tuple[pd.Series, pd.DataFrame]:
    """Equal-risk trend book and its weights, signal lagged one session."""
    returns = prices.pct_change()
    signal = tsmom_signal(prices, lookback=lookback, skip=skip)
    sigma = returns.rolling(vol_window).std() * np.sqrt(252)
    scale = (VOL_TARGET / sigma).clip(upper=MAX_LEVERAGE)

    raw = signal * scale
    weights = raw.div(raw.abs().sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0).shift(1)
    return (weights * returns).sum(axis=1).rename("trend"), weights


def attenuator(
    returns: pd.DataFrame, *, long_window: int = 2520, smooth: int = 10
) -> pd.DataFrame:
    """The practitioner's continuous volatility multiplier, per instrument.

    ``L = 2 - 1.5 Q``, where Q is each instrument's current volatility as a
    percentile of its own long history. The point of the published version is
    what it is *not*: no threshold, no on/off, and a ten-day smoothing applied
    because the raw multiplier "would boost trading costs quite a lot".

    Per instrument and relative to its own past, never against the volatility of
    the market as a whole — a CTA measuring at portfolio level finds trend pays
    *more* in calm markets, so conditioning on market volatility cuts the wrong
    way.
    """
    sigma = returns.rolling(63).std() * np.sqrt(252)
    q = sigma.rolling(long_window, min_periods=756).rank(pct=True)
    return (2.0 - 1.5 * q).clip(lower=0.0, upper=2.0).rolling(smooth).mean().shift(1)

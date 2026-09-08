"""Standardisation that only ever looks backwards.

Scaling a feature by its full-sample mean and standard deviation is the quietest
look-ahead there is: every observation is then expressed relative to a
distribution that includes its own future, and a 2008 reading knows how extreme
2008 turned out to be. Everything here uses an expanding window, so a value is
scaled by what was known when it arrived.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def expanding_zscore(frame: pd.DataFrame, *, min_periods: int = 252) -> pd.DataFrame:
    """Z-score each column against its own expanding history."""
    mean = frame.expanding(min_periods=min_periods).mean()
    std = frame.expanding(min_periods=min_periods).std()
    return (frame - mean) / std.replace(0.0, np.nan)


def expanding_rank(frame: pd.DataFrame, *, min_periods: int = 252) -> pd.DataFrame:
    """Map each column to its expanding percentile rank, in ``[0, 1]``.

    Robust to the fat tails that make a z-score of a volatility feature spend
    2008 at eleven standard deviations, which no downstream model handles well.
    """
    return frame.expanding(min_periods=min_periods).rank(pct=True)


def winsorise(frame: pd.DataFrame, *, limit: float = 5.0) -> pd.DataFrame:
    """Clip standardised values to ``±limit``.

    Applied after standardisation, never inside a feature's own constructor:
    winsorising a raw series before it is scaled changes the scale itself, and
    the effect is invisible downstream.
    """
    return frame.clip(lower=-limit, upper=limit)

"""Asymmetry, drawdown and long-memory features.

This family was promised in the frozen charter and was missing from the first
build of the feature matrix — an omission against the protocol, not a decision,
and recorded as such.

It is also where the Hurst exponent lives. The exponent is popular in
practitioner writing on regimes and much weaker in the careful literature: on a
rolling window of a few hundred returns its standard error is large and the
estimate sits close to 0.5 for reasons that have more to do with the estimator
than with the market. That is an empirical claim, so it is measured here rather
than asserted — the admissibility test reports what the exponent shares with
volatility, and the sparse model decides whether to keep it.

The variance ratio is included beside it deliberately. It tests the same idea —
does the series trend or revert — with a sampling distribution that is known,
which the rolling Hurst estimate does not have.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _dfa_exponent(values: np.ndarray, scales: tuple[int, ...]) -> float:
    """Detrended fluctuation analysis on one window.

    The cumulative deviation series is split into boxes at several scales, each
    box is linearly detrended, and the exponent is the slope of log fluctuation
    against log scale. Detrending is what makes this usable on a series with
    drift, which rescaled range is not.
    """
    y = np.cumsum(values - values.mean())
    n = len(y)
    fluctuations: list[float] = []
    used: list[int] = []

    for scale in scales:
        boxes = n // scale
        if boxes < 4:
            continue
        trimmed = y[: boxes * scale].reshape(boxes, scale)
        x = np.arange(scale)
        # Least squares slope and intercept for every box at once.
        x_mean = x.mean()
        centred = x - x_mean
        denom = (centred**2).sum()
        slopes = (trimmed - trimmed.mean(axis=1, keepdims=True)) @ centred / denom
        intercepts = trimmed.mean(axis=1) - slopes * x_mean
        residual = trimmed - (slopes[:, None] * x + intercepts[:, None])
        fluctuations.append(float(np.sqrt((residual**2).mean())))
        used.append(scale)

    if len(used) < 3:
        return np.nan
    slope = np.polyfit(np.log(used), np.log(fluctuations), 1)[0]
    return float(slope)


def hurst(
    returns: pd.Series, *, window: int = 252, scales: tuple[int, ...] = (8, 16, 32, 64)
) -> pd.Series:
    """Rolling Hurst exponent by detrended fluctuation analysis.

    Above 0.5 the series is persistent, below it mean-reverting, at 0.5 it is a
    random walk. Read the level with suspicion: on a window this short the
    estimator's own noise is comparable with the range of values it produces.
    """
    values = returns.to_numpy(dtype=float)
    out = np.full(len(values), np.nan)
    for end in range(window, len(values) + 1):
        block = values[end - window : end]
        if np.isnan(block).any():
            continue
        out[end - 1] = _dfa_exponent(block, scales)
    return pd.Series(out, index=returns.index, name="asy_hurst")


def variance_ratio(returns: pd.Series, *, window: int = 252, q: int = 5) -> pd.Series:
    """Lo-MacKinlay variance ratio of ``q``-period to one-period variance.

    One means a random walk, above one trending, below one reverting. Same
    question as the Hurst exponent, with a distribution that is actually known.
    """
    var_1 = returns.rolling(window).var()
    var_q = returns.rolling(q).sum().rolling(window).var()
    return (var_q / (q * var_1)).rename(f"asy_vr_{q}")


def build(panel: pd.DataFrame, *, column: str = "eq_us_large") -> pd.DataFrame:
    """Assemble the asymmetry, drawdown and long-memory block."""
    r = np.log(panel[column]).diff()
    out = pd.DataFrame(index=panel.index)

    out["asy_skew_63"] = r.rolling(63).skew()
    out["asy_kurt_63"] = r.rolling(63).kurt()

    # Distance below the running high, over a rolling year: a state variable a
    # volatility measure does not carry, since a market can be calm and deeply
    # underwater at once.
    price = panel[column]
    out["asy_drawdown_252"] = price / price.rolling(252).max() - 1.0

    out["asy_hurst"] = hurst(r)
    out["asy_vr_5"] = variance_ratio(r, q=5)
    out["asy_vr_20"] = variance_ratio(r, q=20)

    return out

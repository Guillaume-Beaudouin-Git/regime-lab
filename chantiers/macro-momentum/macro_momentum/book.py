"""Turning signals into a portfolio, with the choices that are not free.

Three decisions here would each be a parameter if left open, so each is fixed
and stated: positions scale linearly with the signal rather than switching on a
threshold, every asset is volatility-targeted before aggregation so that the
book is not silently a bet on whichever asset happens to be most volatile, and
the signal is lagged one session.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Annualised volatility each asset is scaled to before aggregation.
ASSET_VOL_TARGET = 0.10

#: Cap on any single asset's leverage, so a quiet stretch cannot produce a
#: position that could not be held through the next shock.
MAX_LEVERAGE = 3.0

#: Round-trip cost in basis points, by asset class. Declared before results:
#: futures and large ETFs, not retail spread products.
COST_BPS = {"equities": 1.0, "bonds": 1.0, "commodities": 2.0, "dollar": 1.5}


def inverse_vol_weight(
    returns: pd.DataFrame, *, halflife: int = 60, target: float = ASSET_VOL_TARGET
) -> pd.DataFrame:
    """Scale each column to a common ex-ante volatility, using only the past."""
    sigma = returns.ewm(halflife=halflife, min_periods=halflife).std() * np.sqrt(252)
    return (target / sigma.shift(1)).clip(upper=MAX_LEVERAGE)


def build(
    returns: pd.DataFrame,
    signals: dict[str, pd.Series],
    asset_class: dict[str, str],
) -> tuple[pd.Series, pd.DataFrame]:
    """Return the book's daily return and the position matrix.

    Each asset takes its class's signal, scaled by that asset's own inverse
    volatility, then the book is normalised so gross exposure sums to one.
    """
    scale = inverse_vol_weight(returns)
    raw = pd.DataFrame(index=returns.index, columns=returns.columns, dtype="float64")
    for ticker in returns.columns:
        cls = asset_class.get(ticker)
        if cls in signals:
            raw[ticker] = signals[cls].reindex(returns.index) * scale[ticker]

    gross = raw.abs().sum(axis=1).replace(0.0, np.nan)
    weights = raw.div(gross, axis=0).fillna(0.0).shift(1)
    return (weights * returns).sum(axis=1).rename("book"), weights


def charge(book: pd.Series, weights: pd.DataFrame, asset_class: dict[str, str]) -> pd.Series:
    """Subtract per-class transaction costs from the book's return."""
    traded = weights.diff().abs()
    bps = pd.Series(
        {t: COST_BPS.get(asset_class.get(t, ""), 2.0) for t in weights.columns}
    )
    return (book - (traded * bps / 1e4).sum(axis=1).reindex(book.index).fillna(0.0)).rename(
        "book_net"
    )


def summary(x: pd.Series) -> dict[str, float]:
    """Annualised Sharpe, volatility and worst drawdown."""
    x = x.dropna()
    if len(x) < 2 or x.std(ddof=1) == 0:
        return {"sharpe": np.nan, "vol": np.nan, "max_drawdown": np.nan}
    curve = (1 + x).cumprod()
    return {
        "sharpe": float(x.mean() / x.std(ddof=1) * np.sqrt(252)),
        "vol": float(x.std(ddof=1) * np.sqrt(252)),
        "max_drawdown": float((curve / curve.cummax() - 1).min()),
    }

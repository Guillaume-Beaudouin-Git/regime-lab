"""How many independent forces are moving the market, and does it predict?

A CTA publishes a striking relation: the effective number of risk factors
driving its universe explains its trend performance almost linearly, with 2022 —
eight factors, the fewest outside 2008 — its best year, and 2023, above thirty,
its worst. The measure is Kritzman's absorption ratio wearing a different hat,
and it is already the study's least volatility-like feature.

**Their relation is contemporaneous.** It explains a year after the fact rather
than forecasting one. Whether the same quantity predicts is a separate question
that costs one lag to answer, and answering it decides whether this is a signal
or a dashboard.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def effective_factors(returns: pd.DataFrame, *, window: int = 252) -> pd.Series:
    """Perplexity of the eigenvalue spectrum: how many factors really matter.

    The eigenvalues of the covariance matrix are normalised to sum to one and
    read as a distribution; the exponential of its entropy is the number of
    equally-weighted factors that would produce the same concentration. One
    means a single factor explains everything, N means every direction
    contributes equally.

    Preferred to a top-k share because it needs no choice of k, and a choice of
    k is a parameter that would have to be justified.
    """
    values = returns.to_numpy(dtype=float)
    n, k = values.shape
    out = np.full(n, np.nan)

    for end in range(window, n + 1):
        block = values[end - window : end]
        block = block[~np.isnan(block).any(axis=1)]
        if len(block) < window // 2:
            continue
        eig = np.linalg.eigvalsh(np.cov(block, rowvar=False))
        eig = eig[eig > 0]
        if eig.size == 0:
            continue
        p = eig / eig.sum()
        out[end - 1] = float(np.exp(-(p * np.log(p)).sum()))

    return pd.Series(out, index=returns.index, name="effective_factors")


def absorption_ratio(returns: pd.DataFrame, *, window: int = 252, factors: int = 5) -> pd.Series:
    """Kritzman's share of variance in the leading factors, for comparison."""
    values = returns.to_numpy(dtype=float)
    n, k = returns.shape
    factors = min(factors, k)
    out = np.full(n, np.nan)

    for end in range(window, n + 1):
        block = values[end - window : end]
        block = block[~np.isnan(block).any(axis=1)]
        if len(block) < window // 2:
            continue
        eig = np.linalg.eigvalsh(np.cov(block, rowvar=False))[::-1]
        total = eig.sum()
        if total > 0:
            out[end - 1] = float(eig[:factors].sum() / total)

    return pd.Series(out, index=returns.index, name="absorption")

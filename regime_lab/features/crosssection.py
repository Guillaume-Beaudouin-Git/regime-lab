"""The non-volatility block, without which the placebo test has no content.

A regime model fed only return-derived volatility statistics can, by
construction, reproduce nothing but a volatility quantile — so testing it
against a volatility quantile is decided before it starts. This module supplies
the features that carry information a volatility measure does not: how widely
the cross-section is dispersed, how tightly it moves together, and how much of
its variance sits in a single factor.

Whether these features are genuinely orthogonal to volatility is not assumed. It
is measured, per feature, in ``scripts/build_features.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def absorption_ratio(returns: pd.DataFrame, *, window: int = 252, factors: int = 5) -> pd.Series:
    """Share of cross-sectional variance explained by the leading factors.

    Kritzman's absorption ratio. A market whose variance concentrates in a few
    factors is a market where diversification has quietly stopped working, which
    is a statement about structure rather than about the size of moves.
    """
    values = returns.to_numpy(dtype=float)
    n, k = values.shape
    factors = min(factors, k)
    out = np.full(n, np.nan)

    for end in range(window, n + 1):
        block = values[end - window : end]
        block = block[~np.isnan(block).any(axis=1)]
        if len(block) < window // 2:
            continue
        eigenvalues = np.linalg.eigvalsh(np.cov(block, rowvar=False))[::-1]
        total = eigenvalues.sum()
        if total > 0:
            out[end - 1] = eigenvalues[:factors].sum() / total

    return pd.Series(out, index=returns.index, name="xs_absorption")


def average_correlation(returns: pd.DataFrame, *, window: int = 63) -> pd.Series:
    """Mean off-diagonal pairwise correlation of the cross-section."""
    values = returns.to_numpy(dtype=float)
    n, k = values.shape
    out = np.full(n, np.nan)

    for end in range(window, n + 1):
        block = values[end - window : end]
        block = block[~np.isnan(block).any(axis=1)]
        if len(block) < window // 2:
            continue
        corr = np.corrcoef(block, rowvar=False)
        out[end - 1] = (corr.sum() - k) / (k * (k - 1))

    return pd.Series(out, index=returns.index, name="xs_avg_corr")


def build(industries: pd.DataFrame, size_bm: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    """Assemble the cross-sectional block from the Ken French panels."""
    out = pd.DataFrame(index=industries.index)

    # Dispersion: how far apart the cross-section is on a given day, smoothed.
    out["xs_dispersion_ind"] = industries.std(axis=1).rolling(21).mean()
    out["xs_dispersion_szbm"] = size_bm.std(axis=1).rolling(21).mean()

    out["xs_avg_corr"] = average_correlation(industries)
    out["xs_absorption"] = absorption_ratio(industries)
    out["xs_absorption_chg"] = out["xs_absorption"].diff(63)

    # Breadth: the share of the cross-section in an uptrend. Deliberately a
    # count, not a magnitude, so it does not simply re-express dispersion.
    cumulative = (1 + industries / 100.0).rolling(63).apply(np.prod, raw=True) - 1
    out["xs_breadth_63"] = (cumulative > 0).mean(axis=1)

    for name in ("ff_smb", "ff_hml", "ff_mom"):
        if name in factors:
            out[f"xs_{name}_63"] = factors[name].rolling(63).sum()

    return out

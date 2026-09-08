"""Stationary block bootstrap (Politis and Romano, 1994).

Resampling daily returns independently destroys the volatility clustering that
governs how uncertain a Sharpe estimate is, and reports confidence intervals far
too narrow. Blocks of geometric length preserve that dependence while keeping
the resampled series stationary.
"""

from __future__ import annotations

import numpy as np


def stationary_indices(n: int, mean_block: float, rng: np.random.Generator) -> np.ndarray:
    """Draw one bootstrap index path of length ``n``.

    Blocks start at a uniformly random position and continue with probability
    ``1 - 1 / mean_block``, wrapping at the end of the sample.
    """
    if mean_block <= 1:
        return rng.integers(0, n, size=n)

    p = 1.0 / mean_block
    idx = np.empty(n, dtype=np.int64)
    current = rng.integers(0, n)
    for i in range(n):
        idx[i] = current
        current = rng.integers(0, n) if rng.random() < p else (current + 1) % n
    return idx


def paired_sharpe_difference(
    a: np.ndarray,
    b: np.ndarray,
    *,
    mean_block: float,
    draws: int = 2_000,
    periods: int = 252,
    seed: int = 0,
) -> np.ndarray:
    """Bootstrap the distribution of ``Sharpe(a) - Sharpe(b)``.

    Both legs are resampled on the *same* index path, so the comparison stays
    paired: a draw that happens to contain a crash contains it for both.
    """
    if a.shape != b.shape:
        raise ValueError("legs must be the same length")

    rng = np.random.default_rng(seed)
    n = len(a)
    out = np.empty(draws)
    scale = np.sqrt(periods)

    for d in range(draws):
        idx = stationary_indices(n, mean_block, rng)
        ra, rb = a[idx], b[idx]
        sa = ra.mean() / ra.std(ddof=1) * scale
        sb = rb.mean() / rb.std(ddof=1) * scale
        out[d] = sa - sb

    return out

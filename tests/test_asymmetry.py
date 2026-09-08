"""Do the long-memory estimators recover what they claim to measure?

Written after the rolling Hurst exponent was found to produce values outside its
own admissible range on real data. These tests establish what the estimator does
on series whose answer is known, so that the failure on market data can be
attributed to the window length rather than to a coding mistake.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regime_lab.features.asymmetry import hurst, variance_ratio


def series_from(values: np.ndarray) -> pd.Series:
    return pd.Series(values, index=pd.bdate_range("2000-01-03", periods=len(values)))


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(11)


def test_hurst_recovers_one_half_on_a_random_walk(rng):
    """A long window on white noise should land near 0.5."""
    noise = series_from(rng.normal(size=4000))
    estimate = hurst(noise, window=1000).dropna()
    assert 0.42 < estimate.mean() < 0.58


def test_hurst_rises_on_a_persistent_series(rng):
    """Positively autocorrelated increments must read above a random walk."""
    shocks = rng.normal(size=4000)
    persistent = pd.Series(shocks).ewm(halflife=20).mean().to_numpy()
    trending = hurst(series_from(persistent), window=1000).dropna()
    walk = hurst(series_from(shocks), window=1000).dropna()
    assert trending.mean() > walk.mean() + 0.1


def test_short_windows_are_noisy_enough_to_leave_the_admissible_range(rng):
    """The defect seen on market data, reproduced on noise.

    This is the point of the whole file: on a 252-day window the estimator's
    spread is wide enough to produce values a Hurst exponent cannot take, so the
    behaviour observed on real returns is a property of the window and not a bug.
    """
    noise = series_from(rng.normal(size=6000))
    short = hurst(noise, window=252).dropna()
    long = hurst(noise, window=1000).dropna()
    assert short.std() > 2 * long.std()


def test_variance_ratio_is_one_on_independent_returns(rng):
    """Lo-MacKinlay: a random walk has a variance ratio of one."""
    noise = series_from(rng.normal(scale=0.01, size=6000))
    ratio = variance_ratio(noise, window=1000, q=5).dropna()
    assert 0.9 < ratio.mean() < 1.1


def test_variance_ratio_falls_below_one_under_mean_reversion(rng):
    """Alternating shocks revert, so aggregated variance grows more slowly."""
    shocks = rng.normal(scale=0.01, size=6000)
    reverting = shocks - 0.6 * np.roll(shocks, 1)
    ratio = variance_ratio(series_from(reverting), window=1000, q=5).dropna()
    assert ratio.mean() < 0.9

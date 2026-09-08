"""The phase 1 counterpart of the temporal integrity tests.

Standardising a feature against the full sample is the quietest look-ahead in
this kind of study: a 2008 reading then knows how extreme 2008 turned out to be.
These tests hold the expanding-window property to the same standard as the data
contract — a value computed for a past date must not move when later data
arrives.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regime_lab.features.standardise import expanding_rank, expanding_zscore, winsorise


@pytest.fixture
def series() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2000-01-03", periods=900)
    return pd.DataFrame({"x": rng.normal(size=len(dates))}, index=dates)


def test_zscore_is_stable_when_future_data_arrives(series):
    """The regression that matters: history must not be rewritten."""
    cut = 600
    early = expanding_zscore(series.iloc[:cut], min_periods=252)
    late = expanding_zscore(series, min_periods=252).iloc[:cut]
    pd.testing.assert_frame_equal(early, late)


def test_rank_is_stable_when_future_data_arrives(series):
    cut = 600
    early = expanding_rank(series.iloc[:cut], min_periods=252)
    late = expanding_rank(series, min_periods=252).iloc[:cut]
    pd.testing.assert_frame_equal(early, late)


def test_zscore_withholds_output_until_the_window_is_filled(series):
    scaled = expanding_zscore(series, min_periods=252)
    assert scaled["x"].iloc[:251].isna().all()
    assert scaled["x"].iloc[252:].notna().all()


def test_a_full_sample_zscore_would_have_failed_the_stability_test(series):
    """Guard against silently reverting to the convenient, wrong version."""
    cut = 600
    naive_early = (series.iloc[:cut] - series.iloc[:cut].mean()) / series.iloc[:cut].std()
    naive_late = ((series - series.mean()) / series.std()).iloc[:cut]
    assert not np.allclose(naive_early["x"], naive_late["x"])


def test_winsorise_clips_symmetrically(series):
    extreme = series.copy()
    extreme.iloc[0, 0] = 40.0
    extreme.iloc[1, 0] = -40.0
    clipped = winsorise(extreme, limit=5.0)
    assert clipped["x"].max() == 5.0
    assert clipped["x"].min() == -5.0

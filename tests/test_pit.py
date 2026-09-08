"""Temporal integrity: a panel built for a past date must not move.

These are the tests that make the point-in-time claim checkable rather than
asserted. If a loader ever starts leaking, one of them fails.
"""

from __future__ import annotations

import pandas as pd
import pytest

from regime_lab.data.pit import as_of, build_panel, realtime_trace, validate


def frame(rows: list[tuple[str, str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["series_id", "period", "available_at", "value"])


REVISED = frame(
    [
        # First print of the January figure, published mid-February.
        ("gdp", "2020-01-01", "2020-02-15", 100.0),
        # Revised a month later. A study that uses 101.5 in February is cheating.
        ("gdp", "2020-01-01", "2020-03-15", 101.5),
        ("gdp", "2020-02-01", "2020-03-15", 98.0),
    ]
)


def test_rejects_observation_available_before_its_period():
    bad = frame([("x", "2020-06-01", "2020-05-01", 1.0)])
    with pytest.raises(ValueError, match="available before its period"):
        validate(bad)


def test_as_of_returns_the_first_print_not_the_revision():
    assert as_of(REVISED, "2020-02-20")["gdp"] == pytest.approx(100.0)


def test_as_of_returns_nothing_before_first_publication():
    assert "gdp" not in as_of(REVISED, "2020-02-01").index


def test_as_of_takes_the_latest_period_once_published():
    assert as_of(REVISED, "2020-04-01")["gdp"] == pytest.approx(98.0)


def test_revision_is_visible_in_the_trace():
    trace = realtime_trace(REVISED)
    assert len(trace) == 2, "one row per publication event"
    assert list(trace["available_at"].dt.strftime("%Y-%m-%d")) == ["2020-02-15", "2020-03-15"]


def test_panel_is_stable_when_future_data_arrives():
    """The regression test that matters: appending later rows must not rewrite history."""
    dates = pd.date_range("2020-02-01", "2020-02-29", freq="D")
    before = build_panel(REVISED.iloc[:1], dates)

    late = pd.concat([REVISED, frame([("gdp", "2020-03-01", "2020-04-15", 95.0)])])
    after = build_panel(late, dates)

    pd.testing.assert_frame_equal(before, after)


def test_max_staleness_drops_a_discontinued_series():
    dates = pd.date_range("2020-02-01", "2021-06-01", freq="MS")
    panel = build_panel(REVISED, dates, max_staleness=pd.Timedelta(days=90))
    assert panel["gdp"].notna().any()
    assert pd.isna(panel["gdp"].iloc[-1]), "a year-old print must not be carried forward"

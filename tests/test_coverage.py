"""Reproduce the two silent data failures and prove the guard catches them.

A fix is only closed when the original bug fails the suite. Both of this
project's data failures returned a dataframe that looked complete, so both are
rebuilt here from their real shapes and fed to the collector's guard.
"""

from __future__ import annotations

import pandas as pd
import pytest

from regime_lab.data.coverage import CoverageError, check


def series(start: str, end: str, freq: str, name: str = "x") -> pd.DataFrame:
    dates = pd.date_range(start, end, freq=freq)
    return pd.DataFrame(
        {
            "series_id": name,
            "period": dates,
            "available_at": dates + pd.Timedelta(days=1),
            "value": range(len(dates)),
        }
    )


def test_a_full_length_daily_series_passes():
    frame = series("1990-01-01", "2026-09-01", "B")
    assert check(frame, name="x", start="1990-01-01", frequency="daily") is frame


def test_the_ice_bofa_failure_is_caught():
    """FRED served these credit spreads for the trailing two years only.

    796 rows starting in 2023 against a request from 1990: no error, no warning,
    a licence restriction. The file written to disk looked complete, and a credit
    feature built on it would have existed only after 2023.
    """
    truncated = series("2023-09-05", "2026-09-01", "B")
    with pytest.raises(CoverageError, match="starts 2023"):
        check(truncated, name="BAMLH0A0HYM2", start="1990-01-01", frequency="daily")


def test_the_row_cap_failure_is_caught():
    """The API capped a weekly series at 100,000 of its 579,084 rows.

    The truncation showed up as a series ending in 1995 rather than 2026, which
    the start check would miss on its own — the response begins exactly where it
    should. Density is what catches it.
    """
    capped = series("1990-01-05", "2026-08-28", "W")
    capped = capped.iloc[:300]  # what a mid-response cap leaves behind
    capped.loc[capped.index[-1], "period"] = pd.Timestamp("2026-08-28")
    with pytest.raises(CoverageError, match="of the weekly rate"):
        check(capped, name="NFCI", start="1990-01-05", frequency="weekly")


def test_a_declared_short_series_is_accepted():
    """Accepting a young series has to be somebody's written decision."""
    short = series("2022-11-11", "2026-08-28", "W")
    check(
        short, name="STLFSI4", start="1990-01-01", frequency="weekly", allow_short_from="2022-11-11"
    )


def test_an_undeclared_short_series_is_not():
    short = series("2022-11-11", "2026-08-28", "W")
    with pytest.raises(CoverageError):
        check(short, name="STLFSI4", start="1990-01-01", frequency="weekly")


def test_an_empty_response_is_caught():
    empty = series("1990-01-01", "1990-01-01", "B").iloc[:0]
    with pytest.raises(CoverageError, match="empty"):
        check(empty, name="x", start="1990-01-01", frequency="daily")

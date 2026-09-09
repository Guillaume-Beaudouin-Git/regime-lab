"""Refuse a series that does not cover the window it was asked for.

Both of this project's silent data failures were coverage failures, and neither
raised anything. FRED returned the ICE BofA credit spreads for the trailing two
years only — a licence restriction, no error, no warning — and the API capped a
financial-conditions series at 100,000 of its 579,084 rows while reporting the
real total in a field the loader ignored. In both cases the file written to disk
looked complete.

The lesson is not "check those two endpoints". It is that a collector must state
what it expects and fail when it does not get it, so that the next restriction —
on an endpoint nobody has thought about yet — cannot pass as data.

Two independent checks, because the two failures had different shapes: one
returned a *short* series, the other a *sparse* one over the full span.
"""

from __future__ import annotations

import pandas as pd

#: Observations per year, by declared frequency.
EXPECTED_PER_YEAR = {"daily": 252.0, "weekly": 52.0, "monthly": 12.0, "quarterly": 4.0}

#: A series may return this fraction of the expected observations before the
#: density check fires. Generous: holidays, early history and reporting gaps are
#: normal, a series returning a fifth of its rows is not.
MIN_DENSITY = 0.60

#: Tolerated delay between the requested start and the first observation.
START_TOLERANCE = pd.Timedelta(days=400)


class CoverageError(RuntimeError):
    """A source returned less than the window that was requested."""


def check(
    frame: pd.DataFrame,
    *,
    name: str,
    start: str | pd.Timestamp,
    frequency: str,
    allow_short_from: str | None = None,
) -> pd.DataFrame:
    """Validate that ``frame`` covers ``start`` onwards at ``frequency``.

    Args:
        frame: A point-in-time frame, already validated.
        name: Series name, for the error message.
        start: The first date the caller asked for.
        frequency: One of ``EXPECTED_PER_YEAR``.
        allow_short_from: Declare a genuinely short series by giving its real
            first date. This is the only way to accept one, so accepting it is a
            decision someone wrote down rather than an outcome nobody noticed.

    Raises:
        CoverageError: if the series starts too late, or is too sparse over the
            span it does cover.
    """
    if frequency not in EXPECTED_PER_YEAR:
        raise ValueError(f"unknown frequency {frequency!r}")
    if frame.empty:
        raise CoverageError(f"{name}: empty frame")

    requested = pd.Timestamp(start)
    first = frame["period"].min()
    last = frame["period"].max()

    expected_start = pd.Timestamp(allow_short_from) if allow_short_from else requested
    if first > expected_start + START_TOLERANCE:
        raise CoverageError(
            f"{name}: starts {first:%Y-%m-%d}, expected {expected_start:%Y-%m-%d}. "
            f"If the series really is this short, declare it in universe.py "
            f"rather than letting a truncated download pass as data."
        )

    span_years = max((last - first).days / 365.25, 1e-9)
    observed = len(frame.drop_duplicates(["series_id", "period"]))
    density = observed / (span_years * EXPECTED_PER_YEAR[frequency])
    if density < MIN_DENSITY:
        raise CoverageError(
            f"{name}: {observed:,} observations over {span_years:.1f} years is "
            f"{density:.0%} of the {frequency} rate. A source that silently caps "
            f"or thins a response looks exactly like this."
        )

    return frame

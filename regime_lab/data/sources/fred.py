"""Macro and financial series from FRED, on two deliberately different paths.

The difference between the paths is the substance of this module.

**Vintage path** (``fetch_first_release``). For series that get revised, the FRED
API returns the value *as first published*, with the publication date attached.
A panel rebuilt for March 2008 then holds the figure an observer actually had in
March 2008.

**Lag path** (``fetch_current``). For series that are never revised — market
rates, spreads, index levels — the current value *is* the historical value, and
only the publication delay matters. Here the lag path is not an approximation,
it is exact, and it avoids the API's vintage-date ceiling.

Which path a series takes is declared in ``universe.py`` rather than guessed, so
a reader can audit the choice. The reason the choice cannot be automated is that
ALFRED's vintage depth varies enormously by series: industrial production has
vintages back to 1990, initial claims only to 2009, and the fourth revision of
the St. Louis stress index only to 2022. Asking for vintages on a series that
lacks them silently returns a short history, not an error.

Two failure modes are guarded explicitly, because both are silent:

* the API caps a response at 100,000 rows and reports the true total in
  ``count`` — a caller that ignores ``count`` gets a truncated series that looks
  complete (measured on NFCI: 100,000 rows returned out of 579,084);
* a request spanning more than 2,000 vintage dates returns HTTP 400 rather than
  a partial answer.
"""

from __future__ import annotations

import io

import pandas as pd
import requests

from regime_lab.config import FRED_API_KEY, SAMPLE_START
from regime_lab.data.coverage import check
from regime_lab.data.pit import validate

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv"
FRED_API = "https://api.stlouisfed.org/fred/series/observations"

#: Conservative publication delays for the lag path.
PUBLICATION_LAG = {
    "daily": pd.Timedelta(days=1),
    "weekly": pd.Timedelta(days=7),
    "monthly": pd.Timedelta(days=45),
    "quarterly": pd.Timedelta(days=90),
}

#: FRED caps any single response at this many rows.
API_ROW_CAP = 100_000


def fetch_current(
    series_id: str,
    *,
    frequency: str = "daily",
    start: str = SAMPLE_START,
    timeout: int = 60,
    allow_short_from: str | None = None,
) -> pd.DataFrame:
    """Download the current vintage and stamp it with a publication lag.

    Exact for series that are never revised; a documented approximation
    otherwise, since the values are the revised ones.

    The result is checked for coverage before it is returned. This endpoint is
    the one that served the ICE BofA credit spreads for two years instead of
    thirty, with no error, so it is the last place to trust a row count.

    Raises:
        CoverageError: if the response is short or sparse and the shortfall was
            not declared through ``allow_short_from``.
    """
    if frequency not in PUBLICATION_LAG:
        raise ValueError(f"unknown frequency {frequency!r}; expected {list(PUBLICATION_LAG)}")

    response = requests.get(FRED_CSV, params={"id": series_id, "cosd": start}, timeout=timeout)
    response.raise_for_status()
    table = pd.read_csv(io.StringIO(response.text))
    if table.shape[1] != 2:
        raise RuntimeError(f"unexpected CSV shape {table.shape} for {series_id!r}")
    table.columns = ["period", "value"]

    table["series_id"] = series_id
    table["period"] = pd.to_datetime(table["period"])
    table["value"] = pd.to_numeric(table["value"], errors="coerce")
    table["available_at"] = table["period"] + PUBLICATION_LAG[frequency]
    return check(
        validate(table),
        name=series_id,
        start=start,
        frequency=frequency,
        allow_short_from=allow_short_from,
    )


def fetch_first_release(
    series_id: str,
    *,
    start: str = SAMPLE_START,
    api_key: str | None = None,
    timeout: int = 180,
) -> pd.DataFrame:
    """Download each observation as it was first published.

    ``available_at`` is the real publication date, so the result is a genuine
    real-time series rather than today's numbers wearing old dates.

    Raises:
        RuntimeError: if no key is set, if the response was truncated by the
            row cap, or if the series has no vintage history.
    """
    key = api_key or FRED_API_KEY
    if not key:
        raise RuntimeError(
            "FRED_API_KEY is not set. Copy .env.example to .env and paste a free key from "
            "https://fred.stlouisfed.org/docs/api/api_key.html"
        )

    response = requests.get(
        FRED_API,
        params={
            "series_id": series_id,
            "api_key": key,
            "file_type": "json",
            "observation_start": start,
            "realtime_start": "1776-07-04",
            "realtime_end": "9999-12-31",
            "output_type": 4,  # initial release only
        },
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()

    observations = payload.get("observations", [])
    if not observations:
        raise RuntimeError(f"no vintage history for {series_id!r}")

    total = int(payload.get("count", len(observations)))
    if total > len(observations):
        raise RuntimeError(
            f"{series_id!r}: response truncated at {len(observations):,} of {total:,} rows "
            f"(API cap {API_ROW_CAP:,}). Use the lag path for this series rather than "
            f"accepting a silently shortened history."
        )

    table = pd.DataFrame(observations).rename(
        columns={"date": "period", "realtime_start": "available_at"}
    )
    table["series_id"] = series_id
    table["value"] = pd.to_numeric(table["value"], errors="coerce")
    return validate(table.loc[:, ["series_id", "period", "available_at", "value"]])


def fetch_all_vintages(
    series_id: str,
    *,
    start: str = SAMPLE_START,
    api_key: str | None = None,
    timeout: int = 300,
) -> pd.DataFrame:
    """Download every revision of every observation.

    Used only to quantify how much revision actually moves a feature; too heavy
    and too limited by the vintage-date ceiling to be the default path.
    """
    key = api_key or FRED_API_KEY
    if not key:
        raise RuntimeError("FRED_API_KEY is not set")

    response = requests.get(
        FRED_API,
        params={
            "series_id": series_id,
            "api_key": key,
            "file_type": "json",
            "observation_start": start,
            "realtime_start": "1776-07-04",
            "realtime_end": "9999-12-31",
        },
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    observations = payload.get("observations", [])
    total = int(payload.get("count", len(observations)))
    if total > len(observations):
        raise RuntimeError(
            f"{series_id!r}: {total:,} rows exceeds the {API_ROW_CAP:,} cap; "
            f"full revision history unavailable in one request"
        )

    table = pd.DataFrame(observations).rename(
        columns={"date": "period", "realtime_start": "available_at"}
    )
    table["series_id"] = series_id
    table["value"] = pd.to_numeric(table["value"], errors="coerce")
    return validate(table.loc[:, ["series_id", "period", "available_at", "value"]])

"""The point-in-time data contract.

Every observation in this project carries two dates that must never be confused:

``period``
    The date the observation *describes* (the reference period of a macro
    release, or the session of a market bar).
``available_at``
    The first instant at which a real-time observer could have known the
    value. For a macro release this is its publication timestamp; for a
    revision it is the timestamp of that revision.

A feature may enter a model at time ``t`` only if ``available_at <= t``. Every
observation is stored, including superseded revisions, so that a panel can be
reconstructed exactly as it appeared on any past date. This is what makes
``build_panel`` reproducible: it never sees a value that had not been published.
"""

from __future__ import annotations

import pandas as pd

#: Canonical column order of a point-in-time frame.
PIT_COLUMNS = ["series_id", "period", "available_at", "value"]


def validate(frame: pd.DataFrame) -> pd.DataFrame:
    """Return ``frame`` normalised to the point-in-time contract.

    Raises:
        ValueError: if a required column is missing, a date is null, or an
            observation claims to have been available before the period it
            describes.
    """
    missing = [c for c in PIT_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"missing point-in-time columns: {missing}")

    out = frame.loc[:, PIT_COLUMNS].copy()
    out["series_id"] = out["series_id"].astype("string")
    out["period"] = pd.to_datetime(out["period"])
    out["available_at"] = pd.to_datetime(out["available_at"])
    out["value"] = pd.to_numeric(out["value"], errors="coerce")

    if out["period"].isna().any():
        raise ValueError("null period")
    if out["available_at"].isna().any():
        raise ValueError("null available_at")

    early = out["available_at"] < out["period"]
    if early.any():
        sample = out.loc[early, ["series_id", "period", "available_at"]].head()
        raise ValueError(f"observation available before its period:\n{sample}")

    out = out.dropna(subset=["value"])
    return out.sort_values(["series_id", "available_at", "period"], ignore_index=True)


def realtime_trace(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse revisions into the value a real-time observer would hold.

    Walks each series in publication order, keeping the most recent revision of
    every period, and emits the *latest period then known* after each release.
    The result has one row per publication event, with columns ``series_id``,
    ``available_at``, ``period`` and ``value``.
    """
    frame = validate(frame)
    rows: list[dict[str, object]] = []

    for series_id, block in frame.groupby("series_id", sort=True):
        latest_by_period: dict[pd.Timestamp, float] = {}
        newest_period: pd.Timestamp | None = None

        for available_at, event in block.groupby("available_at", sort=True):
            for period, value in zip(event["period"], event["value"], strict=True):
                latest_by_period[period] = value
                if newest_period is None or period > newest_period:
                    newest_period = period
            rows.append(
                {
                    "series_id": series_id,
                    "available_at": available_at,
                    "period": newest_period,
                    "value": latest_by_period[newest_period],
                }
            )

    trace = pd.DataFrame(rows, columns=["series_id", "available_at", "period", "value"])
    if trace.empty:
        return trace
    return trace.sort_values(["series_id", "available_at"], ignore_index=True)


def as_of(frame: pd.DataFrame, t: str | pd.Timestamp) -> pd.Series:
    """Return the value of every series as it stood at ``t``.

    Returns:
        A Series indexed by ``series_id``. Series with nothing published by
        ``t`` are absent rather than null, so a caller cannot silently treat an
        unpublished series as missing data.
    """
    t = pd.Timestamp(t)
    trace = realtime_trace(frame)
    known = trace.loc[trace["available_at"] <= t]
    if known.empty:
        return pd.Series(dtype="float64", name=t)
    last = known.groupby("series_id").tail(1)
    return pd.Series(last["value"].to_numpy(), index=last["series_id"].to_numpy(), name=t)


def build_panel(
    frame: pd.DataFrame,
    dates: pd.DatetimeIndex,
    *,
    max_staleness: pd.Timedelta | None = None,
) -> pd.DataFrame:
    """Assemble a leak-free panel of every series on ``dates``.

    Args:
        frame: Point-in-time observations, revisions included.
        dates: The evaluation grid, one row per date in the output.
        max_staleness: If given, a value older than this is dropped rather than
            carried forward. Guards against a discontinued series propagating a
            stale level across years of the sample.

    Returns:
        A DataFrame indexed by ``dates`` with one column per ``series_id``.
        Each cell holds the value that was published and current on that date.
    """
    dates = pd.DatetimeIndex(dates).sort_values()
    trace = realtime_trace(frame)
    if trace.empty:
        return pd.DataFrame(index=dates)

    grid = pd.DataFrame({"as_of": dates})
    columns: dict[str, pd.Series] = {}

    for series_id, block in trace.groupby("series_id", sort=True):
        block = block.loc[:, ["available_at", "value"]].sort_values("available_at")
        merged = pd.merge_asof(
            grid,
            block.rename(columns={"available_at": "as_of"}),
            on="as_of",
            direction="backward",
        )
        values = merged["value"].to_numpy()

        if max_staleness is not None:
            stamps = pd.merge_asof(
                grid,
                block.assign(stamp=block["available_at"]).loc[:, ["available_at", "stamp"]],
                left_on="as_of",
                right_on="available_at",
                direction="backward",
            )["stamp"]
            stale = (dates - stamps) > max_staleness
            values = pd.Series(values).mask(stale.to_numpy()).to_numpy()

        columns[str(series_id)] = pd.Series(values, index=dates)

    return pd.DataFrame(columns, index=dates)

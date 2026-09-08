"""Screen every series for the defects that fabricate signal.

Free downloads carry stale prints: a vendor that has no quote for a day often
repeats the previous one rather than reporting a gap. A repeated level produces
an exact zero return, which deflates measured volatility, inflates measured
autocorrelation, and can manufacture mean reversion that was never traded. The
screen below is one pass over the data and it is run before any feature is
built, per series and per year, because these rates are rarely uniform in time.
"""

from __future__ import annotations

import pandas as pd


def screen(panel: pd.DataFrame) -> pd.DataFrame:
    """Report staleness and gap statistics per series and per year.

    Args:
        panel: Wide panel indexed by date, one column per series.

    Returns:
        One row per (series, year) with the share of repeated levels, the share
        of exact zero changes, the longest run of repeats, and the share of
        missing observations.
    """
    rows: list[dict[str, object]] = []

    for name in panel.columns:
        column = panel[name]
        for year, block in column.groupby(column.index.year):
            present = block.dropna()
            if len(present) < 5:
                continue
            change = present.diff()
            repeated = change == 0
            run, longest = 0, 0
            for flag in repeated.to_numpy():
                run = run + 1 if flag else 0
                longest = max(longest, run)
            rows.append(
                {
                    "series_id": name,
                    "year": int(year),
                    "observations": int(len(present)),
                    "repeated_share": float(repeated.mean()),
                    "missing_share": float(block.isna().mean()),
                    "longest_repeat_run": int(longest),
                }
            )

    return pd.DataFrame(rows)


def flags(report: pd.DataFrame, *, repeated_max: float = 0.10, run_max: int = 10) -> pd.DataFrame:
    """Return the series-years that fail the screen.

    Defaults are deliberately loose: a daily price series with more than one
    repeated level in ten, or a run of ten unchanged prints, is not a price
    series that can carry a volatility feature.
    """
    bad = report["repeated_share"] > repeated_max
    stuck = report["longest_repeat_run"] > run_max
    return report.loc[bad | stuck].sort_values("repeated_share", ascending=False)

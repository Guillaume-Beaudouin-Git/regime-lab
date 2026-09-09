"""Populate the raw store. Idempotent: rerunning overwrites with a fresh manifest."""

from __future__ import annotations

import argparse

import pandas as pd
import requests

from regime_lab.config import FRED_API_KEY, SAMPLE_START
from regime_lab.data import store
from regime_lab.data.sources import fred, kenfrench, prices
from regime_lab.data.universe import (
    KNOWN_SHORT,
    MACRO_LAGGED,
    MACRO_VINTAGED,
    PRICES,
    PRICES_FRED,
)


def _report(kind: str, name: str, frame, note: str = "") -> None:
    # Count revisions per series: a multi-series frame has many rows per date
    # without any of them being a restatement.
    revisions = len(frame) - len(frame.drop_duplicates(["series_id", "period"]))
    print(
        f"{kind:<10} {name:<18} {len(frame):>8,} rows  {revisions:>6,} rev  "
        f"{frame['period'].min():%Y-%m-%d} to {frame['period'].max():%Y-%m-%d}  {note}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=SAMPLE_START)
    parser.add_argument("--only", choices=["prices", "macro", "panels"], default=None)
    args = parser.parse_args()

    want = lambda block: args.only in (None, block)  # noqa: E731

    if want("prices"):
        frame = prices.fetch(PRICES, start=args.start)
        store.write(frame, "prices", "cross_asset", origin="Yahoo Finance, adjusted close")
        _report("prices", "cross_asset", frame, f"{frame['series_id'].nunique()} series")

        blocks = []
        for name, series_id in PRICES_FRED.items():
            block = fred.fetch_current(
                series_id,
                frequency="daily",
                start=args.start,
                allow_short_from=KNOWN_SHORT.get(series_id),
            )
            block["series_id"] = name
            blocks.append(block)
        if blocks:
            frame = pd.concat(blocks, ignore_index=True)
            store.write(frame, "prices", "fred_daily", origin="FRED daily prices, unrevised")
            _report("prices", "fred_daily", frame, f"{frame['series_id'].nunique()} series")

    if want("macro"):
        if not FRED_API_KEY:
            print("  ! no FRED_API_KEY: vintaged series will fall back to the lag path")

        for name, series_id in MACRO_VINTAGED.items():
            try:
                frame = fred.fetch_first_release(series_id, start=args.start)
                note = "first release"
            except (RuntimeError, requests.RequestException) as exc:
                # Only a missing vintage history or a transport failure falls
                # back. Anything else is a bug and must not be papered over by
                # quietly producing a lesser dataset.
                print(f"macro      {name:<18} vintage path failed ({exc}); using lag path")
                frame = fred.fetch_current(
                    series_id,
                    frequency="monthly",
                    start=args.start,
                    allow_short_from=KNOWN_SHORT.get(series_id),
                )
                note = "LAG FALLBACK"
            frame["series_id"] = name
            store.write(frame, "macro", name, origin=f"FRED {series_id}, {note}")
            _report("macro", name, frame, note)

        for name, (series_id, frequency, revised) in MACRO_LAGGED.items():
            frame = fred.fetch_current(
                series_id,
                frequency=frequency,
                start=args.start,
                allow_short_from=KNOWN_SHORT.get(series_id),
            )
            frame["series_id"] = name
            note = f"lag {frequency}" + ("" if revised else ", unrevised so exact")
            store.write(frame, "macro", name, origin=f"FRED {series_id}, {note}")
            _report("macro", name, frame, note)

    if want("panels"):
        for panel in kenfrench.PANELS:
            frame = kenfrench.fetch(panel, start=args.start)
            store.write(frame, "panels", panel, origin=f"Ken French library, {panel}")
            _report("panels", panel, frame, f"{frame['series_id'].nunique()} series")


if __name__ == "__main__":
    main()

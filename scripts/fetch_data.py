"""Populate the raw store. Idempotent: rerunning overwrites with a fresh manifest."""

from __future__ import annotations

import argparse

from regime_lab.config import FRED_API_KEY, SAMPLE_START
from regime_lab.data import store
from regime_lab.data.sources import fred, kenfrench, prices
from regime_lab.data.universe import MACRO_LAGGED, MACRO_VINTAGED, PRICES


def _report(kind: str, name: str, frame, note: str = "") -> None:
    revisions = len(frame) - frame["period"].nunique()
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

    if want("macro"):
        if not FRED_API_KEY:
            print("  ! no FRED_API_KEY: vintaged series will fall back to the lag path")

        for name, series_id in MACRO_VINTAGED.items():
            try:
                frame = fred.fetch_first_release(series_id, start=args.start)
                note = "first release"
            except Exception as exc:
                print(f"macro      {name:<18} vintage path failed ({exc}); using lag path")
                frame = fred.fetch_current(series_id, frequency="monthly", start=args.start)
                note = "LAG FALLBACK"
            frame["series_id"] = name
            store.write(frame, "macro", name, origin=f"FRED {series_id}, {note}")
            _report("macro", name, frame, note)

        for name, (series_id, frequency, revised) in MACRO_LAGGED.items():
            frame = fred.fetch_current(series_id, frequency=frequency, start=args.start)
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

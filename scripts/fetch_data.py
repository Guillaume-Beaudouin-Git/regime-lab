"""Populate the raw store. Idempotent: rerunning overwrites with a fresh manifest."""

from __future__ import annotations

import argparse

import pandas as pd

from regime_lab.config import FRED_API_KEY, SAMPLE_START
from regime_lab.data import store
from regime_lab.data.sources import fred, prices
from regime_lab.data.universe import MACRO, PRICES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=SAMPLE_START)
    parser.add_argument("--skip-prices", action="store_true")
    parser.add_argument("--skip-macro", action="store_true")
    args = parser.parse_args()

    if not args.skip_prices:
        frame = prices.fetch(PRICES, start=args.start)
        store.write(frame, "prices", "cross_asset", origin="yahoo finance, adjusted close")
        span = f"{frame['period'].min():%Y-%m-%d} to {frame['period'].max():%Y-%m-%d}"
        print(f"prices     {len(PRICES):>3} series  {len(frame):>7,} rows  {span}")

    if not args.skip_macro:
        path = "ALFRED vintages" if FRED_API_KEY else "current vintage + lag (FALLBACK)"
        for name, (series_id, frequency) in MACRO.items():
            try:
                frame = fred.fetch(series_id, frequency=frequency, start=args.start)
            except Exception as exc:  # a single dead series must not stop the run
                print(f"macro      {name:<18} FAILED  {exc}")
                continue
            frame["series_id"] = name
            store.write(frame, "macro", name, origin=f"FRED {series_id}, {path}")
            revisions = len(frame) - frame["period"].nunique()
            first = frame["period"].min()
            short = " SHORT HISTORY" if first > pd.Timestamp(args.start) + pd.DateOffset(years=2) else ""
            print(
                f"macro      {name:<18} {len(frame):>7,} rows  "
                f"{revisions:>6,} revisions  {first:%Y-%m-%d} to "
                f"{frame['period'].max():%Y-%m-%d}{short}"
            )
        if not FRED_API_KEY:
            print("\n  ! no FRED_API_KEY: macro is on the fallback path, revision leak not removed")


if __name__ == "__main__":
    main()

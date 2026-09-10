"""Populate the store, reusing the point-in-time contract from regime-lab."""

from __future__ import annotations

from regime_lab.data.sources import fred, prices

from macro_momentum import config
from macro_momentum.universe import ASSETS, LAGGED, VINTAGED

START = "1990-01-01"


def main() -> None:
    frame = prices.fetch({t: t for t in ASSETS}, start=START)
    config.write(frame, "mm_prices", "assets", origin="Yahoo Finance, adjusted close")
    print(f"prix       {frame['series_id'].nunique():>3} actifs  {len(frame):>8,} lignes  "
          f"{frame['period'].min():%Y-%m} a {frame['period'].max():%Y-%m}")

    for name, series_id in VINTAGED.items():
        f = fred.fetch_first_release(series_id, start=START)
        f["series_id"] = name
        config.write(f, "mm_macro", name, origin=f"FRED {series_id}, first release")
        print(f"macro      {name:<12} {len(f):>8,} lignes  premiere publication reelle")

    for name, (series_id, freq) in LAGGED.items():
        f = fred.fetch_current(series_id, frequency=freq, start=START)
        f["series_id"] = name
        config.write(f, "mm_macro", name, origin=f"FRED {series_id}, jamais revise")
        print(f"marche     {name:<12} {len(f):>8,} lignes  decalage exact")


if __name__ == "__main__":
    main()

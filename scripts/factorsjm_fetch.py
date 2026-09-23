"""Download the full histories the per-factor jump model study needs (piste `factorsjm`).

What is fetched, and why it is not already on disk:

- the five Fama-French factors, daily, from 1963-07-01 (Ken French data library,
  ``F-F_Research_Data_5_Factors_2x3_daily``). `data/raw/panels/factors_5.parquet` starts
  in 1990, which leaves 24 years out of sample; the full file leaves about 48;
- the 1-year and 10-year constant-maturity Treasury yields (FRED ``DGS1``, ``DGS10``),
  daily from 1962. They stand in for the 2-year yield and the 10-year minus 2-year
  spread of Shu and Mulvey's market features, which only start in 1976.

The momentum factor is already on disk from 1926 (`data/raw/crisis/french_umd.parquet`).

The window 1971-04 to 1989-12 was the AQR sealed holdout; it was opened for the
`longhist` study on 2026-09-23 (`docs/PROTOCOL_FREEZE.md`, commit ``d1bc935``). This
study reads it too and declares it in `docs/PRESPEC_FACTORSJM.md`.

Nothing here reads a state or a strategy return.

Usage:
    .venv/bin/python scripts/factorsjm_fetch.py
"""

from __future__ import annotations

from regime_lab.data import store
from regime_lab.data.sources import fred, kenfrench

SOURCE = "factorsjm"


def main() -> None:
    factors = kenfrench.fetch("factors_5", start="1963-01-01")
    path = store.write(factors, SOURCE, "factors_5_full",
                       origin="Ken French, F-F_Research_Data_5_Factors_2x3_daily, percent")
    print(f"factors_5_full  {len(factors):>7,} rows  {factors['period'].min():%Y-%m-%d} -> "
          f"{factors['period'].max():%Y-%m-%d}  {path.name}")

    for series_id in ("DGS1", "DGS10"):
        rates = fred.fetch_current(series_id, start="1962-01-02")
        path = store.write(rates, SOURCE, series_id.lower(),
                           origin=f"FRED {series_id}, current vintage, lag 1 day")
        print(f"{series_id:<15} {len(rates):>7,} rows  {rates['period'].min():%Y-%m-%d} -> "
              f"{rates['period'].max():%Y-%m-%d}  {path.name}")


if __name__ == "__main__":
    main()

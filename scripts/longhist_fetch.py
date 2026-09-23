"""Download the public series of the long-history study (`docs/PRESPEC_LONGHIST.md`).

Everything is free, public and available without an API key:

1. **Ken French data library**, daily, full history from 1926-07-01:
   - `F-F_Research_Data_Factors_daily` : Mkt-RF, SMB, HML, RF;
   - `49_Industry_Portfolios_daily`    : value-weighted section only;
   - `25_Portfolios_5x5_Daily`         : value-weighted section only.
   Returns are stored in percent, as published; ``-99.99`` and ``-999`` mean missing
   and are dropped. A daily close is known at the close: ``available_at = period``.
2. **FRED**, monthly, through `fredgraph.csv` (no key):
   - `AAA`, `BAA` : Moody's seasoned corporate yields, from 1919-01. The value of
     month m is a monthly average, complete at the end of month m and published in
     the first days of month m+1: ``available_at`` = the 8th calendar day of m+1;
   - `INDPRO` : industrial production, from 1919-01, **current vintage** (revised).
     Used in one declared sensitivity only. Released around mid-month m+1, stamped
     ``available_at`` = the last calendar day of m+1;
   - `USREC` : NBER recession indicator, from 1854. **Validation only**, never a
     feature. Stamped ``available_at = period``, which is irrelevant since no model
     reads it.

The files land in `data/raw/longhist/` (gitignored) through `regime_lab.data.store`,
with a manifest carrying the SHA-256, the row count and the period range.

Usage:
    .venv/bin/python scripts/longhist_fetch.py              # a few minutes
    .venv/bin/python scripts/longhist_fetch.py --fred-only  # the FRED series only
"""

from __future__ import annotations

import io
import sys
import time
import zipfile

import pandas as pd
import requests

from regime_lab.data import store

HEADERS = {"User-Agent": "Mozilla/5.0 (research; regime-lab)"}
FRENCH = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/{name}_CSV.zip"
FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
SOURCE = "longhist"


def get(url: str, headers: dict | None = None) -> bytes:
    for attempt in range(4):
        try:
            resp = requests.get(url, headers=HEADERS if headers is None else headers, timeout=120)
        except requests.RequestException:
            time.sleep(2.0 * (attempt + 1))
            continue
        if resp.status_code == 200:
            return resp.content
        time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"not reachable: {url}")


def french_first_table(name: str) -> pd.DataFrame:
    """The first daily table of a Ken French zip (value-weighted where there are two)."""
    with zipfile.ZipFile(io.BytesIO(get(FRENCH.format(name=name)))) as z:
        member = next(n for n in z.namelist() if n.lower().endswith(".csv"))
        lines = z.read(member).decode("latin-1").splitlines()
    head = next(i for i, line in enumerate(lines) if line.strip().startswith(","))
    body = []
    for line in lines[head + 1:]:
        if not line.strip()[:8].isdigit():
            break
        body.append(line)
    table = pd.read_csv(io.StringIO("\n".join([lines[head], *body])))
    table = table.rename(columns={table.columns[0]: "period"})
    table.columns = [str(c).strip() for c in table.columns]
    table["period"] = pd.to_datetime(table["period"].astype(str).str.strip(), format="%Y%m%d")
    return table.set_index("period").apply(pd.to_numeric, errors="coerce")


def long_frame(wide: pd.DataFrame, prefix: str) -> pd.DataFrame:
    frame = wide.where(wide > -99.0).stack().rename("value").reset_index()
    frame.columns = ["period", "series_id", "value"]
    frame["series_id"] = prefix + frame["series_id"].str.lower().str.replace(r"[^a-z0-9]+", "_",
                                                                             regex=True)
    frame["available_at"] = frame["period"]
    return frame


def fred_monthly(sid: str, lag: str) -> pd.DataFrame:
    # FRED times out on a browser-like User-Agent without a browser behind it and
    # answers the library default at once (measured 2026-09-23).
    table = pd.read_csv(io.BytesIO(get(FRED.format(sid=sid), headers={})))
    table.columns = ["period", "value"]
    table["period"] = pd.to_datetime(table["period"])
    table["value"] = pd.to_numeric(table["value"], errors="coerce")
    table = table.dropna()
    start_next = table["period"] + pd.offsets.MonthBegin(1)
    if lag == "day8_next":
        available = start_next + pd.Timedelta(days=7)
    elif lag == "end_next":
        available = start_next + pd.offsets.MonthEnd(0)
    elif lag == "none":
        available = table["period"]
    else:
        raise ValueError(lag)
    return pd.DataFrame({"series_id": f"fred_{sid.lower()}", "period": table["period"],
                         "available_at": available, "value": table["value"]})


def report(frame: pd.DataFrame, label: str) -> None:
    print(f"{label:<12} {frame['series_id'].nunique():>3} series  {len(frame):>9,} rows  "
          f"{frame['period'].min():%Y-%m-%d} -> {frame['period'].max():%Y-%m-%d}")


def main() -> None:
    fetch_french = "--fred-only" not in sys.argv
    panels = {} if not fetch_french else {
        "ff3_daily": ("F-F_Research_Data_Factors_daily", "ff_"),
        "industry49_daily": ("49_Industry_Portfolios_daily", "ind_"),
        "size_bm25_daily": ("25_Portfolios_5x5_Daily", "p25_"),
    }
    for name, (remote, prefix) in panels.items():
        frame = long_frame(french_first_table(remote), prefix)
        store.write(frame, SOURCE, name, origin=f"Ken French, {remote}")
        report(frame, name)

    fred = pd.concat([
        fred_monthly("AAA", "day8_next"),
        fred_monthly("BAA", "day8_next"),
        fred_monthly("INDPRO", "end_next"),
    ], ignore_index=True)
    store.write(fred, SOURCE, "fred_monthly", origin="FRED fredgraph.csv, AAA BAA INDPRO")
    report(fred, "fred_monthly")

    usrec = fred_monthly("USREC", "none")
    store.write(usrec, SOURCE, "usrec", origin="FRED fredgraph.csv, USREC (validation only)")
    report(usrec, "usrec")


if __name__ == "__main__":
    main()

"""Download both layers. The unbiased one first, because it sets the benchmark."""

from __future__ import annotations

import io
import zipfile

import pandas as pd
import requests
import yfinance as yf
from regime_lab.data.pit import validate

from reversal_lab import config
from reversal_lab.universe import FRENCH, TICKERS

BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp"
START = "1990-01-01"


def french_table(text: str) -> pd.DataFrame:
    """First dated table of a Ken French CSV; later tables are other weightings."""
    lines = text.splitlines()
    head = next(i for i, line in enumerate(lines) if line.strip().startswith(","))
    body = []
    for line in lines[head + 1 :]:
        s = line.strip()
        if not s or not s[:8].isdigit():
            break
        body.append(line)
    table = pd.read_csv(io.StringIO("\n".join([lines[head], *body])))
    table = table.rename(columns={table.columns[0]: "period"})
    table["period"] = pd.to_datetime(table["period"].astype(str), format="%Y%m%d")
    return table.set_index("period")


def main() -> None:
    for name, filename in FRENCH.items():
        resp = requests.get(f"{BASE}/{filename}", timeout=120)
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
            csv = next(n for n in z.namelist() if n.lower().endswith(".csv"))
            table = french_table(z.read(csv).decode("latin-1")).loc[START:]
        table.columns = [f"{name}_{c.strip().lower().replace(' ', '_')}" for c in table.columns]
        for sentinel in (-99.99, -999.0):
            table = table.mask(table == sentinel)
        long = table.stack(future_stack=True).rename("value").reset_index()
        long.columns = ["period", "series_id", "value"]
        long["available_at"] = long["period"]
        config.write(validate(long), "french", name, origin=f"Ken French, {filename}")
        print(f"french     {name:<20} {len(long):>8,} lignes  {table.shape[1]:>3} series  "
              f"{table.index.min():%Y-%m} a {table.index.max():%Y-%m}")

    raw = yf.download(TICKERS, start=START, auto_adjust=True, progress=False, threads=True)
    close = raw["Close"].dropna(axis=1, how="all")
    close = close.loc[:, close.notna().sum() > 1500]
    long = close.stack(future_stack=True).rename("value").reset_index()
    long.columns = ["period", "series_id", "value"]
    long["available_at"] = long["period"]
    config.write(validate(long.dropna()), "stocks", "us_large", origin="Yahoo, BIAISE SURVIE")
    print(f"\nactions    {close.shape[1]:>3} titres  {len(long):>8,} lignes  "
          f"BIAIS DE SURVIE, couche de conditionnement uniquement")


if __name__ == "__main__":
    main()

"""Download the cross-country panel: sovereign yields, policy rates, currencies."""

from __future__ import annotations

import pandas as pd
import yfinance as yf
from regime_lab.data.sources import fred

from macro_momentum import config
from macro_momentum.countries import COUNTRIES, FX, long_rate, short_rate

START = "1990-01-01"


def main() -> None:
    blocks = []
    for code in COUNTRIES:
        for kind, sid in (("long", long_rate(code)), ("short", short_rate(code))):
            try:
                f = fred.fetch_current(sid, frequency="monthly", start=START,
                                       allow_short_from="2002-01-01")
            except Exception as exc:
                print(f"   {code} {kind:<6} indisponible ({type(exc).__name__})")
                continue
            f["series_id"] = f"{kind}_{code}"
            blocks.append(f)
    rates = pd.concat(blocks, ignore_index=True)
    config.write(rates, "h2", "rates", origin="FRED / OECD, monthly sovereign and policy rates")
    n_long = rates["series_id"].str.startswith("long_").groupby(rates["series_id"]).any().sum()
    print(f"taux       {rates['series_id'].nunique():>3} series  {len(rates):>7,} lignes  "
          f"{rates['period'].min():%Y-%m} a {rates['period'].max():%Y-%m}")

    tickers = {**{f"fx_{k}": v for k, v in FX.items()},
               **{f"eq_{c}": t for c, (_, _, t) in COUNTRIES.items() if t}}
    raw = yf.download(list(tickers.values()), start=START, auto_adjust=True,
                      progress=False, threads=True)["Close"]
    inverse = {v: k for k, v in tickers.items()}
    raw = raw.rename(columns=inverse)
    raw = raw.loc[:, [c for c in tickers if c in raw.columns]]
    raw = raw.loc[:, raw.notna().sum() > 1500]

    long = raw.stack(future_stack=True).rename("value").reset_index()
    long.columns = ["period", "series_id", "value"]
    long["available_at"] = long["period"]
    config.write(long.dropna(), "h2", "markets", origin="Yahoo Finance, FX and equity indices")
    fxn = sum(1 for c in raw.columns if c.startswith("fx_"))
    eqn = sum(1 for c in raw.columns if c.startswith("eq_"))
    print(f"marches    {fxn:>3} devises, {eqn} indices actions  {len(long):>7,} lignes")


if __name__ == "__main__":
    main()

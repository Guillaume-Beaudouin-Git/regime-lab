"""Download US indicators with genuine vintage history, plus the assets traded."""

from __future__ import annotations

import yfinance as yf
from regime_lab.data.sources import fred

from macro_momentum import config

START = "1998-01-01"

#: ``{name: (fred id, transform)}``. Twelve monthly releases plus weekly claims
#: is about a hundred and forty events a year — twice the density the source
#: model works with, which is the point of testing here rather than on a pair
#: whose free history carries a dozen.
INDICATORS = {
    "payems": ("PAYEMS", "growth"),
    "unrate": ("UNRATE", "diff"),
    "cpi": ("CPIAUCSL", "growth"),
    "core_cpi": ("CPILFESL", "growth"),
    "indpro": ("INDPRO", "growth"),
    "houst": ("HOUST", "growth"),
    "permit": ("PERMIT", "growth"),
    "dgorder": ("DGORDER", "growth"),
    "retail": ("RSAFS", "growth"),
    "sentiment": ("UMCSENT", "diff"),
    "capacity": ("TCU", "diff"),
    "hours": ("AWHMAN", "diff"),
    "claims": ("ICSA", "growth"),
}

#: Declared in the pre-specification. All four reported whatever they show.
ASSETS = {"usd": "DX-Y.NYB", "equity": "^GSPC", "bond": "^TNX", "gold": "GC=F"}


def main() -> None:
    for name, (series_id, _) in INDICATORS.items():
        try:
            frame = fred.fetch_all_vintages(series_id, start=START)
        except Exception as exc:
            print(f"   {name:<12} millesimes complets indisponibles ({type(exc).__name__}), "
                  f"repli premiere publication")
            frame = fred.fetch_first_release(series_id, start=START)
        frame["series_id"] = name
        config.write(frame, "h3_macro", name, origin=f"FRED {series_id}, millesimes")
        periods = frame["period"].nunique()
        print(f"   {name:<12} {len(frame):>7,} lignes  {periods:>4} periodes  "
              f"{frame['period'].min():%Y-%m} a {frame['period'].max():%Y-%m}")

    raw = yf.download(list(ASSETS.values()), start=START, auto_adjust=True,
                      progress=False, threads=True)["Close"]
    raw = raw.rename(columns={v: k for k, v in ASSETS.items()})
    long = raw.loc[:, list(ASSETS)].stack(future_stack=True).rename("value").reset_index()
    long.columns = ["period", "series_id", "value"]
    long["available_at"] = long["period"]
    config.write(long.dropna(), "h3_assets", "prices", origin="Yahoo Finance")
    print(f"\n   actifs       {len(long):>7,} lignes  {raw.shape[1]} series  "
          f"{raw.index.min():%Y-%m} a {raw.index.max():%Y-%m}")


if __name__ == "__main__":
    main()

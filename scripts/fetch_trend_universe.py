"""A broad cross-asset universe, for the extensions only.

The study's own universe is ten price series chosen to feed regime features. A
concentration measure needs a cross-section wide enough for the eigenvalue
spectrum to mean something, and a trend book needs breadth to be worth
conditioning, so the extensions use their own panel and never touch the study's.
"""

from __future__ import annotations

import warnings

import yfinance as yf

from regime_lab.config import CACHE

warnings.filterwarnings("ignore")

TICKERS = ["^GSPC", "^NDX", "^RUT", "^STOXX50E", "^N225", "^FTSE", "^GDAXI", "^FCHI", "^AEX", "^HSI", "^AXJO", "^GSPTSE", "^IBEX", "TLT", "IEF", "SHY", "AGG", "TIP", "LQD", "HYG", "EMB", "BWX", "GC=F", "SI=F", "CL=F", "NG=F", "HG=F", "ZC=F", "ZS=F", "ZW=F", "KC=F", "PL=F", "CT=F", "SB=F", "HO=F", "DX-Y.NYB", "EURUSD=X", "JPY=X", "GBPUSD=X", "AUDUSD=X", "CAD=X", "CHF=X", "NZDUSD=X", "SEK=X", "NOK=X", "MXN=X"]


def main() -> None:
    raw = yf.download(TICKERS, start="1995-01-01", auto_adjust=True,
                      progress=False, threads=True)["Close"]
    keep = raw.loc[:, raw.notna().sum() > 3000].dropna(how="all")
    keep = keep.loc[keep.index >= "1998-01-01"].ffill(limit=5)
    # Instruments enter as they list rather than the panel starting when the last
    # one does: requiring a complete rectangle cost eight years of history and
    # the whole of the 2000-2002 bear market, which is where a trend book earns.
    keep = keep.loc[keep.notna().sum(axis=1) >= 20]
    target = CACHE / "trend_universe.parquet"
    keep.to_parquet(target)
    print(f"{keep.shape[1]} instruments retenus sur {len(TICKERS)}, "
          f"{len(keep):,} jours communs, {keep.index.min():%Y-%m} a {keep.index.max():%Y-%m}")


if __name__ == "__main__":
    main()

"""Daily market data, mapped onto the point-in-time contract.

A daily bar is knowable at the close of its own session, so ``available_at``
equals ``period``. The one-session delay between observing a signal and trading
it is imposed later, in the strategy layer, and is deliberately not baked in
here: burying the lag in the data loader is how it silently gets applied twice,
or not at all.
"""

from __future__ import annotations

import pandas as pd
import yfinance as yf

from regime_lab.data.pit import validate


def fetch(tickers: dict[str, str], start: str, end: str | None = None) -> pd.DataFrame:
    """Download adjusted closes for ``tickers`` mapped ``{series_id: yahoo_symbol}``."""
    symbols = list(tickers.values())
    raw = yf.download(
        symbols,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    if raw is None or raw.empty:
        raise RuntimeError("yfinance returned no rows")

    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    if isinstance(close, pd.Series):
        close = close.to_frame(symbols[0])
    if len(symbols) == 1 and list(close.columns) == ["Close"]:
        close.columns = symbols

    inverse = {symbol: series_id for series_id, symbol in tickers.items()}
    close = close.rename(columns=inverse)
    close = close.loc[:, [c for c in tickers if c in close.columns]]

    long = (
        close.stack(future_stack=True)
        .rename("value")
        .reset_index()
        .rename(columns={close.index.name or "Date": "period", "level_1": "series_id"})
    )
    long.columns = ["period", "series_id", "value"]
    long["available_at"] = long["period"]
    return validate(long)

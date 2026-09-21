"""M2 and M3 — the vehicle's cost schedule, and the portfolio volatility target.

Every parameter here is fixed in `docs/PRESPEC_TREND_VEHICLE.md` §5 and §6 before
any return was computed. Nothing in this module is a choice made after seeing a
result, and the two places where the locked text needed interpreting are named
below rather than resolved silently.

**M3, and what it removes.** `extensions.trend.book` divides by gross exposure
every session, pinning it to 1.0, so portfolio volatility floats — measured at
0.00% to 15.14% against a 4.54% mean. M3 drops that division and scales the whole
book by `0.10 / σ_book` instead, with σ estimated over 63 sessions of realised
book returns and lagged one session. The per-instrument risk scaling is untouched.

**The cost class, which the locked text left ambiguous.** §6 prices "any
instrument kept on a non-futures vehicle" at 7.5 bp against 1.0 to 1.5 for
futures. After the M2 narrowing of 2026-09-21 every instrument keeps its stored
price series, so a literal reading would put all 46 at 7.5 bp and erase the
distinction the schedule exists to draw.

The reading taken here is that "kept on a non-futures vehicle" is a property of
the *market*, not of which series happens to sit in the cache: an S&P exposure
trades as ES whether or not this repository holds the ES series, and an LQD
exposure has no futures contract and must be traded as an ETF whatever series is
used to model it. That is also what D2 measured when it priced the book
all-futures at 1.0 bp. The literal reading is not discarded — it is reported as
`ALL_CASH`, the most conservative schedule available, so a reader can see the
whole span.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from regime_lab.extensions.trend import MAX_LEVERAGE, VOL_TARGET, tsmom_signal

#: Instruments whose exposure has no liquid futures contract and must be traded
#: as cash or an ETF. NOK and SEK have no liquid CME or ICE contract; the rest are
#: credit, global-aggregate or inflation-linked baskets with no futures equivalent.
NO_FUTURES_VEHICLE = (
    "NOK=X", "SEK=X",
    "AGG", "BWX", "EMB", "HYG", "LQD", "TIP",
)

#: Round-trip basis points by class, docs/PRESPEC_TREND_VEHICLE.md §6.
#: `headline` is the decision column. A result whose sign moves between `headline`
#: and `conservative` is reported as undecided, never as a pass.
COST_SCHEDULE = {
    "equity_index": {"headline": 1.0, "conservative": 2.0, "stress": 5.0},
    "currency": {"headline": 1.0, "conservative": 2.0, "stress": 5.0},
    "fixed_income": {"headline": 1.0, "conservative": 2.0, "stress": 5.0},
    "commodity": {"headline": 1.5, "conservative": 3.0, "stress": 7.5},
    "cash_vehicle": {"headline": 7.5, "conservative": 15.0, "stress": 30.0},
}


def cost_class(instrument: str) -> str:
    """Which row of §6 an instrument pays, by the exposure it represents."""
    if instrument in NO_FUTURES_VEHICLE:
        return "cash_vehicle"
    if instrument.startswith("^"):
        return "equity_index"
    if instrument.endswith("=X") or instrument == "DX-Y.NYB":
        return "currency"
    if instrument.endswith("=F"):
        return "commodity"
    return "fixed_income"


def cost_bps(columns: pd.Index, *, column: str = "headline", all_cash: bool = False) -> pd.Series:
    """Round-trip basis points per instrument.

    ``all_cash`` applies the literal reading of §6 — every instrument at the
    non-futures rate — which is the most conservative schedule and is reported
    alongside rather than instead.
    """
    if all_cash:
        return pd.Series(COST_SCHEDULE["cash_vehicle"][column], index=columns, dtype=float)
    return pd.Series(
        {name: COST_SCHEDULE[cost_class(name)][column] for name in columns}, dtype=float
    )


def unscaled_weights(
    prices: pd.DataFrame, *, lookback: int = 252, skip: int = 21, vol_window: int = 63
) -> pd.DataFrame:
    """Per-instrument trend exposure at equal risk, lagged one session.

    Identical to the first half of ``trend.book``: the sign of a 12-minus-1 trend
    scaled to 10% annualised volatility per instrument and capped at 3.0. What is
    *not* done here is the division by gross exposure, which is the pinning M3
    removes.
    """
    returns = prices.pct_change()
    signal = tsmom_signal(prices, lookback=lookback, skip=skip)
    sigma = returns.rolling(vol_window).std() * np.sqrt(252)
    scale = (VOL_TARGET / sigma).clip(upper=MAX_LEVERAGE)
    return (signal * scale).shift(1).fillna(0.0)


def portfolio_scalar(
    book_returns: pd.Series, *, window: int = 63, target: float = VOL_TARGET,
    cap: float = MAX_LEVERAGE,
) -> tuple[pd.Series, pd.Series]:
    """The M3 multiplier, and whether the cap bound on each session.

    ``target / σ``, σ being the rolling standard deviation of realised book
    returns annualised, lagged one session so the multiplier applied on *t* is
    known at the close of *t-1*.
    """
    sigma = book_returns.rolling(window).std() * np.sqrt(252)
    uncapped = (target / sigma.replace(0.0, np.nan)).shift(1)
    return uncapped.clip(upper=cap).fillna(0.0), (uncapped > cap).fillna(False)


def vol_targeted_book(
    prices: pd.DataFrame,
    *,
    lookback: int = 252,
    skip: int = 21,
    vol_window: int = 63,
    target: float = VOL_TARGET,
    cap: float = MAX_LEVERAGE,
) -> dict[str, object]:
    """The M3 book: equal risk per instrument, one volatility target on the whole.

    Returns the gross book return, the weights actually held, the multiplier, the
    cap-binding flag and the pinned-gross book that M3 replaces, so every table
    can be produced from one causal construction rather than several.
    """
    returns = prices.pct_change()
    raw = unscaled_weights(prices, lookback=lookback, skip=skip, vol_window=vol_window)

    # The book M3 would produce at a multiplier of one. Causal: `raw` is lagged.
    unscaled_returns = (raw * returns).sum(axis=1)
    multiplier, cap_binds = portfolio_scalar(
        unscaled_returns, window=vol_window, target=target, cap=cap
    )
    weights = raw.mul(multiplier, axis=0)

    # The construction M3 replaces, for a like-for-like comparison.
    gross = raw.abs().sum(axis=1).replace(0.0, np.nan)
    pinned_weights = raw.div(gross, axis=0).fillna(0.0)

    return {
        "returns": (weights * returns).sum(axis=1).rename("m3"),
        "weights": weights,
        "multiplier": multiplier.rename("multiplier"),
        "cap_binds": cap_binds.rename("cap_binds"),
        "pinned_returns": (pinned_weights * returns).sum(axis=1).rename("pinned"),
        "pinned_weights": pinned_weights,
    }


def charge_by_instrument(
    book_returns: pd.Series, weights: pd.DataFrame, bps: pd.Series
) -> pd.Series:
    """Subtract the cost of trading each instrument's exposure.

    §5 charges on the absolute change in each instrument's exposure, so a book
    whose leverage path itself moves pays for that movement — which is why a
    volatility-targeted book turns over more than a pinned one.
    """
    traded = weights.diff().abs().fillna(0.0)
    rates = bps.reindex(weights.columns).fillna(COST_SCHEDULE["cash_vehicle"]["headline"])
    drag = traded.mul(rates / 10_000.0, axis=1).sum(axis=1)
    return (book_returns - drag).rename(book_returns.name)


def turnover(weights: pd.DataFrame) -> float:
    """Annualised two-sided turnover, the quantity §6 calibrates against."""
    return float(weights.diff().abs().sum(axis=1).mean() * 252)

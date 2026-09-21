"""M3 — the portfolio volatility target, and whether the construction is sound.

Construction integrity only. No Sharpe, no verdict, no kill criterion is decided
here: an instrument is checked before it is read. The evaluation against K1-K6 is
`run_m3_evaluation.py`.

What this answers, in the order that matters:

1. Does M3 hit the 10% target the pre-registration locks, on all three panels?
2. K6 — does the leverage cap bind? The locked phrase "portfolio leverage cap 3.0"
   has two readings that disagree completely, and both are measured here rather
   than one being assumed.
3. What does the book's gross exposure actually do, including its tail? A cap that
   never binds is only reassuring if the distribution it fails to bind on is also
   reported.
4. What does M3 cost in turnover against the pinned construction it replaces?

Usage:
    .venv/bin/python scripts/run_m3_construction.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from regime_lab.config import CACHE
from regime_lab.extensions.vehicle import (
    MAX_LEVERAGE,
    VOL_TARGET,
    cost_class,
    turnover,
    vol_targeted_book,
)

RULE = "=" * 78

#: docs/PRESPEC_TREND_VEHICLE.md §3.
SAMPLE = slice("2003-07-17", "2026-09-10")

PANELS = {
    "stored": "trend_universe",
    "M1 headline": "trend_universe_m1",
    "M1 sensitivity": "trend_universe_m1_verified",
}


def active_window(series: pd.Series) -> pd.Series:
    """Drop the burn-in before the first session the book holds anything."""
    live = series.dropna()
    return live[live.index >= live.ne(0.0).idxmax()]


def main() -> None:
    print(RULE)
    print("M3  PORTFOLIO VOLATILITY TARGET — CONSTRUCTION INTEGRITY")
    print(RULE)

    universe = pd.read_parquet(CACHE / "trend_universe.parquet")
    classes = pd.Series({c: cost_class(c) for c in universe.columns})
    print("\n   cost classes, §6 applied by the exposure rather than by the series held:")
    for name, count in classes.value_counts().items():
        print(f"     {name:<14} {count:>2}")
    print(f"     of which no futures vehicle exists: "
          f"{', '.join(sorted(classes[classes == 'cash_vehicle'].index))}")

    print(f"\n{RULE}\n1.  DOES M3 HIT ITS TARGET?  locked at {VOL_TARGET:.0%} annualised\n")
    print(f"   {'panel':<18} {'realised vol':>13} {'multiplier median':>19} {'turnover/yr':>13}")
    books = {}
    for label, name in PANELS.items():
        prices = pd.read_parquet(CACHE / f"{name}.parquet").loc[SAMPLE]
        built = vol_targeted_book(prices)
        returns = active_window(built["returns"])
        books[label] = (built, returns, prices)
        print(f"   {label:<18} {returns.std() * np.sqrt(252):>12.2%} "
              f"{built['multiplier'].reindex(returns.index).median():>19.3f} "
              f"{turnover(built['weights'].loc[returns.index]):>12.2f}x")

    built, returns, prices = books["M1 headline"]
    gross = built["weights"].abs().sum(axis=1).reindex(returns.index)
    multiplier = built["multiplier"].reindex(returns.index)

    print(f"\n{RULE}\n2.  K6 — THE TWO READINGS OF A LEVERAGE CAP OF {MAX_LEVERAGE}\n")
    on_multiplier = float((multiplier >= MAX_LEVERAGE).mean())
    on_gross = float((gross > MAX_LEVERAGE).mean())
    print(f"   (a) cap on the MULTIPLIER, the reading §5 names       binds "
          f"{on_multiplier:>7.2%} of sessions")
    print(f"   (b) cap on the BOOK'S GROSS EXPOSURE                  binds "
          f"{on_gross:>7.2%} of sessions")
    print("\n   K6: a cap binding on more than 5% of sessions is a design failure,")
    print("   not a result. Reading (b) fails it, and fails it for a reason worth")
    print("   stating: capping gross at 3.0 pulls realised volatility below the")
    print("   target M3 exists to hit, so the modification stops doing its job.")
    scaled = built["weights"].div((gross / MAX_LEVERAGE).clip(lower=1.0), axis=0)
    capped = (scaled * prices.pct_change()).sum(axis=1).reindex(returns.index)
    print(f"\n   realised volatility under (a) {returns.std() * np.sqrt(252):>6.2%}"
          f"   under (b) {capped.std() * np.sqrt(252):>6.2%}   target {VOL_TARGET:.0%}")

    print(f"\n{RULE}\n3.  WHAT THE INERT CAP LEAVES UNCAPPED\n")
    print(f"   gross exposure   median {gross.median():.3f}   p05 {gross.quantile(0.05):.3f}"
          f"   p95 {gross.quantile(0.95):.3f}   max {gross.max():.1f}")
    for threshold in (3, 5, 10, 20, 40):
        sessions = int((gross > threshold).sum())
        print(f"     above {threshold:>2}x   {(gross > threshold).mean():>7.2%}  "
              f"({sessions:>4} sessions)")
    worst = gross.nlargest(3)
    print("\n   The tail is one episode, not a regime: "
          + ", ".join(f"{d:%Y-%m-%d} at {v:.1f}x" for d, v in worst.items()) + ".")
    print("   It is the standard volatility-target pathology — a 63-session estimate")
    print("   of book volatility collapses and the multiplier spikes. Declared here")
    print("   because 'the cap never binds' would otherwise read as reassurance.")

    print(f"\n{RULE}\n4.  WHAT M3 COSTS IN TURNOVER\n")
    pinned = active_window(built["pinned_returns"])
    rolling_pinned = pinned.rolling(63).std() * np.sqrt(252)
    rolling_m3 = returns.rolling(63).std() * np.sqrt(252)
    print(f"   {'construction':<24} {'realised vol':>13} {'rolling vol min':>16}"
          f" {'max':>8} {'turnover/yr':>13}")
    print(f"   {'pinned gross (published)':<24} {pinned.std() * np.sqrt(252):>12.2%}"
          f" {rolling_pinned.min():>15.2%} {rolling_pinned.max():>7.2%}"
          f" {turnover(built['pinned_weights'].loc[returns.index]):>12.2f}x")
    print(f"   {'M3 volatility target':<24} {returns.std() * np.sqrt(252):>12.2%}"
          f" {rolling_m3.min():>15.2%} {rolling_m3.max():>7.2%}"
          f" {turnover(built['weights'].loc[returns.index]):>12.2f}x")
    print("\n   §6 anticipated turnover rising from 15.15x to 24.71x on the published")
    print("   device. It rises further here, because that device scaled a book whose")
    print("   own volatility was already near its target while M3 levers a 4.1% book")
    print("   to 10%. §6 says higher turnover is expected and is not a reason to")
    print("   alter the construction; the gap against its calibration is reported")
    print("   rather than absorbed.")
    print(f"\n{RULE}")


if __name__ == "__main__":
    main()

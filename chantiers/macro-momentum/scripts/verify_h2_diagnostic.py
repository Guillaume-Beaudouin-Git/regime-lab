"""Recompute the H2 diagnostic numbers that have NO committed code.

RESULTS_H2.md publishes a dispersion table, a euro-member standard deviation
block and a pre/post-1999 split. Nothing in macro-momentum computes any of
them: run_h2.py stops at the nine cells. This script reads the same stored
panel and recomputes them, so the published figures can be confirmed or
contradicted rather than trusted.

Read-only with respect to the repository. Writes nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from macro_momentum.countries import COUNTRIES  # noqa: E402
from scripts.run_h2 import (  # noqa: E402
    build_returns,
    build_signals,
    cross_sectional_ic,
    monthly_panel,
)

warnings.filterwarnings("ignore")
RULE = "=" * 78

PERIODS = {
    "1990-1998": ("1990-01-01", "1998-12-31"),
    "1999-2007": ("1999-01-01", "2007-12-31"),
    "2008-2015": ("2008-01-01", "2015-12-31"),
    "2016-2026": ("2016-01-01", "2026-12-31"),
}

EURO = [c for c, (_, ccy, _) in COUNTRIES.items() if ccy == "EUR"]


def frame(rates: pd.DataFrame, kind: str) -> pd.DataFrame:
    cols = {c: rates[f"{kind}_{c}"] for c in COUNTRIES if f"{kind}_{c}" in rates}
    return pd.DataFrame(cols)


def main() -> None:
    rates, markets = monthly_panel()

    print(RULE)
    print("H2 DIAGNOSTIC — recomputed from data/raw/h2/rates.parquet")
    print(RULE)
    print(f"\npanel: {rates.index.min():%Y-%m} to {rates.index.max():%Y-%m}, "
          f"{rates.shape[1]} series")
    print(f"euro members in COUNTRIES: {len(EURO)} -> {sorted(EURO)}")

    short, long = frame(rates, "short"), frame(rates, "long")
    print(f"short-rate series: {short.shape[1]}   long-rate series: {long.shape[1]}")

    # --- dispersion table -------------------------------------------------
    print("\n" + RULE)
    print("\nTABLE 1 — cross-sectional dispersion of sovereign rates")
    print("  (per month: standard deviation across countries; then averaged)\n")
    print(f"  {'period':<12}{'short: mean':>13}{'median':>9}{'n ctry':>8}"
          f"{'long: mean':>13}{'median':>9}{'n ctry':>8}")
    for label, (a, b) in PERIODS.items():
        row = f"  {label:<12}"
        for f in (short, long):
            w = f.loc[a:b]
            xs = w.std(axis=1, ddof=1).dropna()
            n = w.notna().sum(axis=1)
            row += f"{xs.mean():>12.2f}%{xs.median():>8.2f}%{n.mean():>8.1f}"
        print(row)

    # --- euro bloc --------------------------------------------------------
    print("\n" + RULE)
    print("\nTABLE 2 — standard deviation of the euro members' OWN short rates")
    print(f"  ({len(EURO)} sovereigns that adopted the euro)\n")
    eu = short[[c for c in EURO if c in short.columns]]
    print(f"  countries present with a short rate: {eu.shape[1]} -> "
          f"{sorted(eu.columns)}")
    for label, (a, b) in PERIODS.items():
        w = eu.loc[a:b]
        xs = w.std(axis=1, ddof=1).dropna()
        n = w.notna().sum(axis=1)
        mx = xs.max() if len(xs) else np.nan
        print(f"  {label:<12} mean {xs.mean():>6.2f}%   median {xs.median():>6.2f}%"
              f"   max {mx:>6.2f}%   countries/month {n.mean():>4.1f}")

    n_exact_zero = (eu.loc["1999-01-01":].std(axis=1, ddof=1).dropna() == 0.0).mean()
    print(f"\n  months after 1999-01 where that std is EXACTLY 0.0: "
          f"{100 * n_exact_zero:.1f}%")

    # --- pre / post euro split -------------------------------------------
    print("\n" + RULE)
    print("\nTABLE 3 — the nine cells, split at the euro\n")
    signals, returns = build_signals(rates, markets), build_returns(rates, markets)
    codes = [c for c in COUNTRIES if f"long_{c}" in rates]

    def signal_for(theme: str, cls: str) -> pd.DataFrame:
        if cls == "fx" and theme == "currency":
            return signals["currency_by_ccy"]
        if cls == "fx":
            by = {}
            for code in codes:
                ccy = COUNTRIES[code][1]
                if ccy != "USD" and ccy not in by and code in signals[theme]:
                    by[ccy] = signals[theme][code]
            return pd.DataFrame(by)
        return signals[theme]

    for label, (a, b) in {"before the euro 1990-1998": ("1990-01-01", "1998-12-31"),
                          "after the euro 1999-2026": ("1999-01-01", "2026-12-31")}.items():
        live, best = {}, (None, 0.0)
        for theme in ("policy", "slope", "currency"):
            for cls in ("bonds", "fx", "equities"):
                sig = signal_for(theme, cls).loc[a:b]
                fwd = returns[cls].shift(-1).loc[a:b]
                r = cross_sectional_ic(sig, fwd)
                if np.isfinite(r["t"]):
                    live[(theme, cls)] = r
                    if abs(r["t"]) > abs(best[1]):
                        best = ((theme, cls), r["t"])
        strong = [k for k, v in live.items() if abs(v["t"]) > 1.96]
        print(f"  {label}")
        print(f"     usable cells (>=60 months): {len(live)} of 9")
        print(f"     cells above |t| 1.96      : {len(strong)}")
        print(f"     largest |t|               : {abs(best[1]):.2f}  {best[0]}")
        for k, v in sorted(live.items(), key=lambda kv: -abs(kv[1]["t"])):
            print(f"       {k[0]:<9} {k[1]:<9} IC {v['ic']:+.4f}  t {v['t']:+.2f}"
                  f"  n={v['n']:>3}  breadth={v['breadth']}")
        print()

    print(RULE)


if __name__ == "__main__":
    main()

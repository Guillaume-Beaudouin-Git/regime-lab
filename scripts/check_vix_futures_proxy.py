"""Check the rebuilt one-month VIX futures series against a traded product.

`extensions.crisis.vix_futures_constant_maturity` rebuilds a one-month constant-
maturity VIX futures position from the CFE daily settlements. The official S&P 500 VIX
Short-Term Futures Index is not free; the iPath VXX note tracks it, net of a fee, and
its Yahoo history (auto-adjusted for the reverse splits) starts in January 2018. This
compares daily returns on the common sessions. No state, no strategy result: a data
integrity check of `docs/PRESPEC_CRISE.md`, run after the reading and declared so.

VXX prints at the 4:00 pm New York equity close, the futures settle fifteen minutes
later (3:15 pm Chicago), so daily correlations below one are expected, and a move that
runs past the equity close lands on different days; weekly returns show whether the
two track.

Usage:
    .venv/bin/python scripts/check_vix_futures_proxy.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from regime_lab.extensions import crisis

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_crisis_coupling import VIXF_START, market, vix_settles  # noqa: E402


def main() -> None:
    sessions = market()["spx"].index
    sessions = sessions[sessions >= VIXF_START]
    rebuilt = crisis.vix_futures_constant_maturity(vix_settles(), sessions).dropna()
    raw = yf.download("VXX", start="2018-01-01", auto_adjust=True, progress=False)
    close = raw["Close"].squeeze().dropna()
    close.index = pd.DatetimeIndex(close.index).tz_localize(None)
    vxx = close.pct_change().dropna()
    common = rebuilt.index.intersection(vxx.index)
    a, b = rebuilt.reindex(common), vxx.reindex(common)
    weekly = pd.concat({"rebuilt": a, "vxx": b}, axis=1).add(1).resample("W-FRI").prod().sub(1)
    years = len(common) / 252
    print(f"common sessions {len(common):,}  {common.min():%Y-%m-%d} -> {common.max():%Y-%m-%d}")
    print(f"daily correlation   {np.corrcoef(a, b)[0, 1]:.4f}")
    print(f"weekly correlation  {weekly.corr().iloc[0, 1]:.4f}")
    vol = a.std() * np.sqrt(252), b.std() * np.sqrt(252)
    print(f"annualised vol      rebuilt {vol[0]:.1%}   VXX {vol[1]:.1%}")
    growth = (1 + a).prod() ** (1 / years) - 1, (1 + b).prod() ** (1 / years) - 1
    print(f"annualised return   rebuilt {growth[0]:+.1%}   VXX {growth[1]:+.1%}  "
          "(VXX carries a 0.89% annual fee)")
    feb = slice("2018-02-05", "2018-02-05")
    print(f"2018-02-05          rebuilt {a.loc[feb].iloc[0]:+.1%}   VXX {b.loc[feb].iloc[0]:+.1%}")


if __name__ == "__main__":
    main()

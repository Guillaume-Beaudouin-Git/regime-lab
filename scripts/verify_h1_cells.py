"""Reconstruct the H1 sixteen-cell diagnosis, which has NO committed code.

RESULTS_H1.md publishes two numbers about sixteen theme-by-asset-class cells:
the frozen map agrees with the empirical signs on 7 of them, and one reaches a
Newey-West |t| above 2. scripts/run.py computes neither -- it stops at the book
and the two placebos. So this is a RECONSTRUCTION under a construction stated
here, not a reproduction of the original code, which does not exist.

Construction, fixed before looking at the output:
  * each asset class becomes one equal-weight composite of its instruments,
    each instrument inverse-volatility scaled exactly as book.py does it;
  * each cell regresses the composite's next-day return on the standardised
    theme, lagged one session, with a Newey-West covariance;
  * HAC lag 6, the house rule; lag 21 reported as a sensitivity.

Read-only with respect to the repository. Writes nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from macro_momentum.book import inverse_vol_weight  # noqa: E402
from macro_momentum.exposures import CONTESTED, SIGN_MAP  # noqa: E402
from macro_momentum.universe import ASSETS  # noqa: E402
from scripts.run import load  # noqa: E402
from macro_momentum.signals import all_themes, standardise  # noqa: E402

warnings.filterwarnings("ignore")
RULE = "=" * 78
CLASSES = list(next(iter(SIGN_MAP.values())))


def composite(returns: pd.DataFrame) -> dict[str, pd.Series]:
    """One risk-equalised return series per asset class."""
    scale = inverse_vol_weight(returns)
    out = {}
    for cls in CLASSES:
        tick = [t for t, k in ASSETS.items() if k == cls and t in returns.columns]
        if not tick:
            continue
        w = scale[tick].shift(1)
        w = w.div(w.abs().sum(axis=1), axis=0)
        out[cls] = (w * returns[tick]).sum(axis=1, min_count=1).rename(cls)
    return out


def cell(theme: pd.Series, ret: pd.Series, lags: int) -> tuple[float, float, int]:
    d = pd.concat([theme.shift(1).rename("x"), ret.rename("y")], axis=1).dropna()
    if len(d) < 250:
        return np.nan, np.nan, len(d)
    m = sm.OLS(d["y"].to_numpy(), sm.add_constant(d["x"].to_numpy())).fit(
        cov_type="HAC", cov_kwds={"maxlags": lags}
    )
    return float(m.params[1]), float(m.tvalues[1]), len(d)


def main() -> None:
    prices, macro = load()
    returns = prices.pct_change()
    themes = standardise(all_themes(macro)).dropna()
    returns = returns.loc[returns.index.intersection(themes.index)]
    comp = composite(returns)

    print(RULE)
    print("H1 SIXTEEN-CELL DIAGNOSIS — reconstructed (no committed code exists)")
    print(RULE)
    print(f"\nsample {returns.index.min():%Y-%m} to {returns.index.max():%Y-%m}, "
          f"{len(returns):,} days, {returns.shape[1]} assets")
    print(f"asset-class composites: {list(comp)}\n")

    for lags in (6, 21):
        print(RULE)
        print(f"\nNewey-West maxlags = {lags}\n")
        print(f"  {'theme':<11}{'class':<13}{'slope':>12}{'t':>8}{'emp':>6}"
              f"{'map':>6}{'agree':>7}   flag")
        agree = above = total = 0
        rows = []
        for theme in SIGN_MAP:
            for cls in CLASSES:
                if cls not in comp:
                    continue
                b, t, n = cell(themes[theme], comp[cls], lags)
                if not np.isfinite(t):
                    continue
                total += 1
                emp = int(np.sign(b))
                mp = SIGN_MAP[theme][cls]
                ok = emp == mp
                agree += ok
                above += abs(t) > 2.0
                flag = "contested" if (theme, cls) in CONTESTED else ""
                rows.append((theme, cls, b, t, emp, mp, ok, flag))
                print(f"  {theme:<11}{cls:<13}{b:>+12.2e}{t:>+8.2f}{emp:>+6d}"
                      f"{mp:>+6d}{('yes' if ok else 'NO'):>7}   {flag}")
        print(f"\n  agreement with the frozen map : {agree} of {total}"
              f"   (chance: {total / 2:.1f})")
        print(f"  cells with |t| > 2            : {above} of {total}"
              f"   (chance: {0.05 * total:.1f})")
        if rows:
            big = max(rows, key=lambda r: abs(r[3]))
            print(f"  largest |t|                   : {abs(big[3]):.2f} "
                  f"({big[0]} -> {big[1]})")
        print()

    print(RULE)


if __name__ == "__main__":
    main()

"""T2: the decisive test, in the form the first charter got wrong.

Version 1 compared the overlay with the base book de-levered by a constant to
the overlay's volatility. That test is empty: the Sharpe ratio is invariant to a
constant scaling, so the benchmark had, to the decimal, the Sharpe of the
strategy it was supposed to challenge.

The real competitor is a book scaled by a volatility forecast that moves. It
takes risk off before turbulence in the same way a regime overlay claims to, and
it needs no regimes to do it. So:

* the benchmark is the base book sized by an ex-ante volatility target,
  ``target / sigma(t-1)``, with sigma an exponentially weighted estimate;
* the target is set at every refit to the volatility the overlay realised **on
  that refit's training window**, never on the sample being evaluated;
* the two legs are then checked for comparable realised volatility out of
  sample, and a gap wider than ten percent invalidates the comparison rather
  than being reported through;
* the metrics are ones that a constant scaling would change: certainty
  equivalent under constant relative risk aversion, return over worst drawdown,
  and a Sharpe difference tested by paired block bootstrap.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from regime_lab.analysis.power import minimum_detectable_sharpe_difference
from regime_lab.config import CACHE
from regime_lab.data import build_panel, store
from regime_lab.models.mapping import apply_overlay, state_to_position
from regime_lab.strategies.base import sixty_forty

warnings.filterwarnings("ignore")
RULE = "=" * 78
GAMMA = 5.0
VOL_TOLERANCE = 0.10


def annual_sharpe(x: pd.Series) -> float:
    return float(x.mean() / x.std(ddof=1) * np.sqrt(252)) if x.std(ddof=1) > 0 else np.nan


def certainty_equivalent(x: pd.Series, *, gamma: float = GAMMA) -> float:
    """Mean-variance certainty equivalent, annualised.

    Unlike the Sharpe ratio this responds to leverage, which is the whole point:
    a benchmark that differs from the strategy only by a constant scaling must
    be able to score differently.
    """
    mean = float(x.mean()) * 252
    variance = float(x.var(ddof=1)) * 252
    return mean - 0.5 * gamma * variance


def return_over_drawdown(x: pd.Series) -> float:
    curve = (1 + x).cumprod()
    worst = float((curve / curve.cummax() - 1).min())
    annual = float(curve.iloc[-1] ** (252 / len(x)) - 1)
    return annual / abs(worst) if worst < 0 else np.nan


def stepwise_target(diagnostics: pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series:
    """Piecewise-constant volatility target, changing only at refits."""
    series = diagnostics.set_index("refit")["train_overlay_vol"].sort_index()
    return series.reindex(index, method="ffill")


def managed_benchmark(
    base: pd.Series, target: pd.Series, *, halflife: int = 20, cap: float = 3.0
) -> pd.Series:
    """Base book sized to an ex-ante volatility target that moves at refits."""
    sigma = base.ewm(halflife=halflife, min_periods=halflife).std() * np.sqrt(252)
    leverage = (target / sigma.shift(1)).clip(upper=cap)
    return (leverage * base).rename("vol_targeted")


def main() -> None:
    states = pd.read_parquet(CACHE / "states.parquet")
    diagnostics = pd.read_parquet(CACHE / "refit_diagnostics.parquet")

    prices = store.read("prices", "cross_asset")
    dates = pd.date_range("1990-01-01", prices["period"].max(), freq="B")
    book = sixty_forty(build_panel(prices, dates)).dropna()

    oos = states.dropna(how="all").index
    base = book.reindex(oos).dropna()
    oos = base.index

    print(RULE)
    print(
        f"T2  decisive test, {len(oos):,} out-of-sample days "
        f"({oos.min():%Y-%m} to {oos.max():%Y-%m})"
    )
    print(RULE)
    print("\n   benchmark: base book at an ex-ante volatility target set on training data")
    print("   metrics chosen to respond to leverage, unlike the Sharpe ratio\n")
    print(
        f"   {'family':<18} {'SR ovl':>7} {'SR bmk':>7} {'dSR':>7} "
        f"{'CEQ ovl':>8} {'CEQ bmk':>8} {'vol gap':>8}"
    )

    verdicts: list[tuple[str, float, float, bool]] = []

    for family in states.columns:
        position = state_to_position(states[family]).reindex(oos).fillna(0.0)
        overlay = apply_overlay(base, position).dropna()

        diag = diagnostics[diagnostics["family"] == family]
        if diag.empty:
            continue
        target = stepwise_target(diag, oos)
        benchmark = managed_benchmark(base, target).dropna()

        common = overlay.index.intersection(benchmark.index)
        a, b = overlay.loc[common], benchmark.loc[common]

        vol_a = float(a.std(ddof=1) * np.sqrt(252))
        vol_b = float(b.std(ddof=1) * np.sqrt(252))
        gap = abs(vol_a - vol_b) / vol_a if vol_a > 0 else np.nan
        valid = gap <= VOL_TOLERANCE

        print(
            f"   {family:<18} {annual_sharpe(a):>7.2f} {annual_sharpe(b):>7.2f} "
            f"{annual_sharpe(a) - annual_sharpe(b):>7.2f} "
            f"{certainty_equivalent(a):>8.2%} {certainty_equivalent(b):>8.2%} "
            f"{gap:>7.1%}{'' if valid else ' X'}"
        )
        verdicts.append((family, annual_sharpe(a) - annual_sharpe(b), gap, valid))

    print("\n   X marks a volatility gap above 10%: the comparison is not like-for-like")

    print("\n" + RULE)
    print("\nIS ANY DIFFERENCE LARGER THAN THE NOISE?\n")

    for family, delta, gap, valid in verdicts:
        position = state_to_position(states[family]).reindex(oos).fillna(0.0)
        overlay = apply_overlay(base, position).dropna()
        diag = diagnostics[diagnostics["family"] == family]
        benchmark = managed_benchmark(base, stepwise_target(diag, oos)).dropna()
        common = overlay.index.intersection(benchmark.index)

        res = minimum_detectable_sharpe_difference(
            overlay.loc[common].to_numpy(),
            benchmark.loc[common].to_numpy(),
            mean_block=63,
            draws=1_500,
        )
        detected = abs(res.observed) > res.mde
        flag = "DETECTED" if detected else "inside the noise"
        note = "" if valid else "  (volatility check already failed)"
        print(f"   {family:<18} observed {res.observed:+.3f}  MDE {res.mde:.3f}  -> {flag}{note}")

    print(
        "\n   A difference inside the minimum detectable effect is UNDERPOWERED,\n"
        "   not evidence of no effect, and it is never reported as a pass."
    )
    print("\n" + RULE)


if __name__ == "__main__":
    main()

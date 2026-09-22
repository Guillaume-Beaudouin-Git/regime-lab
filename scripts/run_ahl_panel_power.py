"""AHL day 1, step 3 — the MDE of the panel statistic, under the null.

`pilotage/plans_de_recherche/ahl/PLAN.md` §c refuses to guess this number. It brackets it
analytically between 0.8 points (clustering by date) and 4.5-6.0 points (clustering by
63-session episode), declares an expectation of 2 to 4, and fixes the *procedure*:

    stationary bootstrap on the date dimension, the entire cross-section preserved in
    each draw, three block lengths reported, **under the null and before any
    interaction coefficient is read**.

THE CRITERION, WRITTEN BEFORE THE NUMBER IS PRODUCED, from the arbitration:

    If the measured MDE comes out above ~4.5 points of hit rate, even the panel
    question is undecidable on this sample, level A of the escalation tree is not
    posable, and that is what gets written. Half a day, zero trials from the
    register, a result at no cost.

This script therefore prints the MDE and **not** the observed statistic. Reading Δ is
level A and level A is a trial; this is the instrument check that comes first.

The statistic, exactly as §c specifies it:

    Δ = [hit(fast | consolidation) − hit(slow | consolidation)]
      − [hit(fast | trend)         − hit(slow | trend)]

weighted by risk-adjusted exposure, over 46 instruments × 6,039 sessions.

Two nulls rather than one, because they answer different questions. The bootstrap gives
the sampling error of Δ over dates. The rotation gives the distribution of Δ when the
regime label carries no information but keeps its exact run structure — the P1 placebo
the programme uses everywhere. If the two disagree, the wider one decides.

No return of any strategy is computed. Writes nothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from regime_lab.analysis.bootstrap import stationary_indices
from regime_lab.config import CACHE

RULE = "=" * 78

#: Settled in step 1: the strict reading, 252 sessions strictly before t, which is the
#: one the frozen §3 names and the one a signal formed at t-1 can actually use.
SAMPLE_START = pd.Timestamp("2003-07-17")

#: The criterion, from the arbitration, quoted so it cannot drift.
CRITERION_POINTS = 4.5

SLOW = (252, 21)   # the programme's own 12-minus-1
FAST = (63, 5)     # disagrees with it on 35.9% of instrument-sessions
VOL_WINDOW = 63
ER_WINDOW = 63
DRAWS = 2_000
BLOCKS = (21, 63, 126)
Z_MDE = 1.959964 + 0.841621  # two-sided 5%, 80% power — analysis/power.py's constants


def trend_signal(prices: pd.DataFrame, lookback: int, skip: int) -> pd.DataFrame:
    return np.sign(prices.shift(skip) / prices.shift(lookback) - 1.0)


def efficiency_ratio(prices: pd.DataFrame, window: int = ER_WINDOW) -> pd.Series:
    """Kaufman's ratio per instrument, then the cross-sectional median of the book.

    Directional travel over gross travel: high when the path goes somewhere, low when
    it thrashes. Measured at +0.096 Pearson against realised volatility, which is what
    makes it a different latent variable rather than volatility renamed.
    """
    log_prices = np.log(prices.where(prices > 0))
    steps = log_prices.diff()
    net = (log_prices - log_prices.shift(window)).abs()
    gross = steps.abs().rolling(window).sum()
    return (net / gross.replace(0.0, np.nan)).median(axis=1)


def panel_pieces(
    prices: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    """Per-instrument hit indicators for each speed, the exposure weights, and the state.

    Everything is lagged one session before it meets a return, so a date row carries a
    decision that was available at the previous close.
    """
    returns = prices.pct_change()
    sigma = returns.rolling(VOL_WINDOW).std() * np.sqrt(252)
    weight = (0.10 / sigma).clip(upper=3.0).shift(1)

    slow = trend_signal(prices, *SLOW).shift(1)
    fast = trend_signal(prices, *FAST).shift(1)
    realised = np.sign(returns)

    hit_slow = (slow == realised).where(slow.notna() & realised.notna() & slow.ne(0.0))
    hit_fast = (fast == realised).where(fast.notna() & realised.notna() & fast.ne(0.0))

    ratio = efficiency_ratio(prices)
    # Two states by the instrument's own expanding median — the same shape as the
    # one-line control that beat all six devices, so the state cannot be accused of
    # carrying a parameter the control does not have.
    median = ratio.expanding(min_periods=252).median()
    trending = (ratio > median).astype(float).where(ratio.notna() & median.notna()).shift(1)

    return hit_fast.astype(float), hit_slow.astype(float), weight, trending


def delta(hit_fast, hit_slow, weight, trending, dates) -> float:
    """The weighted difference-in-differences of hit rate, on the given dates."""
    hf, hs = hit_fast.loc[dates], hit_slow.loc[dates]
    w, state = weight.loc[dates], trending.loc[dates]

    def rate(hits: pd.DataFrame, mask: pd.Series) -> float:
        wm = w.where(hits.notna()).mul(mask.astype(float), axis=0)
        wm = wm.where(wm > 0)
        total = np.nansum(wm.to_numpy())
        if total <= 0:
            return np.nan
        return float(np.nansum((hits * wm).to_numpy()) / total)

    consolidating, trend = state.eq(0.0), state.eq(1.0)
    return ((rate(hf, consolidating) - rate(hs, consolidating))
            - (rate(hf, trend) - rate(hs, trend)))


def main() -> None:
    prices = pd.read_parquet(CACHE / "trend_universe_m1.parquet").loc[SAMPLE_START:]
    hit_fast, hit_slow, weight, trending = panel_pieces(prices)

    usable = trending.notna() & weight.notna().any(axis=1) & hit_slow.notna().any(axis=1)
    dates = prices.index[usable.reindex(prices.index, fill_value=False)]

    print(RULE)
    print("AHL DAY 1, STEP 3 — THE MDE OF THE PANEL STATISTIC, UNDER THE NULL")
    print(RULE)
    print(f"\n   sample        {dates.min():%Y-%m-%d} to {dates.max():%Y-%m-%d}, "
          f"{len(dates):,} usable sessions of {len(prices):,}")
    print(f"   panel         {prices.shape[1]} instruments")
    print(f"   speeds        slow {SLOW[0]}-{SLOW[1]}, fast {FAST[0]}-{FAST[1]}")
    print(f"   state         Kaufman ER{ER_WINDOW}, book median, expanding-median split")
    print(f"   criterion     MDE above {CRITERION_POINTS} points => the panel question")
    print("                 is undecidable and level A is not posable")
    print("\n   The observed statistic is deliberately NOT printed. Reading it is level A,")
    print("   and level A is a trial. This is the instrument check that comes first.")

    rng = np.random.default_rng(20260923)
    positions = np.arange(len(dates))

    print(f"\n{RULE}\n1.  SAMPLING ERROR OF THE STATISTIC — stationary bootstrap on dates\n")
    print(f"   {'mean block':>11} {'se(delta)':>12} {'MDE, points':>14}   reading")
    sampling = {}
    for block in BLOCKS:
        draws = np.empty(DRAWS)
        for i in range(DRAWS):
            take = stationary_indices(len(dates), block, rng)
            draws[i] = delta(hit_fast, hit_slow, weight, trending, dates[positions[take]])
        se = float(np.nanstd(draws, ddof=1))
        mde = Z_MDE * se * 100.0
        sampling[block] = mde
        verdict = "decidable" if mde <= CRITERION_POINTS else "ABOVE THE CRITERION"
        print(f"   {block:>11} {se:>12.5f} {mde:>14.2f}   {verdict}")

    print(f"\n{RULE}\n2.  THE NULL PROPER — the regime label rotated against the dates\n")
    print("   Preserves the run structure of the state exactly and destroys only its")
    print("   alignment with the returns. This is the programme's P1 placebo.\n")
    offsets = rng.integers(252, len(dates) - 252, size=400)
    null = np.empty(len(offsets))
    state_values = trending.loc[dates].to_numpy()
    for i, offset in enumerate(offsets):
        rotated = pd.Series(np.roll(state_values, offset), index=dates)
        null[i] = delta(hit_fast, hit_slow, weight, rotated, dates)
    sd_null = float(np.nanstd(null, ddof=1))
    mde_null = Z_MDE * sd_null * 100.0
    print(f"   {'draws':>11} {'sd under null':>15} {'MDE, points':>14}   reading")
    verdict = "decidable" if mde_null <= CRITERION_POINTS else "ABOVE THE CRITERION"
    print(f"   {len(offsets):>11} {sd_null:>15.5f} {mde_null:>14.2f}   {verdict}")
    print(f"\n   null mean {np.nanmean(null) * 100:+.3f} points — it should sit on zero;")
    print("   a null centred away from zero would mean the statistic is biased by its")
    print("   own construction rather than by the state.")

    print(f"\n{RULE}\n3.  THE DECISION\n")
    readings = [*sampling.values(), mde_null]
    if any(not np.isfinite(v) for v in readings):
        print("   ONE OR MORE READINGS IS NaN. No verdict is issued: a NaN is a broken")
        print("   instrument, not an undecidable question, and letting it read as one")
        print("   would be the failure this programme keeps finding in its own work.")
        print(f"\n{RULE}")
        return
    deciding = max(readings)
    print(f"   the widest of the four readings decides: {deciding:.2f} points")
    print(f"   the criterion, written before the number:  {CRITERION_POINTS} points")
    print("   PLAN.md §c declared an expectation of 2 to 4 points.\n")
    if deciding <= CRITERION_POINTS:
        print("   => THE PANEL QUESTION IS DECIDABLE. Level A is posable, and the")
        print("      smallest difference-in-differences this sample can resolve is the")
        print("      number above. Level A becomes a trial and goes to the register.")
    else:
        print("   => THE PANEL QUESTION IS NOT DECIDABLE on this sample. Level A is not")
        print("      posable. This costs half a day and no trial, and it is the result.")
    print(f"\n{RULE}")


if __name__ == "__main__":
    main()

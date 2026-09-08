"""The decomposition the charter exists to produce.

The question is not whether conditioning on regimes beats a volatility target —
T2 answered that, and the honest answer is that this sample cannot tell. The
question with unknown magnitudes is *where the apparent value of a regime
overlay actually comes from*, and this script attributes it.

The ladder runs from the raw book to the traded overlay, one effect at a time.
Each rung changes exactly one thing, so the gap between two rungs is that
effect's contribution:

1. the base book, fully invested
2. **exposure reduction** — the same book held at the overlay's average
   exposure, constant. Pure de-risking, no timing of any kind
3. **volatility timing** — the book at a moving ex-ante volatility target. Still
   no regimes
4. **regime information with hindsight** — the overlay driven by states assigned
   with the whole block visible. Not tradable; it is the ceiling
5. **detection latency** — the same overlay driven by filtered states. The drop
   from rung 4 is the price of recognising a regime as it happens
6. **turnover cost** — rung 5 net of a declared two basis points

The certainty equivalent is the reporting metric throughout, because rung 2 is
invisible to the Sharpe ratio by construction: a constant scaling cannot change
it. That fact is printed rather than hidden, since it is the reason the first
version of this study's decisive test was empty.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from regime_lab.config import CACHE
from regime_lab.models.mapping import apply_overlay, state_to_position
from regime_lab.strategies.book import base_book
from regime_lab.strategies.costs import charge

warnings.filterwarnings("ignore")
RULE = "=" * 78
GAMMA = 5.0
COST = 2.0


def ceq(x: pd.Series, *, gamma: float = GAMMA) -> float:
    return float(x.mean()) * 252 - 0.5 * gamma * float(x.var(ddof=1)) * 252


def sharpe(x: pd.Series) -> float:
    return float(x.mean() / x.std(ddof=1) * np.sqrt(252)) if x.std(ddof=1) > 0 else np.nan


def vol_targeted(base: pd.Series, target: pd.Series, *, halflife: int = 20) -> pd.Series:
    sigma = base.ewm(halflife=halflife, min_periods=halflife).std() * np.sqrt(252)
    return ((target / sigma.shift(1)).clip(upper=3.0) * base).dropna()


def main() -> None:
    online = pd.read_parquet(CACHE / "states.parquet")
    offline = pd.read_parquet(CACHE / "states_offline.parquet")
    diagnostics = pd.read_parquet(CACHE / "refit_diagnostics.parquet")

    book = base_book()
    oos = online.dropna(how="all").index
    base = book.reindex(oos).dropna()
    oos = base.index

    print(RULE)
    print("DECOMPOSITION  where the value of a regime overlay comes from")
    print(
        f"{len(oos):,} out-of-sample days, {oos.min():%Y-%m} to {oos.max():%Y-%m}, excess of cash"
    )
    print(RULE)
    print(
        f"\n   certainty equivalent, annualised, constant relative risk aversion "
        f"gamma = {GAMMA:.0f}"
    )
    print(f"   cost charged at {COST:.0f} basis points round trip, declared in advance\n")

    header = (
        f"   {'family':<18} {'base':>7} {'exposure':>9} {'vol tim.':>9} "
        f"{'hindsight':>10} {'latency':>9} {'costs':>8}"
    )
    print(header)
    print("   " + "-" * (len(header) - 3))

    rows: list[dict[str, float]] = []
    for family in online.columns:
        pos_on = state_to_position(online[family]).reindex(oos).fillna(0.0)
        pos_off = state_to_position(offline[family]).reindex(oos).fillna(0.0)

        diag = diagnostics[diagnostics["family"] == family]
        target = (
            diag.set_index("refit")["train_overlay_vol"].sort_index().reindex(oos, method="ffill")
        )

        rung1 = base
        rung2 = base * float(pos_on.mean())
        rung3 = vol_targeted(base, target)
        rung4 = apply_overlay(base, pos_off).dropna()
        rung5 = apply_overlay(base, pos_on).dropna()
        rung6 = charge(rung5, pos_on, bps=COST)

        common = rung3.index.intersection(rung5.index).intersection(rung4.index)
        values = [ceq(r.loc[common]) for r in (rung1, rung2, rung3, rung4, rung5, rung6)]

        print(
            f"   {family:<18} {values[0]:>6.2%} {values[1] - values[0]:>+9.2%} "
            f"{values[2] - values[1]:>+9.2%} {values[3] - values[2]:>+10.2%} "
            f"{values[4] - values[3]:>+9.2%} {values[5] - values[4]:>+8.2%}"
        )
        rows.append(
            {
                "family": family,
                "exposure": values[1] - values[0],
                "vol_timing": values[2] - values[1],
                "hindsight": values[3] - values[2],
                "latency": values[4] - values[3],
                "costs": values[5] - values[4],
                "total": values[5] - values[0],
                "sharpe_hindsight": sharpe(rung4.loc[common]),
                "sharpe_filtered": sharpe(rung5.loc[common]),
            }
        )

    frame = pd.DataFrame(rows).set_index("family")

    print("\n" + RULE)
    print("\nWHAT THE LADDER SAYS\n")

    print("   the two rungs that need no regime model at all:")
    print(f"      cutting exposure      {frame['exposure'].mean():>+7.2%} on average")
    print(f"      timing volatility     {frame['vol_timing'].mean():>+7.2%} on average")

    print("\n   the two rungs that are the regime model's own contribution:")
    print(f"      regime information    {frame['hindsight'].mean():>+7.2%} with hindsight")
    print(f"      minus detection lag   {frame['latency'].mean():>+7.2%}")
    kept = frame["hindsight"] + frame["latency"]
    print(f"      net of latency        {kept.mean():>+7.2%}")

    print(f"\n   and then trading costs  {frame['costs'].mean():>+7.2%}")

    print("\n   per family, what hindsight is worth in Sharpe terms:")
    for family, row in frame.iterrows():
        gap = row["sharpe_hindsight"] - row["sharpe_filtered"]
        print(
            f"      {family:<18} hindsight {row['sharpe_hindsight']:>5.2f}  "
            f"filtered {row['sharpe_filtered']:>5.2f}  latency costs {gap:>5.2f}"
        )

    print(
        "\n   Rung 2 is invisible to the Sharpe ratio: a constant scaling cannot\n"
        "   change it. That is not a quirk of the reporting, it is the reason the\n"
        "   first version of this study's decisive test compared a strategy with\n"
        "   itself and was arithmetically empty."
    )

    frame.to_parquet(CACHE / "decomposition.parquet")
    print("\n" + RULE)


if __name__ == "__main__":
    main()

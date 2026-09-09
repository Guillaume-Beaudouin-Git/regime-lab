"""Layer 3: apply the state the way the evidence says it can be applied.

The measured signal separates forward variance and not forward mean. An on/off
rule asks it for direction, which it does not have; an inverse-volatility sizing
rule asks it for magnitude, which it does. Both are reported side by side, and
the frozen on/off rule is never dropped from a table just because a better rule
was added afterwards.

The comparison also carries its own power calculation, because the mean test and
the variance test are not equally answerable on this sample.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from regime_lab.config import CACHE
from regime_lab.evaluation.predictive import (
    mean_difference_test,
    volatility_difference_test,
)
from regime_lab.models.mapping import apply_overlay, state_to_position, state_to_size, summary
from regime_lab.strategies.book import base_book
from regime_lab.strategies.costs import charge

warnings.filterwarnings("ignore")
RULE = "=" * 78
COST = 2.0


def annual_sharpe(x: pd.Series) -> float:
    return float(x.mean() / x.std(ddof=1) * np.sqrt(252)) if x.std(ddof=1) > 0 else np.nan


def ceq(x: pd.Series, *, gamma: float = 5.0) -> float:
    return float(x.mean()) * 252 - 0.5 * gamma * float(x.var(ddof=1)) * 252


def main() -> None:
    states = pd.read_parquet(CACHE / "states.parquet")
    diagnostics = pd.read_parquet(CACHE / "refit_diagnostics.parquet")

    book = base_book()
    oos = states.dropna(how="all").index
    base = book.reindex(oos).dropna()
    oos = base.index

    train_vol = float(book.loc[: oos.min()].std() * np.sqrt(252))

    print(RULE)
    print(f"LAYER 3  timing rule against sizing rule, {len(oos):,} out-of-sample days")
    print(RULE)
    print(f"\n   volatility target {train_vol:.1%}, taken from data before {oos.min():%Y-%m}")
    print(f"   costs charged at {COST:.0f} basis points round trip\n")

    header = (
        f"   {'family':<18} {'rule':<10} {'Sharpe':>7} {'net':>7} "
        f"{'vol':>7} {'maxDD':>8} {'CEQ':>8} {'turnover':>9}"
    )
    print(header)
    print("   " + "-" * (len(header) - 3))

    stats = summary(base)
    print(
        f"   {'base 60/40':<18} {'—':<10} {stats['sharpe']:>7.2f} {stats['sharpe']:>7.2f} "
        f"{stats['vol']:>6.1%} {stats['max_drawdown']:>8.1%} {ceq(base):>7.2%} {'—':>9}"
    )

    for family in states.columns:
        diag = diagnostics[diagnostics["family"] == family].set_index("refit").sort_index()
        vol_columns = [c for c in diag.columns if c.startswith("state_vol_")]

        rules: dict[str, pd.Series] = {
            "on/off": state_to_position(states[family]).reindex(oos).fillna(0.0)
        }
        if vol_columns:
            rules["sized"] = (
                state_to_size(states[family], diag[vol_columns], target=train_vol)
                .reindex(oos)
                .fillna(0.0)
            )

        for label, position in rules.items():
            gross = apply_overlay(base, position).dropna()
            net = charge(gross, position, bps=COST)
            s = summary(gross)
            print(
                f"   {family:<18} {label:<10} {s['sharpe']:>7.2f} "
                f"{annual_sharpe(net):>7.2f} {s['vol']:>6.1%} "
                f"{s['max_drawdown']:>8.1%} {ceq(net):>7.2%} "
                f"{position.diff().abs().mean():>9.4f}"
            )

    print("\n" + RULE)
    print("\nPOWER  the mean question and the variance question are not equally answerable\n")
    print(
        f"   {'family':<18} {'mean spread':>12} {'MDE':>8} {'':>4} "
        f"{'vol spread':>11} {'MDE':>8} {'':>4}"
    )

    for family in states.columns:
        mean_test = mean_difference_test(states[family], base, horizon=21, draws=800)
        vol_test = volatility_difference_test(states[family], base, horizon=21, draws=800)
        mean_ok = abs(mean_test["difference"]) > mean_test["mde"]
        vol_ok = abs(vol_test["difference"]) > vol_test["mde"]
        print(
            f"   {family:<18} {mean_test['difference']:>+11.2%} {mean_test['mde']:>7.2%} "
            f"{'YES' if mean_ok else 'no':>4} "
            f"{vol_test['difference']:>+10.2%} {vol_test['mde']:>7.2%} "
            f"{'YES' if vol_ok else 'no':>4}"
        )

    print(
        "\n   A volatility spread is estimated far more precisely than a mean spread\n"
        "   on the same data. That is the statistical reason a regime signal can be\n"
        "   usable for sizing while being useless for timing, and reporting only the\n"
        "   mean test would have hidden it."
    )
    print("\n" + RULE)


if __name__ == "__main__":
    main()

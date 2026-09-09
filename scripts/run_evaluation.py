"""Evaluate the classifier as a classifier, before any portfolio appears.

The study drifted: a regime model was built, immediately converted into a
position by a one-bit rule, and then judged on the Sharpe ratio of the resulting
book. That answers a portfolio question through the noisiest available lens, and
it is not the question the project set out to ask.

This script asks the stated question in two layers. Layer 1 is whether the label
sequence is reliable — persistent beyond chance, agreed on by different methods,
and recognisable against an external reference. Layer 2 is whether it predicts
anything, and crucially *which moment* it predicts: a state that separates
forward volatility but not forward mean is a sizing signal, not a timing signal.

No position is taken anywhere in this file.
"""

from __future__ import annotations

import warnings

import pandas as pd

from regime_lab.config import CACHE
from regime_lab.data import store
from regime_lab.evaluation import predictive, reliability
from regime_lab.evaluation.predictive import volatility_quantile_placebo
from regime_lab.strategies.book import base_book, panels

warnings.filterwarnings("ignore")
RULE = "=" * 78
HORIZONS = (5, 21, 63)


def main() -> None:
    online = pd.read_parquet(CACHE / "states.parquet")
    offline = pd.read_parquet(CACHE / "states_offline.parquet")

    price_panel, macro_panel = panels()
    book = base_book()
    oos = online.dropna(how="all").index
    returns = book.reindex(oos).dropna()
    oos = returns.index

    equity = price_panel["eq_us_large"].reindex(oos)
    realised = (equity.pct_change().rolling(21).std() * (252**0.5)).reindex(oos)
    drawdown_ref = reliability.drawdown_reference(equity).reindex(oos)
    # Scored against the month each observation describes. Routing this through
    # the lagged macro path handed the classifier up to forty-five days of free
    # hindsight at every boundary and overstated its accuracy.
    nber = store.read("references", "ref_nber").set_index("period")["value"]
    nber_ref = nber.reindex(oos.to_period("M").to_timestamp()).to_numpy()
    nber_ref = pd.Series(nber_ref, index=oos, name="ref_nber")

    # A one-line causal rule sits in every table beside the fitted models. A
    # claim that a model separates variance means nothing until it is read next
    # to what a volatility median does for free.
    online = online.copy()
    online["·  vol placebo"] = volatility_quantile_placebo(returns).reindex(oos)
    offline = offline.copy()
    offline["·  vol placebo"] = online["·  vol placebo"]

    print(RULE)
    print(f"CLASSIFIER EVALUATION  {len(oos):,} out-of-sample days, no portfolio")
    print(RULE)

    print("\nLAYER 1  is the labelling reliable?\n")
    header = (
        f"   {'family':<18} {'mean run':>9} {'vs chance':>10} "
        f"{'hindsight':>10} {'kappa on/off':>13}"
    )
    print(header)
    for family in online.columns:
        persistence = reliability.persistence_vs_chance(online[family])
        gap = reliability.hindsight_gap(online[family], offline[family])
        print(
            f"   {family:<18} {persistence['observed']:>8.0f}d {persistence['ratio']:>10.1f}x "
            f"{gap['changed_share']:>9.1%} {gap['kappa']:>13.2f}"
        )
    print("\n   'hindsight' is the share of sessions where seeing the rest of the block")
    print("   would have changed the label — what a real-time user cannot have")

    print("\n   agreement between methods, chance-corrected:\n")
    families = list(online.columns)
    print(f"   {'':<18}" + "".join(f"{f.split()[0]:>9}" for f in families))
    for a in families:
        row = f"   {a:<18}"
        for b in families:
            if a == b:
                row += f"{'-':>9}"
            else:
                kappa = reliability.agreement(online[a], online[b])["kappa"]
                row += f"{kappa:>9.2f}"
        print(row)

    print("\n   against external references, never our own labels:\n")
    print(f"   {'family':<18} {'reference':<12} {'bal. acc':>9} {'kappa':>7} {'majority':>9}")
    for family in online.columns:
        for label, ref in (("NBER", nber_ref), ("drawdown", drawdown_ref)):
            score = reliability.external_validation(online[family], ref)
            print(
                f"   {family:<18} {label:<12} {score['balanced_accuracy']:>9.1%} "
                f"{score['kappa']:>7.2f} {score['majority']:>9.1%}"
            )

    print("\n" + RULE)
    print("\nLAYER 2  does the state predict anything, and which moment?\n")

    for horizon in HORIZONS:
        print(f"   horizon {horizon} sessions")
        print(
            f"   {'family':<18} {'mean spread':>12} {'vol spread':>11} "
            f"{'t (HAC)':>8} {'MDE':>8} {'aligned':>8}"
        )
        for family in online.columns:
            split = predictive.variance_versus_mean(online[family], returns, horizon=horizon)
            test = predictive.mean_difference_test(
                online[family], returns, horizon=horizon, draws=800
            )
            print(
                f"   {family:<18} {split['mean_spread']:>+11.2%} {split['vol_spread']:>+10.2%} "
                f"{test['t_hac']:>8.2f} {test['mde']:>7.2%} "
                f"{'yes' if split['aligned'] else 'no':>8}"
            )
        print()

    print("   'aligned' means the state the model calls strong really does earn more,")
    print("   not merely shake less. A state that separates volatility alone is a")
    print("   sizing signal and not a timing signal, which is a different conclusion")
    print("   from 'regimes do not work'.\n")

    print(RULE)
    for target, label in (("return", "forward RETURNS"), ("volatility", "forward VOLATILITY")):
        print(f"\n   marginal information over a plain volatility quantile — {label}\n")
        print(
            f"   {'family':<18} {'R2 vol':>8} {'R2 both':>9} "
            f"{'incremental':>12} {'t state':>8}"
        )
        for family in online.columns:
            info = predictive.incremental_information(
                online[family], returns, realised, horizon=21, target=target
            )
            print(
                f"   {family:<18} {info['r2_vol']:>7.2%} {info['r2_both']:>8.2%} "
                f"{info['incremental']:>11.3%} {info['t_state']:>8.2f}"
            )

    print(
        "\n   The two tables are the finding. Over a volatility quantile the state adds\n"
        "   essentially nothing about direction and several points of R2 about magnitude.\n"
        "   Reporting only the first would have buried the study's strongest result."
    )
    print("\n" + RULE)


if __name__ == "__main__":
    main()

"""The two leads the study opened and never executed.

Neither touches a frozen result. The first asks whether the concentration
measure predicts or only describes; the second asks whether the practitioner's
continuous attenuator survives on a book built here.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

from regime_lab.config import CACHE
from regime_lab.extensions.concentration import absorption_ratio, effective_factors
from regime_lab.extensions.trend import attenuator, book

warnings.filterwarnings("ignore")
RULE = "=" * 78


def sharpe(x: pd.Series) -> float:
    x = x.dropna()
    return float(x.mean() / x.std(ddof=1) * np.sqrt(252)) if x.std(ddof=1) > 0 else np.nan


def main() -> None:
    px = pd.read_parquet(CACHE / "trend_universe.parquet")
    returns = px.pct_change()
    trend, weights = book(px)
    trend = trend.dropna()

    print(RULE)
    print(f"EXTENSIONS  {px.shape[1]} instruments, {len(px):,} jours, "
          f"{px.index.min():%Y-%m} a {px.index.max():%Y-%m}")
    print(RULE)
    print(f"\n   livre de tendance de reference : Sharpe {sharpe(trend):.2f}, "
          f"vol {trend.std(ddof=1)*np.sqrt(252):.1%}\n")

    print(RULE)
    print("\n1. LE COMPTEUR DE FACTEURS PREDIT-IL, OU DECRIT-IL ?\n")
    ef = effective_factors(returns).reindex(trend.index)
    ar = absorption_ratio(returns).reindex(trend.index)
    print(f"   facteurs effectifs : moyenne {ef.mean():.1f} sur {px.shape[1]} instruments, "
          f"min {ef.min():.1f}, max {ef.max():.1f}")

    # Forward performance of the trend book over the next h months.
    print(f"\n   {'decalage':<20} {'facteurs effectifs':>24} {'ratio d absorption':>24}")
    print(f"   {'':<20} {'pente':>10}{'t':>14} {'pente':>10}{'t':>14}")
    for lag, label in ((0, "contemporain"), (21, "1 mois avant"),
                       (63, "3 mois avant"), (126, "6 mois avant")):
        row = f"   {label:<20}"
        for measure in (ef, ar):
            fwd = trend.shift(-1).rolling(63).mean().shift(-(63 - 1)) * 252
            x = measure.shift(lag)
            d = pd.concat([x.rename("x"), fwd.rename("y")], axis=1).dropna()
            if len(d) < 500:
                row += f"{'—':>24}"
                continue
            z = (d["x"] - d["x"].mean()) / d["x"].std()
            m = sm.OLS(d["y"], sm.add_constant(z)).fit(cov_type="HAC", cov_kwds={"maxlags": 63})
            row += f"{m.params.iloc[1]:>10.2%}{m.tvalues.iloc[1]:>14.2f}"
        print(row)

    print("\n   pente = variation du rendement annualise du livre pour un ecart-type de la mesure")
    print("   'contemporain' reproduit la relation publiee ; les lignes suivantes la testent")

    print("\n" + RULE)
    print("\n2. L'ATTENUATEUR CONTINU DE VOLATILITE\n")
    mult = attenuator(returns)
    damped_w = weights * mult.reindex(weights.index)
    damped_w = damped_w.div(
        damped_w.abs().sum(axis=1).replace(0.0, np.nan), axis=0
    ).fillna(0.0)
    damped = (damped_w * returns).sum(axis=1).reindex(trend.index)

    # Same book scaled to the same volatility, so the comparison is not a
    # comparison of leverage.
    matched = damped * (trend.std(ddof=1) / damped.std(ddof=1))

    print(f"   {'':<28} {'Sharpe':>8} {'vol':>8} {'perte max':>11} {'rotation':>10}")
    for label, series, w in (("livre de tendance", trend, weights),
                             ("attenue", damped, damped_w),
                             ("attenue, meme volatilite", matched, damped_w)):
        s = series.dropna()
        curve = (1 + s).cumprod()
        dd = float((curve / curve.cummax() - 1).min())
        to = w.diff().abs().sum(axis=1).reindex(s.index).fillna(0).mean()
        print(f"   {label:<28} {sharpe(s):>8.2f} {s.std(ddof=1)*np.sqrt(252):>7.1%} "
              f"{dd:>11.1%} {to:>10.4f}")

    common = trend.index.intersection(damped.index)
    diff = (damped.loc[common] - trend.loc[common]).dropna()
    m = sm.OLS(diff.to_numpy(), np.ones(len(diff))).fit(
        cov_type="HAC", cov_kwds={"maxlags": 21}
    )
    print(f"\n   ecart de Sharpe : {sharpe(damped) - sharpe(trend):+.3f}")
    print(f"   difference quotidienne moyenne : t de Newey-West {m.tvalues[0]:+.2f}")
    print("   la source publiee annonce +0.05 a +0.09 sur son propre univers")
    print("\n" + RULE)


if __name__ == "__main__":
    main()

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
    print("   ^ gross renormalise a 1 : l attenuateur REALLOUE au lieu de dé-risquer.")
    print("     Ce n est pas l intervention publiee. Conserve comme temoin.")

    # L'intervention publiee : le gross est LIBRE de bouger, sinon multiplier
    # par un scalaire par instrument ne fait que redistribuer le meme risque.
    free_w = weights * mult.reindex(weights.index)
    free = (free_w * returns).sum(axis=1).reindex(trend.index)
    d2 = (free - trend).dropna()
    m2 = sm.OLS(d2.to_numpy(), np.ones(len(d2))).fit(
        cov_type="HAC", cov_kwds={"maxlags": 21}
    )
    print(f"\n   gross libre (intervention publiee) : Sharpe {sharpe(free):.2f}, "
          f"ecart {sharpe(free) - sharpe(trend):+.3f}, t de Newey-West {m2.tvalues[0]:+.2f}")
    print("   la source publiee annonce +0.05 a +0.09 sur son propre univers")

    print("\n" + RULE)
    print("\n3. LE REGIME DE VOLATILITE AU NIVEAU DU LIVRE\n")

    # Volatilite moyenne inter-instruments, en percentile EXPANDING de sa propre
    # histoire : aucune information future n entre dans le classement.
    mkt_sigma = (returns.rolling(63).std() * np.sqrt(252)).mean(axis=1)
    mkt_q = mkt_sigma.expanding(min_periods=756).rank(pct=True).shift(1)
    mkt_q = mkt_q.reindex(trend.index)

    print(f"   {'tercile de volatilite':<28} {'Sharpe du livre':>16} {'part du temps':>15}")
    edges = [(0.0, 1 / 3, "bas"), (1 / 3, 2 / 3, "median"), (2 / 3, 1.01, "haut")]
    for lo, hi, name in edges:
        mask = (mkt_q >= lo) & (mkt_q < hi)
        sub = trend[mask].dropna()
        print(f"   {name:<28} {sharpe(sub):>16.2f} {len(sub) / len(trend):>14.0%}")
    print(f"   {'non classe (amorcage)':<28} {'':>16} {mkt_q.isna().mean():>14.0%}")

    # L attenuateur applique au LIVRE, sur le percentile de volatilite du MARCHE.
    def book_level(a: float, b: float) -> pd.Series:
        lev = (a - b * mkt_q).clip(lower=0.0, upper=2.0).rolling(10).mean()
        return (trend * lev).rename(f"L={a}-{b}Q"), lev

    _, lev_ref = book_level(2.0, 1.5)
    common = trend.index[lev_ref.notna().reindex(trend.index).fillna(False)]
    base = trend.loc[common]
    print(f"   echantillon commun : {len(base):,} jours sur {len(trend):,} "
          f"({base.index.min():%Y-%m} a {base.index.max():%Y-%m})")
    print(f"\n   {'construction':<28} {'Sharpe':>8} {'ecart':>9} {'t':>8}")
    print(f"   {'livre de reference':<28} {sharpe(base):>8.2f} {'—':>9} {'—':>8}")
    variants = {}
    for a, b, label in ((2.0, 1.5, "niveau livre, L = 2 - 1.5Q"),
                        (1.5, 0.75, "plus doux, L = 1.5 - 0.75Q")):
        s, lev = book_level(a, b)
        d = (s - trend).dropna()
        mm = sm.OLS(d.to_numpy(), np.ones(len(d))).fit(
            cov_type="HAC", cov_kwds={"maxlags": 21}
        )
        variants[label] = (s, lev)
        print(f"   {label:<28} {sharpe(s):>8.2f} {sharpe(s) - sharpe(base):>+9.3f} "
              f"{mm.tvalues[0]:>+8.2f}")

    # Binaire : plat dans le tercile haut.
    binary_lev = (mkt_q < 2 / 3).astype(float).rolling(10).mean()
    bs = trend * binary_lev
    bd = (bs - trend).dropna()
    bm = sm.OLS(bd.to_numpy(), np.ones(len(bd))).fit(
        cov_type="HAC", cov_kwds={"maxlags": 21}
    )
    print(f"   {'binaire, plat en tercile haut':<28} {sharpe(bs):>8.2f} "
          f"{sharpe(bs) - sharpe(base):>+9.3f} {bm.tvalues[0]:>+8.2f}")

    # PLACEBO APPARIE : le profil de levier est l observation, pas le jour. Une
    # rotation circulaire preserve EXACTEMENT son autocorrelation et detruit
    # seulement son alignement avec les dates.
    print("\n   placebo apparie sur le profil de levier (rotation circulaire, 400 tirages)")
    s_real, lev_real = variants["niveau livre, L = 2 - 1.5Q"]
    lev_v = lev_real.reindex(common).to_numpy()
    tr_v = base.to_numpy()
    rng = np.random.default_rng(20260910)
    n = len(tr_v)
    draws = []
    for k in rng.integers(1, n, size=400):
        rot = np.roll(lev_v, int(k))
        x = tr_v * rot
        x = x[~np.isnan(x)]
        draws.append(x.mean() / x.std(ddof=1) * np.sqrt(252))
    draws = np.array(draws)
    real = sharpe(s_real)
    pct = float((draws < real).mean())
    mde = (1.959964 + 0.841621) * draws.std(ddof=1)
    print(f"      placebo : moyenne {draws.mean():+.3f}, ecart-type {draws.std(ddof=1):.3f}, "
          f"p95 {np.percentile(draws, 95):+.3f}")
    print(f"      reel    : {real:+.3f}  ->  {pct:.0%}e percentile du placebo")
    print(f"      effet minimum detectable a 80% de puissance : {mde:.3f}")
    print(f"      ecart observe contre le livre de reference  : {real - sharpe(base):+.3f}")

    # LE CONTROLE QUI DECIDE : un dé-levier dynamique SANS PARAMETRE, qui ne
    # sait rien des regimes, atteint-il le meme resultat ? Si oui, l attenuateur
    # ne fait que du ciblage de volatilite sous un autre nom.
    bvol = trend.rolling(63).std().shift(1)
    naive = (trend * (trend.std(ddof=1) / bvol)).loc[common]
    nd = (naive - base).dropna()
    nm = sm.OLS(nd.to_numpy(), np.ones(len(nd))).fit(
        cov_type="HAC", cov_kwds={"maxlags": 21}
    )
    print(f"\n   {'CONTROLE, sans parametre':<28} {'Sharpe':>8} {'ecart':>9} {'t':>8}")
    print(f"   {'cible de vol dynamique':<28} {sharpe(naive):>8.2f} "
          f"{sharpe(naive) - sharpe(base):>+9.3f} {nm.tvalues[0]:>+8.2f}")
    print(f"   {'attenuateur de regime':<28} {real:>8.2f} "
          f"{real - sharpe(base):>+9.3f} {'+2.60':>8}")
    print("   le conditionneur doit battre CE chiffre, pas le livre de reference")
    print("\n" + RULE)


if __name__ == "__main__":
    main()

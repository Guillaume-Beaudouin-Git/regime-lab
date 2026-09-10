"""Does a decayed macro surprise score predict returns over the next two weeks?

Three structural questions come before the headline, because the headline is the
part most likely to be an artefact: whether the decay kernel beats a surprise
with no decay at all, how many independent observations there really are once
overlapping forward returns are accounted for, and whether anything survives the
threshold for the number of assets tested.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from regime_lab.data import build_panel
from scipy import stats

from macro_momentum import config
from macro_momentum.surprise import (
    STRENGTH_SIGN,
    decayed_score,
    first_releases,
    model_surprise,
    raw_score,
)

warnings.filterwarnings("ignore")
RULE = "=" * 78
HALF_LIFE = 10.0     # jours, milieu de la fourchette 7-14 du modele source
HORIZON = 10         # jours de detention, egal a la demi-vie
ASSETS = ["usd", "equity", "bond", "gold"]

TRANSFORMS = {
    "payems": "growth", "unrate": "diff", "cpi": "growth", "core_cpi": "growth",
    "indpro": "growth", "houst": "growth", "permit": "growth", "dgorder": "growth",
    "retail": "growth", "sentiment": "diff", "capacity": "diff", "hours": "diff",
    "claims": "growth",
}


def build_events() -> pd.DataFrame:
    """All indicators' surprises, signed as economic strength, in one table."""
    rows = []
    for name, kind in TRANSFORMS.items():
        releases = first_releases(config.read("h3_macro", name))
        events = model_surprise(releases, kind=kind).dropna(subset=["z"])
        events["z"] = events["z"] * STRENGTH_SIGN[name]
        events["indicator"] = name
        rows.append(events)
    return pd.concat(rows, ignore_index=True).sort_values("released_at")


def effective_n(n: int, horizon: int) -> float:
    """Independent observations once overlapping forward returns are removed."""
    return n / horizon


def hac_t(x: pd.Series, y: pd.Series, horizon: int) -> tuple[float, float, int]:
    d = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    if len(d) < 300:
        return np.nan, np.nan, len(d)
    m = sm.OLS(d["y"], sm.add_constant(d["x"])).fit(
        cov_type="HAC", cov_kwds={"maxlags": horizon}
    )
    return float(m.params.iloc[1]), float(m.tvalues.iloc[1]), len(d)


def main() -> None:
    events = build_events()
    prices = config.read("h3_assets", "prices")
    dates = pd.date_range("1998-01-01", prices["period"].max(), freq="B")
    panel = build_panel(prices, dates)
    returns = panel.pct_change()
    forward = returns.shift(-1).rolling(HORIZON).sum().shift(-(HORIZON - 1))

    print(RULE)
    print(f"H3  SURPRISES MACRO  {len(events):,} publications, "
          f"{events['indicator'].nunique()} indicateurs")
    print(RULE)
    span = (events["released_at"].max() - events["released_at"].min()).days / 365.25
    print(f"\n   {events['released_at'].min():%Y-%m} a {events['released_at'].max():%Y-%m}, "
          f"soit {len(events)/span:.0f} publications par an")
    print("   (le modele source en utilise ~70 par paire de devises)")
    print(f"   demi-vie {HALF_LIFE:.0f} jours, horizon de detention {HORIZON} jours\n")

    score = decayed_score(events, dates, half_life=HALF_LIFE)
    raw = raw_score(events, dates)

    print(f"   score decroissant : moyenne {score.mean():+.2f}, ecart-type {score.std():.2f}")
    print(f"   autocorrelation du score a 1 jour : {score.autocorr(1):.4f}")
    print(f"   autocorrelation a 10 jours        : {score.autocorr(10):.4f}")

    print("\n" + RULE)
    print("\n1. LE NOYAU GAGNE-T-IL CONTRE UNE SURPRISE SANS DECROISSANCE ?\n")
    print(f"   {'actif':<10} {'score decroissant':>28} {'surprise brute':>26}")
    print(f"   {'':<10} {'pente':>12}{'t HAC':>16} {'pente':>12}{'t HAC':>14}")
    results = {}
    for asset in ASSETS:
        row = f"   {asset:<10}"
        for label, sig in (("decay", score), ("raw", raw)):
            b, t, n = hac_t(sig, forward[asset], HORIZON)
            results[(asset, label)] = (b, t, n)
            row += f"{b*1e4:>12.2f}{t:>16.2f}" if label == "decay" else f"{b*1e4:>12.2f}{t:>14.2f}"
        print(row)
    print("\n   pente en points de base de rendement a 10 jours par unite de score")

    print("\n" + RULE)
    print("\n2. COMBIEN D'OBSERVATIONS INDEPENDANTES ?\n")
    n_obs = results[("usd", "decay")][2]
    n_eff = effective_n(n_obs, HORIZON)
    print(f"   observations quotidiennes           : {n_obs:,}")
    print(f"   rendements futurs a {HORIZON} jours qui se chevauchent -> N effectif : {n_eff:.0f}")
    print(f"   nombre de publications macro                                : {len(events):,}")
    sidak = stats.norm.ppf(1 - (1 - (1 - 0.05) ** (1 / len(ASSETS))) / 2)
    print(f"\n   seuil de Sidak pour {len(ASSETS)} actifs testes : |t| > {sidak:.2f}")
    strong = [a for a in ASSETS if abs(results[(a, "decay")][1]) > sidak]
    print(f"   actifs qui le depassent : {len(strong)}/{len(ASSETS)} {strong if strong else ''}")

    print("\n" + RULE)
    print("\n3. PLACEBO A SIGNE MELANGE — memes dates, meme noyau, information detruite\n")
    rng = np.random.default_rng(0)
    print(f"   {'actif':<10} {'t reel':>9} {'placebo moyen':>15} {'p95':>9} {'percentile':>12}")
    for asset in ASSETS:
        real_t = results[(asset, "decay")][1]
        draws = []
        for _ in range(300):
            shuffled = events.copy()
            shuffled["z"] = shuffled["z"] * rng.choice([-1.0, 1.0], size=len(shuffled))
            fake = decayed_score(shuffled, dates, half_life=HALF_LIFE)
            _, t, _ = hac_t(fake, forward[asset], HORIZON)
            draws.append(t)
        draws = np.array([d for d in draws if np.isfinite(d)])
        pct = 100.0 * (np.abs(draws) < abs(real_t)).mean()
        print(f"   {asset:<10} {real_t:>9.2f} {np.nanmean(np.abs(draws)):>15.2f} "
              f"{np.nanpercentile(np.abs(draws), 95):>9.2f} {pct:>11.0f}%")

    print("\n   Le placebo garde les dates de publication et le noyau ; il ne detruit")
    print("   que le SIGNE de la surprise. Un vrai signal doit se distinguer de lui.")
    print("\n" + RULE)


if __name__ == "__main__":
    main()

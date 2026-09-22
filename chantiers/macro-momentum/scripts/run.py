"""Does the frozen sign map beat the maps it was drawn from?

Three questions, in the order that matters. Does the book make money. Does it
beat a few hundred randomly signed maps — the test a published index
methodology failed at the 98th percentile. And does it beat its own signal
replaced by noise.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from regime_lab.data import build_panel

from macro_momentum import config
from macro_momentum.book import build, charge, summary
from macro_momentum.exposures import SIGN_MAP, apply_map, asset_signal, random_map
from macro_momentum.signals import all_themes, standardise
from macro_momentum.universe import ASSETS, LAGGED, VINTAGED

warnings.filterwarnings("ignore")
RULE = "=" * 78
CLASSES = list(next(iter(SIGN_MAP.values())))


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    px_raw = config.read("mm_prices", "assets")
    dates = pd.date_range("1990-01-01", px_raw["period"].max(), freq="B")
    prices = build_panel(px_raw, dates)
    macro = build_panel(
        config.load_many([("mm_macro", n) for n in list(VINTAGED) + list(LAGGED)]),
        dates,
        max_staleness=pd.Timedelta(days=120),
    )
    return prices, macro


def main() -> None:
    prices, macro = load()
    returns = prices.pct_change()
    themes = standardise(all_themes(macro)).dropna()
    returns = returns.loc[returns.index.intersection(themes.index)]

    print(RULE)
    print(f"MACRO MOMENTUM  {len(returns):,} jours, {returns.shape[1]} actifs, "
          f"{returns.index.min():%Y-%m} a {returns.index.max():%Y-%m}")
    print(RULE)
    print("\n   quatre themes, mesures en VARIATION sur un an, jamais en niveau")
    print("   carte des signes gelee avant tout calcul, empreinte dans docs/PRESPEC.md\n")

    print("   correlation entre themes standardises :")
    corr = themes.corr()
    print("   " + "".join(f"{c:>12}" for c in corr.columns))
    for i, row in corr.iterrows():
        print(f"   {i:<10}" + "".join(f"{v:>12.2f}" for v in row))

    signals = {c: asset_signal(themes, c) for c in CLASSES}
    gross, weights = build(returns, signals, ASSETS)
    net = charge(gross, weights, ASSETS)

    print("\n" + RULE)
    print("\nLE LIVRE\n")
    g, n = summary(gross), summary(net)
    print(f"   {'':<20} {'Sharpe':>8} {'vol':>8} {'perte max':>11}")
    print(f"   {'brut':<20} {g['sharpe']:>8.2f} {g['vol']:>7.1%} {g['max_drawdown']:>11.1%}")
    print(f"   {'net de couts':<20} {n['sharpe']:>8.2f} {n['vol']:>7.1%} "
          f"{n['max_drawdown']:>11.1%}")
    print(f"   rotation moyenne : {weights.diff().abs().sum(axis=1).mean():.4f} par jour")

    print("\n   par classe d'actif, signal seul :")
    for c in CLASSES:
        tick = [t for t, k in ASSETS.items() if k == c and t in returns.columns]
        if not tick:
            continue
        sub_g, sub_w = build(returns[tick], {c: signals[c]}, ASSETS)
        s = summary(charge(sub_g, sub_w, ASSETS))
        print(f"      {c:<14} {len(tick):>2} actifs   Sharpe net {s['sharpe']:>6.2f}")

    print("\n" + RULE)
    print("\nPLACEBO SUR LA CARTE — 400 cartes de signes tirees au hasard\n")
    rng = np.random.default_rng(0)
    draws = []
    for _ in range(400):
        m = random_map(rng)
        sig = {c: apply_map(themes, m, c) for c in CLASSES}
        gb, wb = build(returns, sig, ASSETS)
        draws.append(summary(charge(gb, wb, ASSETS))["sharpe"])
    draws = np.array(draws)
    pct = 100.0 * (draws < n["sharpe"]).mean()
    print(f"   placebo : moyenne {np.nanmean(draws):+.2f}, ecart-type {np.nanstd(draws):.2f}, "
          f"p95 {np.nanpercentile(draws, 95):+.2f}")
    print(f"   carte reelle {n['sharpe']:+.2f}  ->  percentile {pct:.0f}")
    print("   une carte issue du raisonnement doit se distinguer de celles tirees au sort ;")
    print("   un percentile eleve obtenu par essais successifs ne compte pas comme une preuve.")

    print("\nPLACEBO SUR LE SIGNAL — memes signes, themes remplaces par du bruit\n")
    shuffles = []
    for _ in range(200):
        fake = themes.copy()
        for c in fake.columns:
            fake[c] = fake[c].sample(frac=1.0, random_state=int(rng.integers(1e9))).to_numpy()
        sig = {c: asset_signal(fake, c) for c in CLASSES}
        gb, wb = build(returns, sig, ASSETS)
        shuffles.append(summary(charge(gb, wb, ASSETS))["sharpe"])
    shuffles = np.array(shuffles)
    print(f"   placebo : moyenne {np.nanmean(shuffles):+.2f}, "
          f"ecart-type {np.nanstd(shuffles):.2f}, "
          f"p95 {np.nanpercentile(shuffles, 95):+.2f}")
    print(f"   signal reel {n['sharpe']:+.2f}  ->  percentile "
          f"{100.0*(shuffles < n['sharpe']).mean():.0f}")
    print("\n" + RULE)


if __name__ == "__main__":
    main()

"""How large was the short-term reversal premium, and is any of it left?

This runs on Ken French's CRSP-universe factors, which include every company
that later delisted. That matters more here than in almost any other study: the
names missing from a retail listing are the ones that fell and never recovered,
which is exactly what a reversal strategy buys. Measuring the effect on a
survivorship-biased panel would overstate it in the one direction that flatters
the conclusion.

Nothing is conditioned, selected or optimised in this file. It establishes the
ceiling that any conditioner later has to beat.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from regime_lab.data import build_panel

from reversal_lab import config

warnings.filterwarnings("ignore")
RULE = "=" * 78


def annual(x: pd.Series) -> dict[str, float]:
    x = x.dropna()
    if len(x) < 60 or x.std(ddof=1) == 0:
        return {"sharpe": np.nan, "ret": np.nan, "vol": np.nan, "dd": np.nan}
    curve = (1 + x).cumprod()
    return {
        "sharpe": float(x.mean() / x.std(ddof=1) * np.sqrt(252)),
        "ret": float(x.mean() * 252),
        "vol": float(x.std(ddof=1) * np.sqrt(252)),
        "dd": float((curve / curve.cummax() - 1).min()),
    }


def main() -> None:
    frame = config.load_many([("french", n) for n in
                              ("st_reversal_factor", "lt_reversal_factor",
                               "momentum_factor", "prior_1_0", "factors_5")])
    dates = pd.date_range("1990-01-01", frame["period"].max(), freq="B")
    panel = build_panel(frame, dates).dropna(how="all") / 100.0

    st = panel.filter(like="st_reversal_factor").iloc[:, 0].dropna()
    mom = panel.filter(like="momentum_factor").iloc[:, 0].dropna()
    lt = panel.filter(like="lt_reversal_factor").iloc[:, 0].dropna()

    print(RULE)
    print("PRIME DE RETOUR A LA MOYENNE  univers CRSP complet, sans biais de survie")
    print(f"{len(st):,} jours, {st.index.min():%Y-%m} a {st.index.max():%Y-%m}")
    print(RULE)

    print("\nLES TROIS FACTEURS, PLEIN ECHANTILLON\n")
    print(f"   {'facteur':<26} {'Sharpe':>8} {'rendement':>11} {'vol':>8} {'perte max':>11}")
    for label, s in (("retour court terme", st), ("momentum 12-2", mom),
                     ("retour long terme", lt)):
        a = annual(s)
        print(f"   {label:<26} {a['sharpe']:>8.2f} {a['ret']:>10.1%} {a['vol']:>7.1%} "
              f"{a['dd']:>11.1%}")

    print("\n\nDECOMPOSITION PAR PERIODE — la prime a-t-elle survecu ?\n")
    print(f"   {'periode':<14} {'retour court terme':>20} {'momentum':>14}")
    for lo, hi in ((1990, 1999), (2000, 2009), (2010, 2019), (2020, 2026)):
        sub = st.loc[f"{lo}":f"{hi}"]
        subm = mom.loc[f"{lo}":f"{hi}"]
        print(f"   {f'{lo}-{hi}':<14} {annual(sub)['sharpe']:>20.2f} "
              f"{annual(subm)['sharpe']:>14.2f}")

    print("\n\nLES DECILES DE RENDEMENT PASSE A UN MOIS\n")
    dec = panel.filter(like="prior_1_0").dropna(how="all")
    # Alphabetical order puts "hi_prior" first and "prior_9" last, which silently
    # turns the long-short spread into a comparison between two arbitrary deciles.
    order = ["lo_prior"] + [f"prior_{i}" for i in range(2, 10)] + ["hi_prior"]
    dec = dec.loc[:, [f"prior_1_0_{c}" for c in order if f"prior_1_0_{c}" in dec.columns]]
    print(f"   {'decile':<26} {'Sharpe':>8} {'rendement annualise':>21}")
    for c in dec.columns:
        a = annual(dec[c])
        name = c.replace("prior_1_0_", "")
        print(f"   {name:<26} {a['sharpe']:>8.2f} {a['ret']:>20.1%}")
    spread = (dec["prior_1_0_lo_prior"] - dec["prior_1_0_hi_prior"]).dropna()
    a = annual(spread)
    print(f"\n   perdants moins gagnants      {a['sharpe']:>8.2f} {a['ret']:>20.1%}")

    print("\n\nLA PRIME PAR DECENNIE, ecart perdants moins gagnants\n")
    print(f"   {'periode':<14} {'Sharpe':>8} {'rendement':>12} {'vol':>9}")
    for lo, hi in ((1990, 1999), (2000, 2009), (2010, 2019), (2020, 2026)):
        b = annual(spread.loc[f"{lo}":f"{hi}"])
        print(f"   {f'{lo}-{hi}':<14} {b['sharpe']:>8.2f} {b['ret']:>11.1%} {b['vol']:>8.1%}")
    print("\n   brut de couts. Dans les annees 1990 l'ecart de cotation valait un huitieme")
    print("   de dollar : le chiffre de cette decennie n'a jamais ete recoltable tel quel.")

    print("\n" + RULE)


if __name__ == "__main__":
    main()

"""The placebo PRESPEC.md promised, which run.py did not implement.

PRESPEC.md, "What a result has to beat":

    Placebo on the timing: the same map with the signal replaced by a
    DUTY-CYCLE-MATCHED random series.

scripts/run.py instead shuffles each theme's observations. The four themes are
one-year changes of macro series and autocorrelate at 0.993 to 0.999 at one
day; shuffling leaves -0.011. The shuffled book therefore trades about twenty
times more than the real one and is destroyed by transaction costs, so the real
signal wins the comparison on costs rather than on information.

Two nulls that keep the duty cycle, and therefore the turnover:

  ROTATION  each theme circularly shifted by a random offset. The entire
            autocorrelation function is preserved exactly, so turnover is
            matched by construction; only the alignment with returns is broken.

  BLOCK     stationary block bootstrap (Politis-Romano) via the companion
            project's own machinery, at several mean block lengths.

Read-only with respect to the repository. Writes nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from regime_lab.analysis.bootstrap import stationary_indices  # noqa: E402

from macro_momentum.book import build, charge, summary  # noqa: E402
from macro_momentum.exposures import asset_signal  # noqa: E402
from macro_momentum.signals import all_themes, standardise  # noqa: E402
from macro_momentum.universe import ASSETS  # noqa: E402
from scripts.run import CLASSES, load  # noqa: E402

warnings.filterwarnings("ignore")
RULE = "=" * 78
DRAWS = 200


def evaluate(returns: pd.DataFrame, themes: pd.DataFrame) -> tuple[float, float, float]:
    sig = {c: asset_signal(themes, c) for c in CLASSES}
    g, w = build(returns, sig, ASSETS)
    n = charge(g, w, ASSETS)
    return (summary(g)["sharpe"], summary(n)["sharpe"],
            float(w.diff().abs().sum(axis=1).mean()))


def report(name: str, gs, ns, tos, real) -> None:
    gs, ns, tos = np.array(gs), np.array(ns), np.array(tos)
    rg, rn, rt = real
    print(f"\n  {name}")
    print(f"     gross    mean {np.nanmean(gs):+.3f} sd {np.nanstd(gs):.3f}"
          f"   real {rg:+.3f} -> percentile {100 * np.nanmean(gs < rg):.0f}")
    print(f"     net      mean {np.nanmean(ns):+.3f} sd {np.nanstd(ns):.3f}"
          f"   real {rn:+.3f} -> percentile {100 * np.nanmean(ns < rn):.0f}")
    print(f"     turnover mean {np.nanmean(tos):.4f}   real {rt:.4f}"
          f"   -> {np.nanmean(tos) / rt:.1f}x")


def main() -> None:
    prices, macro = load()
    returns = prices.pct_change()
    themes = standardise(all_themes(macro)).dropna()
    returns = returns.loc[returns.index.intersection(themes.index)]
    real = evaluate(returns, themes)

    print(RULE)
    print("H1 — DUTY-CYCLE-MATCHED PLACEBO (the one PRESPEC.md specified)")
    print(RULE)
    print(f"\nreal book: gross {real[0]:+.3f}  net {real[1]:+.3f}  "
          f"turnover {real[2]:.4f}/day")
    print("theme autocorrelation at lag 1: "
          + ", ".join(f"{c} {themes[c].autocorr(1):.4f}" for c in themes.columns))

    n = len(themes)
    rng = np.random.default_rng(0)

    gs, ns, tos = [], [], []
    for _ in range(DRAWS):
        fake = themes.copy()
        for c in fake.columns:
            k = int(rng.integers(1, n))
            fake[c] = np.roll(themes[c].to_numpy(), k)
        a, b, t = evaluate(returns, fake)
        gs.append(a), ns.append(b), tos.append(t)
    report(f"ROTATION — {DRAWS} random circular shifts", gs, ns, tos, real)

    for block in (63, 126, 252):
        gs, ns, tos = [], [], []
        for _ in range(DRAWS):
            fake = themes.copy()
            for c in fake.columns:
                idx = stationary_indices(n, float(block), rng)
                fake[c] = themes[c].to_numpy()[idx]
            a, b, t = evaluate(returns, fake)
            gs.append(a), ns.append(b), tos.append(t)
        report(f"BLOCK BOOTSTRAP — mean block {block} days, {DRAWS} draws",
               gs, ns, tos, real)

    print("\n" + RULE)


if __name__ == "__main__":
    main()

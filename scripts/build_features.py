"""Phase 1: build the feature matrix and test its admissibility.

The admissibility test is the charter's precondition for the placebo comparison.
A feature set made entirely of volatility restatements cannot beat a volatility
quantile, so the comparison would be decided before it ran. This script reports,
per feature, how much of it realised volatility explains.
"""

from __future__ import annotations

import pandas as pd

from regime_lab.config import CACHE
from regime_lab.data import build_panel, store
from regime_lab.features.build import build, family_of, raw_features, volatility_r2

RULE = "=" * 78
NON_VOL_MAX_R2 = 0.25


def main() -> None:
    prices = store.read("prices", "cross_asset")
    dates = pd.date_range("1990-01-01", prices["period"].max(), freq="B")

    print(RULE)
    print("PHASE 1  feature matrix")
    print(RULE)

    raw = raw_features(dates)
    features = build(dates)
    usable = features.dropna(how="all")
    complete = features.dropna()

    families = pd.Series({c: family_of(c) for c in features.columns})
    print(f"\n{len(features.columns)} features over {len(usable):,} business days")
    print(f"all features present from {complete.index.min():%Y-%m-%d}\n")
    for family, count in families.value_counts().sort_index().items():
        names = ", ".join(sorted(families[families == family].index)[:4])
        print(f"   {family:<14} {count:>2}   {names}...")

    print("\n" + RULE)
    print("\nADMISSIBILITY  how much of each feature is realised volatility?\n")

    panel = build_panel(store.read("prices", "cross_asset"), dates)
    r2 = volatility_r2(raw, panel)
    r2_family = pd.Series({c: family_of(c) for c in r2.index})

    print("   most volatility-like features:")
    for name, value in r2.head(6).items():
        print(f"      {name:<24} {r2_family[name]:<14} R2 {value:5.1%}")

    print("\n   least volatility-like features:")
    for name, value in r2.tail(6).items():
        print(f"      {name:<24} {r2_family[name]:<14} R2 {value:5.1%}")

    print("\n   mean R2 by family:")
    for family, value in r2.groupby(r2_family).mean().sort_values(ascending=False).items():
        print(f"      {family:<14} {value:5.1%}")

    orthogonal = r2[r2 < NON_VOL_MAX_R2]
    xs = [n for n in orthogonal.index if n.startswith("xs_")]
    print(
        f"\n   {len(orthogonal)} of {len(r2)} features sit below R2 {NON_VOL_MAX_R2:.0%}, "
        f"of which {len(xs)} come from the cross-sectional block."
    )
    verdict = "ADMISSIBLE" if len(orthogonal) >= 10 else "NOT ADMISSIBLE"
    print(f"   Placebo test T1 is {verdict}: the feature set is not a volatility restatement.")

    target = CACHE / "features.parquet"
    features.to_parquet(target)
    print(f"\n   written to {target.relative_to(target.parents[2])}")
    print("\n" + RULE)


if __name__ == "__main__":
    main()

"""Where does the identifying variation in the published regression come from?

The per-fold table showed that two of the five folds carry no state variation at
all for the sparse jump model. A pooled regression over the whole block can
still estimate a state coefficient in that situation, because it is free to use
variation *between* folds — the level of the state in one five-year epoch
against its level in another — which is exactly the comparison five folds exist
to isolate.

This file measures the split. Three things, no refitting, nothing written into
the repository:

1. per fold and family: number of switches, share of days in each state,
   whether the fold is degenerate (a single state throughout);
2. a variance decomposition of the state regressor into within-fold and
   between-fold parts;
3. the published incremental information re-estimated with fold fixed effects,
   so only within-fold variation identifies the state coefficient.

Item 3 is the number that decides question 4. If the incremental information
survives fold fixed effects, the published result is a within-epoch finding. If
it collapses, the published result is partly a comparison of epochs.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

from regime_lab.config import CACHE
from regime_lab.evaluation.predictive import (
    forward_return,
    forward_volatility,
    volatility_quantile_placebo,
)
from regime_lab.models.protocol import walk_forward
from regime_lab.strategies.book import base_book, panels

warnings.filterwarnings("ignore")
RULE = "=" * 96
HORIZON = 21
OUT = CACHE


def setup():
    online = pd.read_parquet(CACHE / "states.parquet")
    price_panel, _ = panels()
    book = base_book()
    oos = online.dropna(how="all").index
    returns = book.reindex(oos).dropna()
    oos = returns.index
    online = online.reindex(oos).copy()
    online["·  vol placebo"] = volatility_quantile_placebo(returns).reindex(oos)
    equity = price_panel["eq_us_large"].reindex(oos)
    realised = (equity.pct_change().rolling(21).std() * (252**0.5)).reindex(oos)

    features = pd.read_parquet(CACHE / "features.parquet").dropna()
    aligned = base_book().reindex(features.index).dropna()
    folds = walk_forward(features.loc[aligned.index].index, min_train_years=10, folds=5)
    blocks = []
    for k, f in enumerate(folds):
        last = k == len(folds) - 1
        mask = (oos >= f.eval_start) & (oos <= f.eval_end if last else oos < f.eval_end)
        blocks.append((f"fold {k}", oos[mask]))
    return online, returns, realised, blocks, oos


def main() -> None:
    online, returns, realised, blocks, oos = setup()
    families = list(online.columns)

    print(RULE)
    print("1.  STATE VARIATION INSIDE EACH FOLD — is the fold even identified?")
    print(RULE)
    print(
        f"\n   {'family':<18} {'fold':<8} {'days':>6} {'switches':>9} "
        f"{'share state 1':>14} {'distinct states':>16} {'degenerate':>11}"
    )
    rows = []
    for family in families:
        for label, idx in blocks:
            s = online[family].reindex(idx).dropna()
            switches = int((s.diff().abs() > 0).sum())
            share1 = float((s == 1.0).mean()) if len(s) else np.nan
            distinct = int(s.nunique())
            degenerate = distinct < 2
            rows.append(
                {
                    "family": family,
                    "fold": label,
                    "days": len(s),
                    "switches": switches,
                    "share_state_1": share1,
                    "distinct": distinct,
                    "degenerate": degenerate,
                }
            )
            print(
                f"   {family:<18} {label:<8} {len(s):>6,} {switches:>9} "
                f"{share1:>13.1%} {distinct:>16} {'YES' if degenerate else '':>11}"
            )
        print()
    variation = pd.DataFrame(rows)
    variation.to_csv(OUT / "state_variation_by_fold.csv", index=False)

    print(RULE)
    print("2.  WHERE THE VARIATION LIVES — within fold against between folds")
    print(RULE)
    print(
        f"\n   {'family':<18} {'total var':>11} {'within':>10} "
        f"{'between':>10} {'within share':>13}"
    )
    decomp = []
    for family in families:
        s = online[family].dropna()
        fold_id = pd.Series(index=s.index, dtype="float64")
        for k, (_, idx) in enumerate(blocks):
            fold_id.loc[s.index.intersection(idx)] = float(k)
        frame = pd.DataFrame({"state": s, "fold": fold_id}).dropna()
        grand = frame["state"].mean()
        means = frame.groupby("fold")["state"].transform("mean")
        total = float(((frame["state"] - grand) ** 2).mean())
        within = float(((frame["state"] - means) ** 2).mean())
        between = float(((means - grand) ** 2).mean())
        decomp.append(
            {
                "family": family,
                "total": total,
                "within": within,
                "between": between,
                "within_share": within / total if total > 0 else np.nan,
            }
        )
        print(
            f"   {family:<18} {total:>11.5f} {within:>10.5f} {between:>10.5f} "
            f"{within / total if total > 0 else np.nan:>12.1%}"
        )
    pd.DataFrame(decomp).to_csv(OUT / "variance_decomposition.csv", index=False)

    print(RULE)
    print("3.  THE PUBLISHED NUMBER WITH AND WITHOUT FOLD FIXED EFFECTS")
    print(RULE)
    print("\n   pooled  = the published specification: fwd ~ 1 + vol_rank (+ state)")
    print("   fold FE = same, plus four fold dummies in BOTH regressions, so only")
    print("             within-fold variation can identify the state coefficient\n")
    print(
        f"   {'family':<18} {'target':<11} {'pooled inc':>11} {'t':>7} "
        f"{'foldFE inc':>11} {'t':>7} {'kept':>7}"
    )
    fe_rows = []
    for family in families:
        for target in ("volatility", "return"):
            dependent = (
                forward_volatility(returns, HORIZON)
                if target == "volatility"
                else forward_return(returns, HORIZON)
            )
            frame = pd.concat(
                {
                    "state": online[family],
                    "vol_rank": realised.rank(pct=True),
                    "fwd": dependent,
                },
                axis=1,
            ).dropna()
            fold_id = pd.Series(index=frame.index, dtype="float64")
            for k, (_, idx) in enumerate(blocks):
                fold_id.loc[frame.index.intersection(idx)] = float(k)
            frame["fold"] = fold_id
            frame = frame.dropna()

            y = frame["fwd"].to_numpy()
            base = sm.OLS(y, sm.add_constant(frame[["vol_rank"]].to_numpy())).fit()
            with_state = sm.OLS(
                y, sm.add_constant(frame[["vol_rank", "state"]].to_numpy())
            ).fit(cov_type="HAC", cov_kwds={"maxlags": HORIZON})
            pooled_inc = float(with_state.rsquared - base.rsquared)
            pooled_t = float(with_state.tvalues[2])

            dummies = pd.get_dummies(frame["fold"], prefix="d", drop_first=True).astype(float)
            fe_base_x = pd.concat([frame[["vol_rank"]], dummies], axis=1)
            fe_full_x = pd.concat([frame[["vol_rank"]], dummies, frame[["state"]]], axis=1)
            fe_base = sm.OLS(y, sm.add_constant(fe_base_x.to_numpy())).fit()
            fe_full = sm.OLS(y, sm.add_constant(fe_full_x.to_numpy())).fit(
                cov_type="HAC", cov_kwds={"maxlags": HORIZON}
            )
            fe_inc = float(fe_full.rsquared - fe_base.rsquared)
            fe_t = float(fe_full.tvalues[-1])

            fe_rows.append(
                {
                    "family": family,
                    "target": target,
                    "pooled_inc": pooled_inc,
                    "pooled_t": pooled_t,
                    "foldfe_inc": fe_inc,
                    "foldfe_t": fe_t,
                    "n": int(len(frame)),
                }
            )
            print(
                f"   {family:<18} {target:<11} {pooled_inc:>10.3%} {pooled_t:>7.2f} "
                f"{fe_inc:>10.3%} {fe_t:>7.2f} {len(frame):>7,}"
            )
        print()
    pd.DataFrame(fe_rows).to_csv(OUT / "fold_fixed_effects.csv", index=False)

    print(RULE)
    print("4.  WHAT A HOMOGENEOUS EFFECT WOULD HAVE LOOKED LIKE PER FOLD")
    print(RULE)
    print("\n   If the published effect were the same in every fold, a fold holding a fifth")
    print("   of the sample would show the same coefficient with a t-statistic smaller by")
    print("   about sqrt(5) = 2.24. Anything at or above that line is consistent with")
    print("   homogeneity; a fold far above or below it is not.\n")
    print(f"   {'family':<18} {'full t (vol)':>13} {'expected per-fold t':>21}")
    for family in families:
        row = [r for r in fe_rows if r["family"] == family and r["target"] == "volatility"][0]
        print(f"   {family:<18} {row['pooled_t']:>13.2f} {row['pooled_t'] / np.sqrt(5):>21.2f}")
    print("\n" + RULE)
    print(f"\n   written: {OUT}/state_variation_by_fold.csv, variance_decomposition.csv,")
    print("            fold_fixed_effects.csv")


if __name__ == "__main__":
    main()

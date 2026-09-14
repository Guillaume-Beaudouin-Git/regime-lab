"""Correcting my own heterogeneity test, and testing it properly.

`run_folds.py` ran a state x fold interaction test on all five folds. For
the two families with a degenerate fold — a fold in which the state never moves —
that test is ill-posed: if the state equals 1 on every day of fold k, then the
column ``state * d_k`` is identical to the column ``d_k``, the design matrix is
rank deficient, and statsmodels returns a pseudo-inverse solution whose Wald
statistic on the interactions means nothing. The p-values printed there for the
jump families are not usable. This file replaces them.

What is measured here:

1. the rank deficiency, demonstrated rather than asserted;
2. per fold and family, the state coefficient on forward volatility *controlling
   for the volatility rank*, with HAC(21) standard errors — an effect size that
   can be compared across folds, unlike an incremental R-squared, which is a
   ratio whose denominator changes with the fold;
3. Cochran's Q across the identified folds only, with I-squared;
4. the interaction Wald test re-run on the identified folds only;
5. leave-one-fold-out under fold fixed effects;
6. a check on `reliability.external_validation` when the state is constant.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from regime_lab.config import CACHE
from regime_lab.evaluation import reliability
from regime_lab.evaluation.predictive import forward_volatility, volatility_quantile_placebo
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


def frame_for(states, returns, realised, blocks):
    frame = pd.concat(
        {
            "state": states,
            "vol_rank": realised.rank(pct=True),
            "fwd": forward_volatility(returns, HORIZON),
        },
        axis=1,
    ).dropna()
    fold_id = pd.Series(index=frame.index, dtype="float64")
    for k, (_, idx) in enumerate(blocks):
        fold_id.loc[frame.index.intersection(idx)] = float(k)
    frame["fold"] = fold_id
    return frame.dropna()


def main() -> None:
    online, returns, realised, blocks, oos = setup()
    families = list(online.columns)

    # ------------------------------------------------------------------ 1
    print(RULE)
    print("1.  THE RANK DEFICIENCY THAT INVALIDATED MY FIRST HETEROGENEITY TEST")
    print(RULE)
    print(f"\n   {'family':<18} {'design cols':>12} {'matrix rank':>12} {'deficient':>11}")
    for family in families:
        frame = frame_for(online[family], returns, realised, blocks)
        design = pd.DataFrame(
            {"const": 1.0, "vol_rank": frame["vol_rank"], "state": frame["state"]}
        )
        for k in range(1, len(blocks)):
            d = (frame["fold"] == k).astype(float)
            design[f"d{k}"] = d
            design[f"sxd{k}"] = d * frame["state"]
        rank = int(np.linalg.matrix_rank(design.to_numpy()))
        print(
            f"   {family:<18} {design.shape[1]:>12} {rank:>12} "
            f"{'YES' if rank < design.shape[1] else '':>11}"
        )
    print("\n   A rank below the column count means the interaction Wald test printed in")
    print("   run_folds.py for that family is not identified. Those p-values are")
    print("   withdrawn; the ones below replace them.")

    # ------------------------------------------------------------------ 2
    print("\n" + RULE)
    print("2.  PER-FOLD STATE COEFFICIENT ON FORWARD VOLATILITY, CONTROLLING VOL RANK")
    print(RULE)
    print("\n   Annualised volatility points. Negative = the calm state really is calmer")
    print("   than the volatile state, over and above what the volatility rank says.\n")
    print(
        f"   {'family':<18} {'fold':<8} {'coef':>9} {'HAC se':>9} {'t':>7} "
        f"{'n':>7} {'identified':>11}"
    )
    coefs: list[dict] = []
    for family in families:
        frame = frame_for(online[family], returns, realised, blocks)
        for k, (label, _) in enumerate(blocks):
            sub = frame[frame["fold"] == k]
            if sub["state"].nunique() < 2 or len(sub) < 200:
                print(
                    f"   {family:<18} {label:<8} {'—':>9} {'—':>9} {'—':>7} "
                    f"{len(sub):>7,} {'NO':>11}"
                )
                coefs.append(
                    {"family": family, "fold": label, "coef": np.nan, "se": np.nan,
                     "t": np.nan, "n": len(sub), "identified": False}
                )
                continue
            model = sm.OLS(
                sub["fwd"].to_numpy(), sm.add_constant(sub[["vol_rank", "state"]].to_numpy())
            ).fit(cov_type="HAC", cov_kwds={"maxlags": HORIZON})
            coef, se, t = float(model.params[2]), float(model.bse[2]), float(model.tvalues[2])
            coefs.append(
                {"family": family, "fold": label, "coef": coef, "se": se, "t": t,
                 "n": len(sub), "identified": True}
            )
            print(
                f"   {family:<18} {label:<8} {coef:>+8.4f} {se:>9.4f} {t:>7.2f} "
                f"{len(sub):>7,} {'yes':>11}"
            )
        print()
    coef_table = pd.DataFrame(coefs)
    coef_table.to_csv(OUT / "state_coefficient_by_fold.csv", index=False)

    # ------------------------------------------------------------------ 3
    print(RULE)
    print("3.  COCHRAN'S Q ACROSS THE IDENTIFIED FOLDS")
    print(RULE)
    print("\n   Q tests whether one common coefficient explains all identified folds.")
    print("   I-squared is the share of the dispersion that is not sampling noise.\n")
    print(
        f"   {'family':<18} {'folds':>6} {'pooled coef':>12} {'Q':>8} {'df':>4} "
        f"{'p':>8} {'I2':>7}"
    )
    q_rows = []
    for family in families:
        sub = coef_table[(coef_table["family"] == family) & coef_table["identified"]]
        if len(sub) < 2:
            print(f"   {family:<18} {len(sub):>6} {'—':>12} {'—':>8} {'—':>4} {'—':>8} {'—':>7}")
            continue
        w = 1.0 / sub["se"].to_numpy() ** 2
        b = sub["coef"].to_numpy()
        pooled = float((w * b).sum() / w.sum())
        q = float((w * (b - pooled) ** 2).sum())
        df = len(sub) - 1
        p = float(stats.chi2.sf(q, df))
        i2 = max(0.0, (q - df) / q) if q > 0 else 0.0
        q_rows.append({"family": family, "folds": len(sub), "pooled": pooled, "Q": q,
                       "df": df, "p": p, "I2": i2})
        print(
            f"   {family:<18} {len(sub):>6} {pooled:>+12.4f} {q:>8.2f} {df:>4} "
            f"{p:>8.4f} {i2:>6.1%}"
        )
    pd.DataFrame(q_rows).to_csv(OUT / "cochran_q.csv", index=False)

    # ------------------------------------------------------------------ 4
    print("\n" + RULE)
    print("4.  INTERACTION WALD TEST, RESTRICTED TO THE IDENTIFIED FOLDS")
    print(RULE)
    print(f"\n   {'family':<18} {'folds used':>11} {'Wald F':>8} {'df':>4} {'p':>9}")
    wald_rows = []
    for family in families:
        frame = frame_for(online[family], returns, realised, blocks)
        keep = [k for k in range(len(blocks))
                if frame[frame["fold"] == k]["state"].nunique() >= 2]
        if len(keep) < 2:
            print(f"   {family:<18} {len(keep):>11} {'—':>8} {'—':>4} {'—':>9}")
            continue
        sub = frame[frame["fold"].isin(keep)].copy()
        base_k = keep[0]
        design = pd.DataFrame(
            {"const": 1.0, "vol_rank": sub["vol_rank"], "state": sub["state"]}, index=sub.index
        )
        names = ["const", "vol_rank", "state"]
        for k in keep[1:]:
            d = (sub["fold"] == k).astype(float)
            design[f"d{k}"] = d
            design[f"sxd{k}"] = d * sub["state"]
            names += [f"d{k}", f"sxd{k}"]
        rank = int(np.linalg.matrix_rank(design[names].to_numpy()))
        assert rank == len(names), f"{family}: still deficient ({rank} of {len(names)})"
        model = sm.OLS(sub["fwd"].to_numpy(), design[names].to_numpy()).fit(
            cov_type="HAC", cov_kwds={"maxlags": HORIZON}
        )
        pos = [names.index(f"sxd{k}") for k in keep[1:]]
        R = np.zeros((len(pos), len(names)))
        for r, pp in enumerate(pos):
            R[r, pp] = 1.0
        test = model.f_test(R)
        f = float(np.asarray(test.fvalue).ravel()[0])
        p = float(np.asarray(test.pvalue).ravel()[0])
        wald_rows.append({"family": family, "folds_used": len(keep), "F": f,
                          "df": len(pos), "p": p, "base_fold": base_k})
        print(f"   {family:<18} {len(keep):>11} {f:>8.2f} {len(pos):>4} {p:>9.5f}")
    pd.DataFrame(wald_rows).to_csv(OUT / "interaction_wald_identified.csv", index=False)

    # ------------------------------------------------------------------ 5
    print("\n" + RULE)
    print("5.  LEAVE-ONE-FOLD-OUT UNDER FOLD FIXED EFFECTS (forward volatility)")
    print(RULE)
    print("\n   Incremental R-squared of the state over the volatility rank, with fold")
    print("   dummies in both regressions, dropping one fold at a time.\n")
    headers = ["all folds"] + [b[0] for b in blocks]
    print(f"   {'family':<18} " + "".join(f"{lbl:>16}" for lbl in headers))
    loo_rows = []
    for family in families:
        frame = frame_for(online[family], returns, realised, blocks)
        line = f"   {family:<18}"
        for drop in [None] + list(range(len(blocks))):
            keep = frame if drop is None else frame[frame["fold"] != drop]
            dummies = pd.get_dummies(keep["fold"], prefix="d", drop_first=True).astype(float)
            y = keep["fwd"].to_numpy()
            x0 = pd.concat([keep[["vol_rank"]], dummies], axis=1)
            x1 = pd.concat([keep[["vol_rank"]], dummies, keep[["state"]]], axis=1)
            m0 = sm.OLS(y, sm.add_constant(x0.to_numpy())).fit()
            m1 = sm.OLS(y, sm.add_constant(x1.to_numpy())).fit(
                cov_type="HAC", cov_kwds={"maxlags": HORIZON}
            )
            inc = float(m1.rsquared - m0.rsquared)
            t = float(m1.tvalues[-1])
            loo_rows.append({"family": family,
                             "dropped": "none" if drop is None else f"fold {drop}",
                             "incremental": inc, "t": t, "n": int(len(keep))})
            line += f"{inc:>9.3%}({t:>+5.2f})".rjust(16)
        print(line)
    pd.DataFrame(loo_rows).to_csv(OUT / "leave_one_out_foldfe.csv", index=False)

    # ------------------------------------------------------------------ 6
    print("\n" + RULE)
    print("6.  A LATENT DEFECT: external_validation ON A CONSTANT STATE")
    print(RULE)
    constant = pd.Series(1.0, index=oos[:1279], name="state")
    ref = pd.Series(0.0, index=constant.index, name="ref")
    out = reliability.external_validation(constant, ref)
    print("\n   external_validation(constant state, all-zero reference) returns:")
    print(f"   {out}")
    print("\n   It returns a number, not NaN. `weak = state == state.min()` is True")
    print("   everywhere when the state never moves, so the classifier is scored as if")
    print("   it called every single day a recession. No published figure is affected —")
    print("   every published number is full-sample, where the state does move — but a")
    print("   per-fold table reads 0.0% balanced accuracy and that is an artefact of the")
    print("   function, not a measurement of the model.")
    print("\n" + RULE)


if __name__ == "__main__":
    main()

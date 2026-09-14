"""Evaluate the five declared walk-forward folds on states that already exist.

Nothing is refitted and nothing is written into the repository. The five folds
returned by ``regime_lab.models.protocol.walk_forward`` are used only to
partition the out-of-sample state series that ``run_phase2.py`` already
produced, and every metric is the same function ``scripts/run_evaluation.py``
calls, applied to each slice.

What this can and cannot be
---------------------------
The executed protocol is one contiguous out-of-sample block with 49 semi-annual
refits. Slicing it by the fold edges gives the **per-fold dispersion of the
protocol that was actually run**. It is *not* a strict five-fold walk-forward,
in which fold *k*'s model would be frozen at ``edges[k]`` and held for five
years. Inside fold *k* here, the model in force at date *t* was refitted at the
most recent semi-annual date before *t*, so it has seen more (never less) data
than a frozen fold model would. Both are causal; they are different estimators
and this file measures the first one.

Decisions fixed before any output was read
------------------------------------------
1. A fold is self-contained: states, returns, the volatility rank and the
   forward windows are all computed *inside* the fold window, so no forward
   window crosses a fold boundary and no fold borrows another fold's ranks.
2. A sensitivity is reported beside it with the volatility rank computed once
   over the whole out-of-sample block, because that is the regressor the
   published table uses.
3. The volatility-quantile placebo is built once on the whole out-of-sample
   return series and then sliced, exactly as ``run_evaluation.py`` builds it;
   rebuilding it inside each fold would blank the first 252 sessions of every
   fold through its own expanding median.
4. The bootstrap draw count is 800, the value ``run_evaluation.py`` uses.
5. The headline horizon is 21 sessions, the horizon of the published table.

One test printed here is WITHDRAWN
----------------------------------
The fold-interaction Wald test below is not identified for families A and A'.
Two of the five folds contain a single state for A' sparse jump and one for A
jump, so the design matrix loses rank and the p-value it prints is meaningless.
It is superseded by ``scripts/run_fold_heterogeneity.py``, which checks the rank
first, drops the unidentified folds, and reports Cochran's Q over the folds that
remain. Read the p-values from that script, not from this one. Everything else
here — the layer-1 and layer-2 per-fold tables — stands.

Per-fold external validation is mostly NaN, and that is correct
---------------------------------------------------------------
NBER declares a recession in only **two of these five folds**: fold 1, which
holds 2007-2009, and fold 3, which holds 2020. Against a one-class reference
there is no balanced accuracy to compute and kappa's denominator vanishes, so
``reliability.external_validation`` now declines those cells instead of scoring
them. The honest per-fold NBER table has two rows.

An earlier run of this script printed numbers there. It showed 0.0% for the three
folds whose state never moves, and — worse, because it looks like a result — it
showed 75.6% and 98.0% for A jump in folds 0 and 4 and 16.8% for C gradient boost
in fold 4. Those were *specificities*, not balanced accuracies, the only class
present being the absence of recession. Six misleading cells in all, three low and
three high, from one defect with opposite signs. Fixed in
``regime_lab/evaluation/reliability.py``, covered by ``tests/test_reliability.py``.
No published figure was affected: every published number is full-sample, where
both series move, and those are bit-identical after the fix.

Usage:
    .venv/bin/python scripts/run_folds.py
"""

from __future__ import annotations

import json
import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

from regime_lab.config import CACHE
from regime_lab.data import store
from regime_lab.evaluation import predictive, reliability
from regime_lab.evaluation.predictive import (
    forward_return,
    forward_volatility,
    volatility_quantile_placebo,
)
from regime_lab.models.mapping import apply_overlay, state_to_position, summary
from regime_lab.models.protocol import walk_forward
from regime_lab.strategies.book import base_book, panels

warnings.filterwarnings("ignore")

RULE = "=" * 96
DRAWS = 800
HORIZON = 21
OUT = CACHE


def load() -> dict[str, object]:
    """Rebuild exactly the objects ``run_evaluation.py`` works with."""
    online = pd.read_parquet(CACHE / "states.parquet")
    offline = pd.read_parquet(CACHE / "states_offline.parquet")

    price_panel, _ = panels()
    book = base_book()
    oos = online.dropna(how="all").index
    returns = book.reindex(oos).dropna()
    oos = returns.index

    equity = price_panel["eq_us_large"].reindex(oos)
    realised = (equity.pct_change().rolling(21).std() * (252**0.5)).reindex(oos)
    drawdown_ref = reliability.drawdown_reference(equity).reindex(oos)

    nber = store.read("references", "ref_nber").set_index("period")["value"]
    nber_ref = pd.Series(
        nber.reindex(oos.to_period("M").to_timestamp()).to_numpy(), index=oos, name="ref_nber"
    )

    online = online.reindex(oos).copy()
    offline = offline.reindex(oos).copy()
    placebo = volatility_quantile_placebo(returns).reindex(oos)
    online["·  vol placebo"] = placebo
    offline["·  vol placebo"] = placebo

    # The fold edges come from the same call run_phase2.py makes, on the same
    # index, so the boundaries are the declared ones and not a reconstruction.
    features = pd.read_parquet(CACHE / "features.parquet").dropna()
    aligned = base_book().reindex(features.index).dropna()
    folds = walk_forward(features.loc[aligned.index].index, min_train_years=10, folds=5)

    return {
        "online": online,
        "offline": offline,
        "returns": returns,
        "realised": realised,
        "nber": nber_ref,
        "drawdown": drawdown_ref,
        "folds": folds,
        "oos": oos,
    }


def fold_windows(folds, oos: pd.DatetimeIndex) -> list[tuple[str, pd.DatetimeIndex]]:
    """Half-open [eval_start, eval_end) blocks, the last one closed."""
    blocks = []
    for k, f in enumerate(folds):
        last = k == len(folds) - 1
        mask = (oos >= f.eval_start) & (oos <= f.eval_end if last else oos < f.eval_end)
        blocks.append((f"fold {k}", oos[mask]))
    return blocks


def layer1_row(online: pd.Series, offline: pd.Series, nber, drawdown) -> dict[str, float]:
    persistence = reliability.persistence_vs_chance(online)
    gap = reliability.hindsight_gap(online, offline)
    nber_score = reliability.external_validation(online, nber)
    dd_score = reliability.external_validation(online, drawdown)
    return {
        "n": int(online.dropna().shape[0]),
        "mean_run": persistence["observed"],
        "persist_ratio": persistence["ratio"],
        "hindsight_changed": gap["changed_share"],
        "hindsight_kappa": gap["kappa"],
        "nber_bal_acc": nber_score["balanced_accuracy"],
        "nber_kappa": nber_score["kappa"],
        "nber_base_rate": nber_score.get("base_rate", np.nan),
        "dd_bal_acc": dd_score["balanced_accuracy"],
        "dd_kappa": dd_score["kappa"],
    }


def layer2_row(states: pd.Series, returns: pd.Series, realised: pd.Series) -> dict[str, float]:
    split = predictive.variance_versus_mean(states, returns, horizon=HORIZON)
    mean_test = predictive.mean_difference_test(states, returns, horizon=HORIZON, draws=DRAWS)
    vol_test = predictive.volatility_difference_test(
        states, returns, horizon=HORIZON, draws=DRAWS
    )
    inc_ret = predictive.incremental_information(
        states, returns, realised, horizon=HORIZON, target="return"
    )
    inc_vol = predictive.incremental_information(
        states, returns, realised, horizon=HORIZON, target="volatility"
    )
    return {
        "mean_spread": split["mean_spread"],
        "vol_spread": split["vol_spread"],
        "aligned": split["aligned"],
        "mean_diff": mean_test["difference"],
        "mean_t": mean_test["t_hac"],
        "mean_mde": mean_test.get("mde", np.nan),
        "vol_diff": vol_test["difference"],
        "vol_t": vol_test["t_hac"],
        "vol_mde": vol_test["mde"],
        "vol_detected": vol_test.get("detected", np.nan),
        "inc_ret": inc_ret["incremental"],
        "inc_ret_t": inc_ret.get("t_state", np.nan),
        "inc_vol": inc_vol["incremental"],
        "inc_vol_t": inc_vol.get("t_state", np.nan),
        "n_reg": inc_vol.get("n", np.nan),
    }


def interaction_test(
    states: pd.Series,
    returns: pd.Series,
    realised: pd.Series,
    blocks: list[tuple[str, pd.DatetimeIndex]],
) -> dict[str, float]:
    """Is the state's effect on forward volatility the same in all five folds?

    Pooled regression of forward volatility on the volatility rank, the state,
    fold dummies and state x fold interactions, HAC lag 21. The Wald test on the
    four interaction terms is the heterogeneity test the five-fold rule is
    implicitly asking for.
    """
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
    frame = frame.dropna()

    design = pd.DataFrame({"const": 1.0, "vol_rank": frame["vol_rank"], "state": frame["state"]})
    names = ["const", "vol_rank", "state"]
    for k in range(1, len(blocks)):
        d = (frame["fold"] == k).astype(float)
        design[f"d{k}"] = d
        design[f"state_x_d{k}"] = d * frame["state"]
        names += [f"d{k}", f"state_x_d{k}"]

    model = sm.OLS(frame["fwd"].to_numpy(), design[names].to_numpy()).fit(
        cov_type="HAC", cov_kwds={"maxlags": HORIZON}
    )
    positions = [names.index(f"state_x_d{k}") for k in range(1, len(blocks))]
    restriction = np.zeros((len(positions), len(names)))
    for r, p in enumerate(positions):
        restriction[r, p] = 1.0
    wald = model.f_test(restriction)
    return {
        "wald_F": float(np.asarray(wald.fvalue).ravel()[0]),
        "wald_p": float(np.asarray(wald.pvalue).ravel()[0]),
        "df_num": len(positions),
        "n": int(len(frame)),
        "base_state_coef": float(model.params[2]),
        "interactions": {
            f"fold {k}": float(model.params[names.index(f"state_x_d{k}")])
            for k in range(1, len(blocks))
        },
    }


def leave_one_out(
    states: pd.Series,
    returns: pd.Series,
    realised: pd.Series,
    blocks: list[tuple[str, pd.DatetimeIndex]],
    *,
    target: str,
) -> pd.DataFrame:
    """Full-sample incremental information with each fold removed in turn.

    Ranks and forward windows are built on the full out-of-sample block, as in
    the published table; only the rows belonging to the dropped fold are removed
    before the regression. If the published result is carried by one fold, that
    fold is the one whose removal collapses it.
    """
    dependent = (
        forward_return(returns, HORIZON)
        if target == "return"
        else forward_volatility(returns, HORIZON)
    )
    frame = pd.concat(
        {"state": states, "vol_rank": realised.rank(pct=True), "fwd": dependent}, axis=1
    ).dropna()

    rows = []
    for label, idx in [("all folds", pd.DatetimeIndex([]))] + list(blocks):
        keep = frame.drop(index=frame.index.intersection(idx))
        y = keep["fwd"].to_numpy()
        vol_only = sm.OLS(y, sm.add_constant(keep[["vol_rank"]].to_numpy())).fit()
        both = sm.OLS(y, sm.add_constant(keep[["vol_rank", "state"]].to_numpy())).fit(
            cov_type="HAC", cov_kwds={"maxlags": HORIZON}
        )
        rows.append(
            {
                "dropped": label,
                "n": int(len(keep)),
                "incremental": float(both.rsquared - vol_only.rsquared),
                "t_state": float(both.tvalues[2]),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    data = load()
    online: pd.DataFrame = data["online"]
    offline: pd.DataFrame = data["offline"]
    returns: pd.Series = data["returns"]
    realised: pd.Series = data["realised"]
    oos: pd.DatetimeIndex = data["oos"]
    folds = data["folds"]
    blocks = fold_windows(folds, oos)

    print(RULE)
    print("FIVE DECLARED FOLDS, EVALUATED ON STATES THAT ALREADY EXIST — nothing refitted")
    print(RULE)
    print(
        f"\n   out-of-sample block: {oos.min():%Y-%m-%d} to "
        f"{oos.max():%Y-%m-%d}, {len(oos):,} days"
    )
    for f, (label, idx) in zip(folds, blocks, strict=True):
        print(
            f"   {label}: {idx.min():%Y-%m-%d} to {idx.max():%Y-%m-%d}  "
            f"{len(idx):>5,} days   ({f})"
        )

    families = list(online.columns)

    # ---------------------------------------------------------------- layer 1
    print("\n" + RULE)
    print("\nLAYER 1  reliability, per fold\n")
    layer1: list[dict] = []
    for family in families:
        for label, idx in blocks:
            row = layer1_row(
                online[family].reindex(idx),
                offline[family].reindex(idx),
                data["nber"].reindex(idx),
                data["drawdown"].reindex(idx),
            )
            row |= {"family": family, "fold": label}
            layer1.append(row)
        row = layer1_row(online[family], offline[family], data["nber"], data["drawdown"])
        row |= {"family": family, "fold": "ALL"}
        layer1.append(row)
    l1 = pd.DataFrame(layer1).set_index(["family", "fold"])

    header = (
        f"   {'family':<18} {'fold':<8} {'n':>6} {'mean run':>9} {'vs chance':>10} "
        f"{'hindsight':>10} {'k on/off':>9} {'NBER acc':>9} {'NBER k':>7} {'DD acc':>7} {'DD k':>6}"
    )
    print(header)
    for family in families:
        for fold in [b[0] for b in blocks] + ["ALL"]:
            r = l1.loc[(family, fold)]
            print(
                f"   {family:<18} {fold:<8} {r['n']:>6,.0f} {r['mean_run']:>8.0f}d "
                f"{r['persist_ratio']:>10.1f}x {r['hindsight_changed']:>9.1%} "
                f"{r['hindsight_kappa']:>9.2f} {r['nber_bal_acc']:>8.1%} {r['nber_kappa']:>7.2f} "
                f"{r['dd_bal_acc']:>6.1%} {r['dd_kappa']:>6.2f}"
            )
        print()

    # ---------------------------------------------------------------- layer 2
    print(RULE)
    print(
        f"\nLAYER 2  conditional moments, per fold, horizon {HORIZON}, "
        f"ranks computed INSIDE the fold\n"
    )
    layer2: list[dict] = []
    for family in families:
        for label, idx in blocks:
            row = layer2_row(
                online[family].reindex(idx), returns.reindex(idx), realised.reindex(idx)
            )
            row |= {"family": family, "fold": label}
            layer2.append(row)
        row = layer2_row(online[family], returns, realised)
        row |= {"family": family, "fold": "ALL"}
        layer2.append(row)
    l2 = pd.DataFrame(layer2).set_index(["family", "fold"])

    print(
        f"   {'family':<18} {'fold':<8} {'mean sprd':>10} {'vol sprd':>9} {'mean t':>7} "
        f"{'vol diff':>9} {'vol t':>7} {'vol MDE':>8} {'det':>4} "
        f"{'incR2 ret':>10} {'t':>6} {'incR2 vol':>10} {'t':>6}"
    )
    for family in families:
        for fold in [b[0] for b in blocks] + ["ALL"]:
            r = l2.loc[(family, fold)]
            print(
                f"   {family:<18} {fold:<8} {r['mean_spread']:>+9.2%} {r['vol_spread']:>+8.2%} "
                f"{r['mean_t']:>7.2f} {r['vol_diff']:>+8.2%} {r['vol_t']:>7.2f} "
                f"{r['vol_mde']:>7.2%} {'yes' if r['vol_detected'] else 'no':>4} "
                f"{r['inc_ret']:>9.3%} {r['inc_ret_t']:>6.2f} "
                f"{r['inc_vol']:>9.3%} {r['inc_vol_t']:>6.2f}"
            )
        print()

    # sensitivity: global ranks, fold slices
    print(RULE)
    print("\nSENSITIVITY  same table, volatility rank computed ONCE over the whole block\n")
    global_rank = realised.rank(pct=True)
    sens: list[dict] = []
    print(f"   {'family':<18} {'fold':<8} {'incR2 vol':>10} {'t':>7} {'incR2 ret':>10} {'t':>7}")
    for family in families:
        for label, idx in blocks:
            sub_states = online[family].reindex(idx)
            sub_returns = returns.reindex(idx)
            inc_v = predictive.incremental_information(
                sub_states, sub_returns, global_rank.reindex(idx), horizon=HORIZON,
                target="volatility",
            )
            inc_r = predictive.incremental_information(
                sub_states, sub_returns, global_rank.reindex(idx), horizon=HORIZON,
                target="return",
            )
            sens.append(
                {
                    "family": family,
                    "fold": label,
                    "inc_vol": inc_v["incremental"],
                    "inc_vol_t": inc_v.get("t_state", np.nan),
                    "inc_ret": inc_r["incremental"],
                    "inc_ret_t": inc_r.get("t_state", np.nan),
                }
            )
            print(
                f"   {family:<18} {label:<8} {inc_v['incremental']:>9.3%} "
                f"{inc_v.get('t_state', np.nan):>7.2f} {inc_r['incremental']:>9.3%} "
                f"{inc_r.get('t_state', np.nan):>7.2f}"
            )
        print()

    # ------------------------------------------------- the central result
    print(RULE)
    print("\nTHE CENTRAL RESULT, FOLD BY FOLD — leave-one-fold-out on the full block\n")
    loo: dict[str, pd.DataFrame] = {}
    for family in families:
        for target in ("volatility", "return"):
            table = leave_one_out(
                online[family], returns, realised, blocks, target=target
            )
            loo[f"{family}|{target}"] = table
    for family in families:
        print(f"   {family}")
        for target in ("volatility", "return"):
            table = loo[f"{family}|{target}"]
            print(f"      forward {target:<11}", end="")
            for _, r in table.iterrows():
                print(
                    f"  {r['dropped']}: {r['incremental']:>+7.3%} "
                    f"(t {r['t_state']:>+5.2f})",
                    end="",
                )
            print()
        print()

    # ------------------------------------------------- heterogeneity
    print(RULE)
    print("\nHETEROGENEITY  state x fold interactions on forward volatility, HAC lag 21\n")
    hetero = {}
    print(f"   {'family':<18} {'Wald F':>8} {'p':>8} {'base coef':>10}   interactions by fold")
    for family in families:
        h = interaction_test(online[family], returns, realised, blocks)
        hetero[family] = h
        inter = "  ".join(f"{k}: {v:>+7.4f}" for k, v in h["interactions"].items())
        print(
            f"   {family:<18} {h['wald_F']:>8.2f} {h['wald_p']:>8.4f} "
            f"{h['base_state_coef']:>+10.4f}   {inter}"
        )

    # ------------------------------------------------- stopping rule 1
    print("\n" + RULE)
    print("\nSTOPPING RULE 1, computed for the first time — 'at least three folds out of five'\n")
    print("   The charter's stop 1 is written at strategy level (T1), which was never")
    print("   implemented. This is its layer-2 analogue on the same states: per fold,")
    print("   does the family separate forward volatility by more than its own MDE, and")
    print("   does it add information beyond the volatility quantile at |t| > 1.96?\n")
    print(
        f"   {'family':<18} {'folds vol > MDE':>16} "
        f"{'folds |t inc vol| > 1.96':>26} {'folds inc vol > 0':>19}"
    )
    counts = {}
    for family in families:
        sub = l2.loc[family].drop(index="ALL")
        n_mde = int(sub["vol_detected"].astype(bool).sum())
        n_t = int((sub["inc_vol_t"].abs() > 1.96).sum())
        n_pos = int((sub["inc_vol"] > 0).sum())
        counts[family] = {"vol_gt_mde": n_mde, "abs_t_gt_196": n_t, "inc_positive": n_pos}
        print(f"   {family:<18} {n_mde:>13} /5 {n_t:>23} /5 {n_pos:>16} /5")

    # ------------------------------------------------- strategy level, descriptive
    print("\n" + RULE)
    print("\nDESCRIPTIVE  the frozen on/off rule, per fold (excess returns, no costs)\n")
    print(f"   {'family':<18} " + "".join(f"{b[0]:>12}" for b in blocks) + f"{'ALL':>12}")
    base_row = "   " + f"{'base 60/40':<18}"
    for _, idx in blocks:
        base_row += f"{summary(returns.reindex(idx))['sharpe']:>12.2f}"
    base_row += f"{summary(returns)['sharpe']:>12.2f}"
    print(base_row)
    strat_rows = {}
    for family in families:
        position = state_to_position(online[family]).reindex(oos).fillna(0.0)
        gated = apply_overlay(returns, position)
        row = "   " + f"{family:<18}"
        vals = []
        for _, idx in blocks:
            s = summary(gated.reindex(idx))["sharpe"]
            vals.append(s)
            row += f"{s:>12.2f}"
        allv = summary(gated)["sharpe"]
        row += f"{allv:>12.2f}"
        strat_rows[family] = vals + [allv]
        print(row)

    print("\n" + RULE)

    # ------------------------------------------------- persist
    l1.to_csv(OUT / "layer1_by_fold.csv")
    l2.to_csv(OUT / "layer2_by_fold.csv")
    pd.DataFrame(sens).to_csv(OUT / "layer2_global_rank_sensitivity.csv", index=False)
    pd.concat({k: v for k, v in loo.items()}).to_csv(OUT / "leave_one_fold_out.csv")
    with open(OUT / "heterogeneity.json", "w") as fh:
        json.dump({"hetero": hetero, "stop_rule_counts": counts}, fh, indent=2)
    print(f"\n   written: {OUT}/layer1_by_fold.csv, layer2_by_fold.csv,")
    print("            layer2_global_rank_sensitivity.csv, "
          "leave_one_fold_out.csv, heterogeneity.json")


if __name__ == "__main__":
    sys.exit(main())

"""Exploration: can the sparse jump model (family A') be made to switch more often?

The study's A' changes state 13 times in 6,377 out-of-sample sessions. Two levers
are tried side by side, on one training window, and nothing else is changed:

1. a lower jump penalty, below the floor of the declared calibration grid;
2. faster inputs: a subset of the existing features built on short windows.

This is a descriptive look at the classifier and nothing more. No position is
taken, no return or Sharpe is computed, nothing is written to the trials log or
to the cache. None of these settings replaces the frozen study's classifier; a
faster variant used in a strategy test would have to be declared beforehand.

Protocol, kept as simple as the existing code allows: one fit on 1992 to the end
of 2006 through the study's own wrapper (``JumpRegimes``, max_features 10, states
ranked by the base book's volatility on the training window, state 0 = stress),
then ``predict_online`` run forward over the whole sample, so each 2007-2026
label uses only rows up to its own date. The study itself refits every six
months; this does not.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from regime_lab.config import CACHE
from regime_lab.data import store
from regime_lab.evaluation.predictive import volatility_quantile_placebo
from regime_lab.evaluation.reliability import agreement, external_validation
from regime_lab.models.jump import JumpRegimes
from regime_lab.strategies.book import base_book, panels

warnings.filterwarnings("ignore")
RULE = "=" * 78

TRAIN_END = "2006-12-31"
OOS_START = "2007-01-01"
MAX_FEATURES = 10.0

#: Short windows (5-21 sessions) or instantaneous, plus one 63-day credit change
#: so the set is not volatility alone.
FAST = [
    "vol_rv_5",
    "vol_rv_21",
    "vol_term",
    "vol_vix",
    "vol_vrp",
    "mom_eq_21",
    "xs_dispersion_ind",
    "cre_baa_chg63",
]

#: 10 is what the study's in-training calibration chose for the window ending in
#: 2006 (trials log, train_end 2006-03-30); 1 is the grid floor; 0.3 and 0.1 are
#: below the declared grid.
PENALTIES = (10.0, 1.0, 0.3, 0.1)


def eta_squared(labels: pd.Series, values: pd.Series) -> float:
    """Share of the variance of ``values`` explained by the partition."""
    both = pd.concat([labels.rename("g"), values.rename("y")], axis=1).dropna()
    y = both["y"].to_numpy(dtype=float)
    total = float(((y - y.mean()) ** 2).sum())
    groups = both.groupby("g")["y"]
    between = float((groups.size() * (groups.mean() - y.mean()) ** 2).sum())
    return between / total if total > 0 else float("nan")


def vol_filter(realised: pd.Series, *, quantile: float = 0.8) -> pd.Series:
    """Calm (1) unless 21d RV sits above its own expanding ``quantile``; causal.

    At 0.8 the stress share is about a fifth, the share the fitted models give,
    so eta2 and kappa can be compared like for like. The study's median placebo
    has half the sessions in stress, which caps its eta2 lower on a skewed RV.
    """
    threshold = realised.expanding(min_periods=252).quantile(quantile)
    return (realised <= threshold).astype(float).where(threshold.notna()).rename("vol80")


def eta2_ceiling(realised: pd.Series) -> tuple[float, float]:
    """Best eta2 any single threshold on RV reaches, and where (hindsight)."""
    clean = realised.dropna()
    best = max(
        (eta_squared((clean > clean.quantile(q)).astype(float), clean), q)
        for q in np.arange(0.50, 0.99, 0.01)
    )
    return best


def stress_episodes(states: pd.Series) -> pd.DataFrame:
    """Start, end and length of every uninterrupted run in state 0."""
    clean = states.dropna()
    run = clean.ne(clean.shift()).cumsum()
    frame = pd.DataFrame({"state": clean, "run": run, "date": clean.index})
    runs = frame.groupby("run").agg(
        state=("state", "first"), start=("date", "first"), end=("date", "last"),
        length=("state", "size"),
    )
    return runs.loc[runs["state"] == 0.0, ["start", "end", "length"]].reset_index(drop=True)


def describe(
    name: str, states: pd.Series, realised: pd.Series, nber: pd.Series, matched: pd.Series
) -> dict:
    """Print the descriptive readings for one label series and return them."""
    s = states.loc[OOS_START:].dropna()
    years = len(s) / 252.0
    transitions = int((s.diff().abs() > 0).sum())
    episodes = stress_episodes(s)
    score = external_validation(s, nber.reindex(s.index))
    recession = nber.reindex(s.index) == 1.0
    stress = s == 0.0
    row = {
        "fit": name,
        "sessions": len(s),
        "transitions": transitions,
        "per_year": transitions / years,
        "stress_share": float((s == 0.0).mean()),
        "episodes": len(episodes),
        "short_episodes": int((episodes["length"] < 10).sum()),
        "median_length": float(episodes["length"].median()) if len(episodes) else np.nan,
        "eta2_rv21": eta_squared(s, realised),
        "kappa_vol80": agreement(s, matched.reindex(s.index))["kappa"],
        "nber_ba": score["balanced_accuracy"],
        "nber_kappa": score["kappa"],
        "nber_sens": float(stress[recession].mean()),
        "nber_spec": float((~stress[~recession]).mean()),
    }
    print(
        f"\n   {name}\n"
        f"      transitions {transitions} in {len(s):,} sessions = {row['per_year']:.2f}/yr, "
        f"stress share {row['stress_share']:.1%}\n"
        f"      eta2 vs 21d RV of eq_us_large {row['eta2_rv21']:.3f}, "
        f"kappa vs matched vol filter {row['kappa_vol80']:.2f}\n"
        f"      NBER balanced accuracy {row['nber_ba']:.1%} (kappa {row['nber_kappa']:.2f}; "
        f"sensitivity {row['nber_sens']:.1%}, specificity {row['nber_spec']:.1%})\n"
        f"      {len(episodes)} stress episodes ({row['short_episodes']} shorter than 10 "
        f"sessions), median length {row['median_length']:.0f} sessions"
    )
    if len(episodes) <= 12:
        for ep in episodes.itertuples():
            print(f"         {ep.start:%Y-%m-%d} -> {ep.end:%Y-%m-%d}  ({ep.length} sessions)")
    else:
        per_year = episodes["start"].dt.year.value_counts().sort_index()
        print("         episodes starting per year: "
              + " ".join(f"{y}:{n}" for y, n in per_year.items()))
        long = episodes.nlargest(5, "length").sort_values("start")
        print("         five longest: " + ", ".join(
            f"{ep.start:%Y-%m}..{ep.end:%Y-%m} ({ep.length})" for ep in long.itertuples()
        ))
    return row


def main() -> None:
    features = pd.read_parquet(CACHE / "features.parquet").dropna()
    book = base_book().reindex(features.index).dropna()
    features = features.loc[book.index]
    missing = [c for c in FAST if c not in features]
    if missing:
        raise KeyError(f"fast features absent from features.parquet: {missing}")

    price_panel, _ = panels()
    equity = price_panel["eq_us_large"]
    equity_returns = equity.pct_change()
    realised = (equity_returns.rolling(21).std() * np.sqrt(252)).rename("rv21")
    nber_raw = store.read("references", "ref_nber").set_index("period")["value"]
    oos_index = features.loc[OOS_START:].index
    nber = pd.Series(
        nber_raw.reindex(oos_index.to_period("M").to_timestamp()).to_numpy(),
        index=oos_index,
        name="ref_nber",
    )

    train_index = features.loc[:TRAIN_END].index
    print(RULE)
    print("SPARSE JUMP MODEL: SWITCHING SPEED  (descriptive, no returns, no trials log)")
    print(RULE)
    print(f"   train {train_index.min():%Y-%m-%d} to {train_index.max():%Y-%m-%d} "
          f"({len(train_index):,} sessions), label {OOS_START} to "
          f"{features.index.max():%Y-%m-%d} online, one fit, no refit")
    print(f"   full set: all {features.shape[1]} features, max_features {MAX_FEATURES:g}")
    print(f"   fast set ({len(FAST)}): {', '.join(FAST)}")
    print("   jumpmodels divides the penalty by sqrt(n_features): effective penalty is "
          f"lambda/{np.sqrt(features.shape[1]):.2f} (full), lambda/{np.sqrt(len(FAST)):.2f} (fast)")

    matched = vol_filter(realised).reindex(features.index)
    ceiling, where = eta2_ceiling(realised.reindex(oos_index))
    print(f"   eta2 ceiling of any single threshold on 21d RV, 2007-2026: {ceiling:.3f} "
          f"(split at the {where:.0%} quantile, hindsight)")

    rows = []
    print("\n" + RULE + "\nREFERENCES\n" + RULE)
    study = pd.read_parquet(CACHE / "states.parquet")["A' sparse jump"]
    rows.append(
        describe("study A' (states.parquet, 6-month refits)", study, realised, nber, matched)
    )
    placebo = volatility_quantile_placebo(equity_returns).reindex(features.index)
    for name, series in (
        ("vol placebo (21d RV < expanding median)", placebo),
        ("vol filter (21d RV > expanding 80th pct)", matched),
    ):
        rows.append(describe(name, series, realised, nber, matched))

    print("\n" + RULE + "\nFITS\n" + RULE)
    for set_name, columns in (("full", list(features.columns)), ("fast", FAST)):
        for penalty in PENALTIES:
            model = JumpRegimes(jump_penalty=penalty, max_features=MAX_FEATURES)
            train = features.loc[train_index, columns]
            model.fit(train, book.loc[train_index])
            states = model.predict_online(features[columns])
            name = f"{set_name} lambda={penalty:g}"
            row = describe(name, states, realised, nber, matched)
            row["in_train_per_year"] = float(
                (states.loc[train_index].diff().abs() > 0).sum() / (len(train_index) / 252.0)
            )
            weights = model.model_.w.sort_values(ascending=False)
            kept = weights[weights > 1e-6]
            print(f"      in-training {row['in_train_per_year']:.2f}/yr; {len(kept)} features "
                  "weighted, top: " + ", ".join(f"{k} {v:.2f}" for k, v in kept.head(5).items()))
            rows.append(row)

    table = pd.DataFrame(rows).set_index("fit")
    print("\n" + RULE + "\nSUMMARY  2007-2026, out of sample\n" + RULE)
    cols = [
        "per_year", "stress_share", "episodes", "short_episodes",
        "eta2_rv21", "kappa_vol80", "nber_ba",
    ]
    with pd.option_context("display.width", 120, "display.float_format", "{:.3f}".format):
        print(table[cols].to_string())
    print("\n   descriptive only: none of these settings replaces the frozen classifier")


if __name__ == "__main__":
    main()

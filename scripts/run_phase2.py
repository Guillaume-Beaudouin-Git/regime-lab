"""Phase 2: every family under one protocol, with lambda calibrated in-training.

The families differ in method and in nothing else: same features, same refit
dates, same rule turning a state into a position, same trials log. The jump
penalty is calibrated inside each training window rather than fixed by hand,
because it is an optimised parameter and hiding that would understate the
search.
"""

from __future__ import annotations

import warnings

import pandas as pd

from regime_lab.analysis import trials
from regime_lab.config import CACHE
from regime_lab.models.base import run_expanding
from regime_lab.models.calibrate import LAMBDA_GRID, choose_jump_penalty
from regime_lab.models.hmm import FilteredHMM
from regime_lab.models.jump import JumpRegimes
from regime_lab.models.mapping import apply_overlay, state_to_position, summary, turnover
from regime_lab.models.protocol import walk_forward
from regime_lab.models.supervised import ForwardVolModel, har_features
from regime_lab.strategies.book import base_book

warnings.filterwarnings("ignore")
RULE = "=" * 78

#: Lambda is re-calibrated every fourth refit, i.e. every two years. A desk
#: does not re-tune a structural parameter twice a year, and re-tuning at every
#: refit multiplies the search by six for no realism.
CALIBRATE_EVERY = 4


def jump_factory(*, sparse: bool):
    """Factory that calibrates lambda on the training window, then caches it."""
    state = {"penalty": None, "last": -999}

    def make(train, train_returns, refit_index):
        if refit_index - state["last"] >= CALIBRATE_EVERY:
            calibration = choose_jump_penalty(
                lambda p: JumpRegimes(jump_penalty=p, max_features=10.0 if sparse else None),
                train,
                train_returns,
            )
            state["penalty"] = calibration.jump_penalty
            state["last"] = refit_index
            for penalty, sharpe, rate in calibration.candidates:
                trials.log(
                    "sparse_jump" if sparse else "jump",
                    {"lambda": penalty, "train_end": str(train.index.max().date())},
                    {"sharpe": sharpe, "switches_per_year": rate, "stage": "calibration"},
                )
        return JumpRegimes(jump_penalty=state["penalty"], max_features=10.0 if sparse else None)

    return make


def main() -> None:
    features = pd.read_parquet(CACHE / "features.parquet").dropna()
    book = base_book().reindex(features.index).dropna()
    features = features.loc[book.index]
    har = har_features(book).reindex(features.index)

    folds = walk_forward(features.index, min_train_years=10, folds=5)
    first_refit = folds[0].eval_start

    print(RULE)
    print(
        f"PHASE 2  {len(features.columns)} features, {len(features):,} days "
        f"({features.index.min():%Y-%m} to {features.index.max():%Y-%m})"
    )
    print(RULE)
    print(f"\n   {len(folds)} expanding folds, refits every 6 months from {first_refit:%Y-%m}")
    print(f"   lambda calibrated in-training every {CALIBRATE_EVERY} refits over {LAMBDA_GRID}")
    print("   position rule fixed in advance: hold in the stronger state, flat otherwise\n")

    families = {
        "A  jump": (features, jump_factory(sparse=False)),
        "A' sparse jump": (features, jump_factory(sparse=True)),
        "B  filtered HMM": (features, lambda t, r, i: FilteredHMM()),
        "C  gradient boost": (features, lambda t, r, i: ForwardVolModel()),
        "C' HAR-RV": (har, lambda t, r, i: ForwardVolModel(linear=True)),
    }

    results: dict[str, pd.Series] = {}
    offline_states: dict[str, pd.Series] = {}
    diagnostics: list[pd.DataFrame] = []
    for name, (design, factory) in families.items():
        states, fitted, diag, offline = run_expanding(
            factory, design.dropna(), book, first_refit=first_refit, months=6
        )
        results[name] = states
        offline_states[name] = offline
        diagnostics.append(diag.assign(family=name))
        oos = states.dropna()
        switches = int((oos.diff().abs() > 0).sum())
        years = len(oos) / 252
        print(
            f"   {name:<18} {len(fitted):>3} refits  {len(oos):,} OOS days  "
            f"{switches / years:>5.1f} switches/yr"
        )

    print("\n" + RULE)
    print("\nOUT OF SAMPLE  base book gated by each model\n")
    print(f"   {'':<18} {'Sharpe':>7} {'vol':>7} {'maxDD':>8} {'exposed':>8} {'turnover':>9}")

    oos_index = book.index[book.index >= first_refit]
    base = book.loc[oos_index]
    stats = summary(base)
    print(
        f"   {'base 60/40':<18} {stats['sharpe']:>7.2f} {stats['vol']:>6.1%} "
        f"{stats['max_drawdown']:>8.1%} {'100.0%':>8} {'-':>9}"
    )

    for name, states in results.items():
        position = state_to_position(states).reindex(oos_index).fillna(0.0)
        gated = apply_overlay(base, position)
        stats = summary(gated)
        trials.log(
            name.split()[0],
            {"family": name, "stage": "oos"},
            {"sharpe": stats["sharpe"], "vol": stats["vol"], "exposure": float(position.mean())},
        )
        print(
            f"   {name:<18} {stats['sharpe']:>7.2f} {stats['vol']:>6.1%} "
            f"{stats['max_drawdown']:>8.1%} {position.mean():>7.1%} "
            f"{turnover(position):>9.4f}"
        )

    pd.DataFrame(results).to_parquet(CACHE / "states.parquet")
    pd.DataFrame(offline_states).to_parquet(CACHE / "states_offline.parquet")
    pd.concat(diagnostics).reset_index().to_parquet(CACHE / "refit_diagnostics.parquet")
    log = trials.summary()
    print(
        f"\n   trials log: {log['n_trials']} evaluations, {log['n_distinct']} distinct "
        f"configurations, Sharpe variance {log['sharpe_var']:.4f}"
    )
    print("   these are the inputs the deflated Sharpe needs, measured not assumed")
    print("\n" + RULE)


if __name__ == "__main__":
    main()

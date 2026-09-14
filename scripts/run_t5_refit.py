"""T5 stage 1 — re-run the frozen expanding protocol, keeping every fitted model.

The published pipeline (`scripts/run_phase2.py`) throws the fitted models away:
`states.parquet` holds one contiguous label series in which each semi-annual
block was written by a different model. Successive partitions therefore never
share a single date, and the adjusted Rand index between refits cannot be read
off the cache. This stage refits on exactly the frozen schedule and stores, for
every refit, that model's labels over the WHOLE common date index, so any pair
of refits can later be compared on a common window.

Two guards:

* the only thing written is `data/cache/t5_refits/`, which is gitignored and
  regenerable — in particular `trials.log` is not called, because it appends to
  `data/trials.parquet` and this stage is a re-derivation of the frozen fits, not
  a new trial;
* the block-wise states are rebuilt with the same context logic as
  `run_expanding` and compared with `data/cache/states.parquet`. If the
  reproduction is not exact, the models fitted here are not the models the study
  published and nothing downstream is admissible.
"""

from __future__ import annotations

import sys
import time
import warnings

import numpy as np
import pandas as pd

from regime_lab.config import CACHE
from regime_lab.models.calibrate import choose_jump_penalty
from regime_lab.models.hmm import FilteredHMM
from regime_lab.models.jump import JumpRegimes
from regime_lab.models.protocol import refit_dates, walk_forward
from regime_lab.models.supervised import ForwardVolModel, har_features
from regime_lab.strategies.book import base_book

warnings.filterwarnings("ignore")

OUT = CACHE / "t5_refits"
OUT.mkdir(parents=True, exist_ok=True)

CALIBRATE_EVERY = 4  # identical to scripts/run_phase2.py
MAX_REFITS = int(sys.argv[1]) if len(sys.argv) > 1 else 0  # 0 = all, >0 = smoke test


def jump_factory(*, sparse: bool):
    """Byte-for-byte the factory of run_phase2, minus the trials-log side effect."""
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
        return JumpRegimes(jump_penalty=state["penalty"], max_features=10.0 if sparse else None)

    return make


def run_family(name: str, design: pd.DataFrame, factory, book: pd.Series, first_refit):
    """Replicate run_expanding, and additionally keep each model's full-sample labels."""
    features = design.dropna()
    returns = book
    schedule = refit_dates(features.index, first=first_refit, months=6)
    if MAX_REFITS:
        schedule = schedule[:MAX_REFITS]

    states = pd.Series(index=features.index, dtype="float64")
    online_cols: dict[str, pd.Series] = {}
    offline_cols: dict[str, pd.Series] = {}
    penalties: dict[str, float] = {}

    for i, refit in enumerate(schedule):
        train = features.loc[features.index < refit]
        if len(train) < 504:
            continue
        t0 = time.time()
        model = factory(train, returns.loc[train.index], i)
        model.fit(train, returns.loc[train.index])

        # Same call, same order as run_expanding, so any hidden state in the
        # library wrapper is exercised identically.
        _ = model.predict_online(train)

        stop = schedule[i + 1] if i + 1 < len(schedule) else features.index.max()
        block = features.loc[(features.index >= refit) & (features.index <= stop)]
        if not block.empty:
            context = pd.concat([train.tail(252), block])
            predicted = model.predict_online(context)
            states.loc[block.index] = np.asarray(predicted)[-len(block) :]

        # The T5 payload: this model's view of the ENTIRE sample, so refit i and
        # refit j can be scored on identical dates.
        key = refit.strftime("%Y-%m-%d")
        online_cols[key] = model.predict_online(features).astype("float64")
        offline_cols[key] = model.predict_offline(features).astype("float64")
        penalties[key] = float(getattr(model, "jump_penalty", np.nan))
        print(
            f"   {name:<18} refit {i:>2} {key}  train={len(train):>5}  "
            f"lambda={penalties[key]}  {time.time() - t0:>5.1f}s",
            flush=True,
        )

    tag = name.replace(" ", "_").replace("'", "p")
    pd.DataFrame(online_cols).to_parquet(OUT / f"labels_online_{tag}.parquet")
    pd.DataFrame(offline_cols).to_parquet(OUT / f"labels_offline_{tag}.parquet")
    pd.Series(penalties, name="lambda").to_frame().to_parquet(OUT / f"lambda_{tag}.parquet")
    return states.rename(name)


def main() -> None:
    features = pd.read_parquet(CACHE / "features.parquet").dropna()
    book = base_book().reindex(features.index).dropna()
    features = features.loc[book.index]
    har = har_features(book).reindex(features.index)

    folds = walk_forward(features.index, min_train_years=10, folds=5)
    first_refit = folds[0].eval_start
    print(f"first refit {first_refit:%Y-%m-%d}, {len(features):,} days, "
          f"{len(features.columns)} features", flush=True)

    families = {
        "A  jump": (features, jump_factory(sparse=False)),
        "A' sparse jump": (features, jump_factory(sparse=True)),
        "B  filtered HMM": (features, lambda t, r, i: FilteredHMM()),
        "C  gradient boost": (features, lambda t, r, i: ForwardVolModel()),
        "C' HAR-RV": (har, lambda t, r, i: ForwardVolModel(linear=True)),
    }

    rebuilt = {}
    for name, (design, factory) in families.items():
        rebuilt[name] = run_family(name, design, factory, book, first_refit)

    frame = pd.DataFrame(rebuilt)
    frame.to_parquet(OUT / "states_rebuilt.parquet")

    published = pd.read_parquet(CACHE / "states.parquet")
    print("\nREPRODUCTION CHECK against data/cache/states.parquet")
    for col in published.columns:
        a = published[col].dropna()
        b = frame[col].reindex(a.index)
        common = b.dropna().index
        exact = bool((published.loc[common, col] == b.loc[common]).all())
        print(f"   {col:<18} {len(common):>5} dates compared  identical={exact}  "
              f"mismatches={int((published.loc[common, col] != b.loc[common]).sum())}")


if __name__ == "__main__":
    main()

"""Walk-forward re-estimation of the reduced sparse jump model on 1926-2026.

The protocol is `docs/PRESPEC_LONGHIST.md` §3. Same method as the programme's A'
(`scripts/run_phase2.py`): `models.jump.JumpRegimes` with ``max_features = 10``, two
states ordered by training volatility (state 0 = stress), expanding window, refits
every six months, online (filtered) prediction between refits, and the jump penalty
chosen on each training window from `models.calibrate.LAMBDA_GRID` every fourth refit
by `models.calibrate.choose_jump_penalty`.

What differs from A', and only this:
- the features: the 30 of `extensions.longhist.HEADLINE_FEATURES`, computable since
  1926 (with ``--indpro``, the declared sensitivity adds the two INDPRO growth rates,
  current vintage);
- the return series that orders the states and scores the calibration: the CRSP
  value-weighted market in excess of the bill (Ken French), where A' used a 60/40
  book that does not exist before 1990;
- the first refit, 1937-01-01, which leaves 8.6 years of complete features to train
  on (features are complete from 1928-05-24) and puts 14 NBER recessions out of
  sample;
- the dynamic programme runs through `extensions.longhist.fast_dp`, bit-identical to
  the reference implementation and much faster (no numpy reduction per session).

The calibration candidates are written to the trials log (family
``longhist_calibration``, or ``longhist_calibration_indpro``), as `run_phase2.py` does
for A', once each: a candidate already logged by an interrupted run is skipped. They
are in-training Sharpe ratios of the gated market, the model's own estimation; no
return of the tested object (UMD) is read here.

Outputs, all in `data/cache/` (gitignored), never `states.parquet`:
    longhist_features[_indpro].parquet   the standardised feature matrix
    longhist_states[_indpro].parquet     online and offline states, one row per session
    longhist_refits[_indpro].parquet     per refit: penalty, training size, feature weights

Usage:
    .venv/bin/python scripts/longhist_fit.py            # the headline model
    .venv/bin/python scripts/longhist_fit.py --indpro   # the declared sensitivity
"""

from __future__ import annotations

import sys
import time
import warnings

import pandas as pd

from regime_lab.analysis import trials
from regime_lab.config import CACHE
from regime_lab.data import store
from regime_lab.extensions import longhist as lh
from regime_lab.models.base import run_expanding
from regime_lab.models.calibrate import LAMBDA_GRID, choose_jump_penalty
from regime_lab.models.jump import JumpRegimes

warnings.filterwarnings("ignore")
RULE = "=" * 78
FIRST_REFIT = pd.Timestamp("1937-01-01")
MONTHS = 6
CALIBRATE_EVERY = 4
MAX_FEATURES = 10.0


def build_features(with_indpro: bool) -> pd.DataFrame:
    raw = lh.raw_features(lh.load_wide("ff3_daily"), lh.load_wide("industry49_daily"),
                          lh.load_wide("size_bm25_daily"), store.read(lh.SOURCE, "fred_monthly"),
                          with_indpro=with_indpro)
    return lh.standardise(raw).dropna()


def logged_hashes(family: str) -> set[str]:
    """Configuration hashes of ``family`` already in the trials log."""
    log = trials.read()
    if log.empty:
        return set()
    return set(log.loc[log["family"] == family, "config_hash"])


def factory(family: str, tag: str, chosen: dict[int, float]):
    """Calibrate the penalty on the training window every fourth refit, then reuse it.

    A candidate whose configuration is already logged under ``family`` is not logged
    again: the fit is deterministic, so a re-run after an interruption re-evaluates
    the same candidates, and duplicating them would count the search twice.
    """
    state = {"penalty": None, "last": -999}
    seen = logged_hashes(family)
    counts = {"logged": 0, "skipped": 0}

    def make(train, train_returns, refit_index):
        if refit_index - state["last"] >= CALIBRATE_EVERY:
            started = time.time()
            calibration = choose_jump_penalty(
                lambda p: JumpRegimes(jump_penalty=p, max_features=MAX_FEATURES),
                train, train_returns)
            state["penalty"] = calibration.jump_penalty
            state["last"] = refit_index
            for penalty, sharpe, rate in calibration.candidates:
                config = {"lambda": penalty, "train_end": str(train.index.max().date()),
                          "features": tag, "model": "longhist reduced SJM"}
                if trials.config_hash(config) in seen:
                    counts["skipped"] += 1
                    continue
                trials.log(family, config,
                           {"sharpe": sharpe, "switches_per_year": rate, "stage": "calibration"})
                counts["logged"] += 1
            print(f"   calibration at refit {refit_index:>3} (train to "
                  f"{train.index.max():%Y-%m-%d}, {len(train):,} rows): lambda "
                  f"{calibration.jump_penalty:g}  [{time.time() - started:.0f} s]", flush=True)
        chosen[refit_index] = state["penalty"]
        return JumpRegimes(jump_penalty=state["penalty"], max_features=MAX_FEATURES)

    make.counts = counts
    return make


def main() -> None:
    with_indpro = "--indpro" in sys.argv
    suffix = "_indpro" if with_indpro else ""
    tag = "headline+indpro" if with_indpro else "headline"
    lh.use_fast_dp()

    features = build_features(with_indpro)
    market = lh.market_returns(lh.load_wide("ff3_daily"))["excess"].reindex(features.index)
    print(RULE)
    print(f"LONGHIST FIT ({tag})  {features.shape[1]} features, {len(features):,} sessions, "
          f"{features.index.min():%Y-%m-%d} -> {features.index.max():%Y-%m-%d}")
    print(f"   first refit {FIRST_REFIT:%Y-%m-%d}, every {MONTHS} months, lambda over "
          f"{LAMBDA_GRID} every {CALIBRATE_EVERY} refits, max_features {MAX_FEATURES:g}")
    print(RULE, flush=True)

    chosen: dict[int, float] = {}
    started = time.time()
    make = factory(f"longhist_calibration{suffix}", tag, chosen)
    online, fitted, diagnostics, offline = run_expanding(
        make, features, market, first_refit=FIRST_REFIT, months=MONTHS)
    print(f"\n   {len(fitted)} refits in {(time.time() - started) / 60:.1f} min; calibration "
          f"candidates logged {make.counts['logged']}, already in the log and skipped "
          f"{make.counts['skipped']}", flush=True)

    rows = []
    for i, (refit, model) in enumerate(fitted.items()):
        weights = pd.Series(model.model_.feat_weights, index=model.features_)
        rows.append({"refit": refit, "penalty": chosen.get(i), "train_rows": int(
            diagnostics.loc[refit, "train_days"]), **{f"w_{k}": float(v) for k, v in
                                                       weights.items()}})
    refits = pd.DataFrame(rows).set_index("refit")
    states = pd.DataFrame({"state": online, "state_offline": offline})

    features.to_parquet(CACHE / f"longhist_features{suffix}.parquet")
    states.to_parquet(CACHE / f"longhist_states{suffix}.parquet")
    refits.to_parquet(CACHE / f"longhist_refits{suffix}.parquet")
    oos = online.dropna()
    print(f"   out of sample {oos.index.min():%Y-%m-%d} -> {oos.index.max():%Y-%m-%d}, "
          f"{len(oos):,} sessions")
    print(f"   written: longhist_features{suffix}, longhist_states{suffix}, "
          f"longhist_refits{suffix} (data/cache)")
    print(RULE)


if __name__ == "__main__":
    main()

"""Two Sigma plan — the power of level A, as disclosed and on the declared object. §1 P4, P5.

`PRESPEC_TWOSIGMA.md` is a DRAFT. Its §1 P4 discloses the resolution level A would
have (hard switch 0.474; tilt 0.221 to 0.294, median 0.241; 0.338 at α = 0.05/6; 0.376
at 0.05/21) and §5 prices the selector (+2.6x/yr, +0.013 Sharpe, kill at 12x/yr). The
scripts that produced those figures were never committed. They measured a TOY library
of five signals (12-1 momentum, 1-month reversal, a 1,008-session long-term reversal,
low volatility, and a "seasonality" that is the 12-month return lagged 12 months), with
a random tilt redrawn every 21 sessions, on the full sample, gross, in raw returns.
§6 A-1 is not that object. This script does three things, in order:

    a  REPRODUCE P4 exactly as the pre-lock scripts computed it, through committed code.
       Match or mismatch at the draft's precision; nothing is tuned to meet a figure.
    b  RE-MEASURE on the declared object: the ten-signal library, instrument returns in
       excess of ff_rf, both books held at the 10% daily volatility target and net of
       5 bp on the weights actually held; the selector is §4's tilt driven by the
       walk-forward, out-of-sample, orthogonalised K = 4 labels, lagged one session,
       with a RANDOM m table; the legs are paired on the five test folds only.
    c  PRICE the declared object: held-weight turnover of both arms, the selector's
       increment, its cost in Sharpe, the breakeven, §5's kill turnover and the
       control's whole cost — the level-C ceiling — re-derived on the right blend.

THE BLINDNESS RULE. The pre-registration is not locked. Every leg entering a power
calculation goes through `protocol.blinded_mde`, which removes both means before the
bootstrap sees them and refuses the call if the `observed` field is not zero. Nothing
here computes, prints or saves a mean, a Sharpe, a Sharpe difference, an IC or a hit
rate of any signal, blend or selector. The m tables are drawn from N(0, 1) and are not
estimated from returns. Printed: standard errors and MDEs of demeaned legs, turnover
|Δw|, costs priced from turnover, counts, dates, transitions, occupancy, the share of
sessions on which the leverage cap binds, and the standard deviation of the demeaned
net legs — an unconditional second moment, the ingredient of the standard error.

THE GUARD. If any reading is not finite, no verdict is printed: a broken instrument is
not a mismatch, and a NaN must never read as a result.

Usage:
    .venv/bin/python scripts/measure_twosigma_power.py [--workers 6] [--tables 8]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
import pandas as pd

from regime_lab.analysis.placebo import transitions
from regime_lab.config import CACHE, RAW
from regime_lab.data.pit import validate
from regime_lab.selection.context import (
    N_STATES,
    as_of,
    context_panel,
    fit_context,
    load_context_features,
    load_log_realised_vol,
    walk_forward_context,
)
from regime_lab.selection.folds import (
    describe_folds,
    fold_of,
    inferential_sessions,
    walk_forward_folds,
)
from regime_lab.selection.library import (
    INDUSTRY_CANDIDATES,
    LIBRARY,
    complete_sessions,
    cross_sectional_z,
    industry_library,
    load_panel,
    log_returns,
    positions,
    window_return,
)
from regime_lab.selection.protocol import (
    COST_BPS,
    DECLARED_EVALUATIONS,
    FAMILY_ALPHA,
    POWER_BLOCKS,
    PRIMARY_TESTS,
    TILT_D,
    TILT_SENSITIVITIES,
    TargetedBook,
    annual_turnover,
    blend,
    blinded_mde,
    breakeven_bps,
    cost_in_sharpe,
    excess_returns,
    map_states,
    mde_z,
    net_of_costs,
    risk_free,
    scale_to_target,
    standalone_sharpe_threshold,
    switch_table,
    target_volatility,
    tilt_mix,
    tilt_table,
    turnover_kill,
)

RULE = "=" * 78
PERIODS = 252
DRAWS = 2_000
MIN_TABLES = 8
#: The pre-lock scripts redrew the toy tilt, and the toy switch, every 21 sessions.
TOY_REDRAW = 21
TOY_BLOCK = 63
ALPHAS: dict[str, float] = {
    "0.05": FAMILY_ALPHA,
    f"0.05/{PRIMARY_TESTS}": FAMILY_ALPHA / PRIMARY_TESTS,
    f"0.05/{DECLARED_EVALUATIONS}": FAMILY_ALPHA / DECLARED_EVALUATIONS,
}
CORRECTED = f"0.05/{PRIMARY_TESTS}"
#: The label of the declared pairing: §4's tilt at d = 0.50 against the control.
PRIMARY = f"tilt d {TILT_D:.2f}"
#: The figure §6 A-1 and §7 set as level A's bar.
DRAFT_CORRECTED_MDE = 0.338
#: The figure §5's cost-kill rule was derived against.
DRAFT_MDE = 0.241
DRAFT_KILL_TURNOVER = 12.0
REALISTIC = COST_BPS["realistic"]
STRESS = COST_BPS["stress"]

INDUSTRY_PANEL = RAW / "panels" / "industry_49.parquet"
FACTORS_PANEL = RAW / "panels" / "factors_5.parquet"
FEATURES = CACHE / "features.parquet"
CROSS_ASSET = RAW / "prices" / "cross_asset.parquet"


# --------------------------------------------------------------------------- the pool


@dataclass(frozen=True)
class Job:
    """One blinded bootstrap: two legs, a mean block, a draw count."""

    key: tuple
    a: np.ndarray
    b: np.ndarray
    block: int
    draws: int


def _run(job: Job) -> tuple[tuple, float, float]:
    result = blinded_mde(job.a, job.b, mean_block=job.block, draws=job.draws, seed=0)
    return job.key, result.se, result.observed


def run_jobs(jobs: list[Job], workers: int) -> tuple[dict[tuple, float], float]:
    """Standard error of every job, and the largest |observed| met (zero if blind)."""
    if workers <= 1:
        results = [_run(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_run, jobs, chunksize=1))
    se = {key: s for key, s, _ in results}
    return se, max(abs(o) for _, _, o in results)


def summary(values: Iterable[float]) -> tuple[float, float, float]:
    """min, median, max."""
    v = np.asarray(list(values), dtype=float)
    return float(v.min()), float(np.median(v)), float(v.max())


def fmt3(values: tuple[float, float, float]) -> str:
    return f"{values[0]:.3f} / {values[1]:.3f} / {values[2]:.3f}"


# ------------------------------------------------------------------ a. the toy library


def toy_library(industries: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The pre-lock power scripts' five signals, held (lagged one session, unit gross).

    ``window_return(lr, 1260, 252)`` is their ``lr.rolling(1008).sum().shift(252)`` and
    ``window_return(lr, 504, 252)`` their "SEAS", ``lr.rolling(252).sum().shift(252)``:
    the 12-month return lagged 12 months, not a same-month seasonality.
    """
    lr = log_returns(industries)
    scores = {
        "MOM": window_return(lr, 252, 21),
        "REV": -lr.rolling(21).sum(),
        "LTREV": -window_return(lr, 1260, 252),
        "LOWVOL": -lr.rolling(63).std(),
        "SEAS": window_return(lr, 504, 252),
    }
    return {k: positions(cross_sectional_z(v)) for k, v in scores.items()}


def toy_targeted(x: np.ndarray) -> np.ndarray:
    """The pre-lock ``vt``: 63-session sd lagged, 10% target, cap 3, warm-up NaN."""
    return scale_to_target(pd.Series(x))[0].to_numpy()


def toy_tilt(legs: np.ndarray, tilt: np.ndarray) -> np.ndarray:
    """Book return of a tilt ``(1 + tilt)/n``, floored and renormalised, via `tilt_table`.

    ``tilt`` is signals x redraw blocks, already scaled by d, as the scripts drew it.
    """
    n, t = legs.shape
    table = tilt_table(pd.DataFrame(tilt.T), d=1.0).to_numpy()
    mix = np.repeat(table, TOY_REDRAW, axis=0)[:t]
    return (legs * mix.T).sum(axis=0)


def section_a(industries: pd.DataFrame, workers: int) -> dict:
    print(f"\n{RULE}\na.  P4 AS DISCLOSED — the pre-lock construction, through committed code\n")
    weights = toy_library(industries)
    idx = complete_sessions(weights)
    stack = np.stack([weights[k].loc[idx].to_numpy() for k in weights])
    legs = np.einsum("stn,tn->st", stack, industries.loc[idx].to_numpy())
    control = toy_targeted(legs.mean(axis=0))
    n_sig, t = legs.shape
    blocks = int(np.ceil(t / TOY_REDRAW))
    years = (idx[-1] - idx[0]).days / 365.25
    print(f"   toy library {list(weights)}, raw returns, gross of cost")
    print(f"   sample {idx[0]:%Y-%m-%d} to {idx[-1]:%Y-%m-%d}, {t:,} sessions, {years:.2f} "
          f"years — the FULL sample, not the test folds")

    jobs: list[Job] = []
    # measure_power.py: a random hard switch every 21 sessions, seed 0, three blocks.
    rng = np.random.default_rng(0)
    switch = np.repeat(rng.integers(0, n_sig, size=blocks), TOY_REDRAW)[:t]
    switched = toy_targeted(legs[switch, np.arange(t)])
    for block in POWER_BLOCKS:
        jobs.append(Job(("switch", block), switched, control, block, DRAWS))
    # measure_power2.py: one generator, d = 0.25 then 0.50 then 1.00, block 63.
    rng = np.random.default_rng(0)
    for d in (0.25, 0.50, 1.00):
        tilt = rng.normal(0.0, d, size=(n_sig, blocks))
        book = toy_targeted(toy_tilt(legs, tilt))
        jobs.append(Job(("power2", d), book, control, TOY_BLOCK, DRAWS))
    # measure_fix.py: eight generators at d = 0.50, 1,500 draws.
    # measure_amtp.py: generator 0 at d = 0.50, 2,000 draws, read at three alphas.
    for seed in range(8):
        tilt = np.random.default_rng(seed).normal(0.0, 0.5, size=(n_sig, blocks))
        book = toy_targeted(toy_tilt(legs, tilt))
        jobs.append(Job(("fix", seed), book, control, TOY_BLOCK, 1_500))
        if seed == 0:
            jobs.append(Job(("amtp", 0), book, control, TOY_BLOCK, DRAWS))

    se, observed = run_jobs(jobs, workers)
    z = {k: mde_z(a) for k, a in ALPHAS.items()}
    switch_mde = {b: se[("switch", b)] * z["0.05"] for b in POWER_BLOCKS}
    fix = [se[("fix", s)] * z["0.05"] for s in range(8)]
    amtp = {k: se[("amtp", 0)] * v for k, v in z.items()}
    power2 = {d: se[("power2", d)] * z["0.05"] for d in (0.25, 0.50, 1.00)}

    print("\n   hard switch, random signal every 21 sessions (measure_power.py), α 0.05:")
    for b in POWER_BLOCKS:
        print(f"      block {b:>3}: SE {se[('switch', b)]:.4f}   MDE {switch_mde[b]:.3f}")
    print("\n   tilt d = 0.50, eight generators, 1,500 draws, block 63 (measure_fix.py):")
    print("      " + "  ".join(f"{m:.3f}" for m in fix))
    print(f"      min / median / max   {fmt3(summary(fix))}")
    print("\n   tilt d = 0.50, generator 0, 2,000 draws, block 63 (measure_amtp.py):")
    for k, v in amtp.items():
        print(f"      α = {k:<8} MDE {v:.3f}")
    corrected_median = summary(fix)[1] * z[CORRECTED] / z["0.05"]
    print(f"      the eight-generator median, read at α = {CORRECTED}: {corrected_median:.3f}")
    print("\n   one generator, d = 0.25 / 0.50 / 1.00 in sequence (measure_power2.py):")
    print("      " + "   ".join(f"d {d:.2f}: {m:.3f}" for d, m in power2.items()))
    print(f"\n   largest |observed| on the demeaned legs: {observed:.1e}")
    return {
        "sessions": t,
        "years": years,
        "switch": switch_mde,
        "fix": fix,
        "amtp": amtp,
        "power2": power2,
        "corrected_median": corrected_median,
        "observed": observed,
    }


# ------------------------------------------------------------- b. the declared object


@dataclass(frozen=True)
class Declared:
    """Everything the declared object is built from."""

    signals: dict[str, pd.DataFrame]
    excess: pd.DataFrame
    sessions: pd.DatetimeIndex
    test: pd.DatetimeIndex
    states: pd.Series
    folds: tuple
    fold: pd.Series
    within_feature_transitions: int
    feature_labels: int


def declared_inputs(industries: pd.DataFrame, market: pd.Series) -> Declared:
    """The ten-signal library in excess, the folds, and the lagged-ready state path."""
    sessions = inferential_sessions()
    weights = industry_library(industries, market, LIBRARY)
    complete = complete_sessions(weights)
    if not sessions.isin(complete).all():
        raise SystemExit("the ten-signal library is not complete on every inferential session")
    signals = {k: v.loc[sessions] for k, v in weights.items()}
    factors = validate(pd.read_parquet(FACTORS_PANEL))
    rf = risk_free(factors, sessions)
    if rf.isna().any():
        raise SystemExit("ff_rf is missing on an inferential session")
    excess = excess_returns(industries.loc[sessions], rf)

    folds = walk_forward_folds(sessions, anchor="end")
    fold = fold_of(sessions, folds)
    panel = context_panel(load_context_features(), load_log_realised_vol())
    wf = walk_forward_context(panel, folds)
    # Labels live on the feature calendar, which carries exchange holidays; the lag of
    # map_states is positional, so the path is read as of each trading session first.
    states = as_of(wf.aligned, sessions)
    return Declared(
        signals=signals,
        excess=excess,
        sessions=sessions,
        test=sessions[fold.to_numpy() > 0],
        states=states,
        folds=folds,
        fold=fold,
        within_feature_transitions=wf.within_fold_transitions(),
        feature_labels=len(wf.labels),
    )


def random_m(seed: int) -> pd.DataFrame:
    """A 4 x 10 m table drawn from N(0, 1). Not estimated from any return."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(size=(N_STATES, len(LIBRARY))), columns=list(LIBRARY))


@dataclass(frozen=True)
class Arm:
    """One book on the declared object, restricted to the test folds where it matters."""

    net: pd.Series
    held_turnover: float
    cap_share: float
    fallback_sessions: int


def build_arm(inputs: Declared, mix: pd.DataFrame | None) -> tuple[TargetedBook, Arm]:
    book = target_volatility(blend(inputs.signals, mix), inputs.excess)
    net = net_of_costs(book.returns, book.weights, bps=REALISTIC)
    test = inputs.test
    fallback = 0 if mix is None else int(inputs.states.shift(1).loc[test].isna().sum())
    return book, Arm(
        net=net.loc[test],
        held_turnover=annual_turnover(book.weights.loc[test]),
        cap_share=float(book.cap_binds.loc[test].astype(float).mean()),
        fallback_sessions=fallback,
    )


def clock_section(inputs: Declared) -> dict:
    print(f"\n{RULE}\nb.  THE DECLARED OBJECT — ten signals, test folds, excess, net of 5 bp\n")
    s, test = inputs.sessions, inputs.test
    print(f"   inferential sample {s[0]:%Y-%m-%d} to {s[-1]:%Y-%m-%d}, {len(s):,} sessions")
    table = describe_folds(inputs.folds, s)
    print("   " + table.to_string(float_format=lambda v: f"{v:.2f}").replace("\n", "\n   "))
    years = (test[-1] - test[0]).days / 365.25
    path = inputs.states.loc[test]
    within = sum(transitions(p) for _, p in path.groupby(inputs.fold.loc[test]))
    total = transitions(path)
    occupancy = path.value_counts(normalize=True).sort_index()
    print(f"\n   paired legs: the union of the five test folds, {test[0]:%Y-%m-%d} to "
          f"{test[-1]:%Y-%m-%d}, {len(test):,} sessions, {years:.2f} years")
    print(f"   state path on the trading calendar: {total} transitions = {total / years:.2f}/yr "
          f"({within} inside folds, {total - within} at fold boundaries, aligned labels)")
    print(f"   on the feature calendar ({inputs.feature_labels:,} labelled dates, holidays "
          f"included): {inputs.within_feature_transitions} transitions inside folds")
    print("   occupancy " + ", ".join(f"{int(k)}: {v:.3f}" for k, v in occupancy.items()))
    return {"years": years, "transitions": total, "per_year": total / years}


#: The paired comparisons measured on the declared object: (kind, d, label, opponent).
#: The last two are OPTIONS for the lock, measured and not chosen: m demeaned across
#: states per signal, so that no static tilt is left; and the conditional tilt paired
#: against a static tilt (the same m, collapsed to one state) instead of the control.
PAIRINGS: tuple[tuple[str, float | None, str, str], ...] = (
    ("tilt", TILT_D, f"tilt d {TILT_D:.2f}", "control"),
    *(("tilt", d, f"tilt d {d:.2f}", "control") for d in TILT_SENSITIVITIES),
    ("switch", None, "hard switch", "control"),
    ("demeaned", TILT_D, "m demeaned", "control"),
    ("tilt", TILT_D, "vs static K=1", "static"),
)


def static_mix(inputs: Declared, m: pd.DataFrame, d: float) -> pd.DataFrame:
    """The tilt of ``m`` averaged over states, held whenever a state is known.

    The random stand-in for "the same estimator at K = 1". It shares the conditional
    arm's lag and fallback, so the pair differs only where the state does.
    """
    table = tilt_table(m.mean(axis=0).to_frame().T, d=d)
    known = pd.Series(np.where(inputs.states.notna(), 0.0, np.nan), index=inputs.states.index)
    return map_states(known, table)


def section_b(inputs: Declared, tables: int, workers: int) -> dict:
    control_book, control = build_arm(inputs, None)
    arms: dict[tuple, Arm] = {}
    for seed in range(tables):
        m = random_m(seed)
        for d in (TILT_D, *TILT_SENSITIVITIES):
            arms[("tilt", d, seed)] = build_arm(inputs, tilt_mix(inputs.states, m, d=d))[1]
        arms[("switch", None, seed)] = build_arm(
            inputs, map_states(inputs.states, switch_table(m)))[1]
        arms[("demeaned", TILT_D, seed)] = build_arm(
            inputs, tilt_mix(inputs.states, m - m.mean(axis=0), d=TILT_D))[1]
        arms[("static", TILT_D, seed)] = build_arm(inputs, static_mix(inputs, m, TILT_D))[1]

    jobs: list[Job] = []
    for kind, d, label, opponent in PAIRINGS:
        for seed in range(tables):
            a = arms[(kind, d, seed)].net.to_numpy()
            b = (control if opponent == "control" else arms[("static", d, seed)]).net.to_numpy()
            if not (np.isfinite(a).all() and np.isfinite(b).all()):
                raise SystemExit(f"a paired leg is not complete on the test folds: {label}")
            for block in POWER_BLOCKS:
                jobs.append(Job((label, seed, block), a, b, block, DRAWS))

    se, observed = run_jobs(jobs, workers)
    z = {k: mde_z(a) for k, a in ALPHAS.items()}
    mde: dict[tuple, tuple[float, float, float]] = {}
    for _, _, label, _ in PAIRINGS:
        for block in POWER_BLOCKS:
            ses = [se[(label, seed, block)] for seed in range(tables)]
            for key, zv in z.items():
                mde[(label, block, key)] = summary(s * zv for s in ses)
            mde[(label, block, "se")] = summary(ses)

    print(f"\n   selector: §4 tilt on {tables} random m tables (4 x 10, N(0,1)), lagged one "
          f"session;\n   {arms[('tilt', TILT_D, 0)].fallback_sessions} test session(s) "
          "hold the equal-weight fallback (no state known the evening before)")
    print(f"   blinded bootstrap, {DRAWS:,} draws; min / median / max over the tables\n")
    header = "   ".join(f"α {k:<8}" + " " * 14 for k in ALPHAS)
    print(f"   {'arm':<14} {'block':>5}   {'SE':<22}   {header}")
    for _, _, label, _ in PAIRINGS:
        if label == "m demeaned":
            print("   options for the static-tilt question, measured, not chosen:")
        for block in POWER_BLOCKS:
            cells = "   ".join(fmt3(mde[(label, block, k)]) for k in ALPHAS)
            print(f"   {label:<14} {block:>5}   {fmt3(mde[(label, block, 'se')])}   {cells}")
    print("   (every arm against the equal-weight control except the last, which pairs the "
          "conditional\n   tilt with the same m collapsed to one state)")
    print(f"\n   largest |observed| on the demeaned legs: {observed:.1e}")
    return {"control_book": control_book, "control": control, "arms": arms, "mde": mde,
            "observed": observed}


def lo_section(years: float, full_years: float) -> dict:
    print("\n   Lo's standalone threshold, the other convention, for comparison:")
    out = {}
    for label, alpha in ALPHAS.items():
        out[label] = standalone_sharpe_threshold(years, alpha=alpha)
        full = standalone_sharpe_threshold(full_years, alpha=alpha)
        print(f"      α = {label:<8} {years:.2f} years: {out[label]:.3f}    "
              f"({full_years:.2f} years: {full:.3f})")
    return out


# --------------------------------------------------------------------------- c. costs


def disclosed_increment(
    industries: pd.DataFrame, market: pd.Series
) -> tuple[float, float, int]:
    """§5's +2.6x/yr as measured before the lock: eleven signals, in-sample clock.

    measure_costs.py: the eleven signals that still hold IND_REV_1W, one K-means fit on
    the full context sample without orthogonalisation, in-sample labels read on the
    same session (no lag), a random M ~ N(0, d) drawn after the d = 0.25 one from the
    same generator, turnover of the UNSCALED blend. Returns (control, tilt, sessions).
    """
    names = [n for n in INDUSTRY_CANDIDATES if n != "IND_ACCEL"]
    weights = industry_library(industries, market, names)
    idx = complete_sessions(weights)
    signals = {k: v.loc[idx] for k, v in weights.items()}
    panel = context_panel(load_context_features(), load_log_realised_vol())
    labels = fit_context(panel, orthogonalise=False).train_labels
    states = labels.reindex(idx).ffill()
    rng = np.random.default_rng(0)
    rng.normal(0.0, 0.25, size=(N_STATES, len(names)))
    m = pd.DataFrame(rng.normal(0.0, 0.50, size=(N_STATES, len(names))), columns=names)
    tilt = blend(signals, map_states(states, tilt_table(m, d=1.0), lag=0))
    return annual_turnover(blend(signals)), annual_turnover(tilt), len(idx)


def leg_sd(leg: pd.Series) -> float:
    """Annualised sd of a leg. A standard deviation is invariant to the mean, so none is
    formed: the same second moment the bootstrap standard error is built from."""
    return float(np.std(leg.to_numpy(), ddof=1) * np.sqrt(PERIODS))


def section_c(inputs: Declared, b: dict, tables: int, prelock: tuple) -> dict:
    print(f"\n{RULE}\nc.  COSTS ON THE DECLARED OBJECT — held weights, after the multiplier\n")
    control: Arm = b["control"]
    test = inputs.test
    unscaled = blend(inputs.signals)
    control_unscaled_full = annual_turnover(unscaled)
    control_unscaled_test = annual_turnover(unscaled.loc[test])
    sd = leg_sd(control.net)

    def arm_rows(kind: str, d: float | None) -> dict:
        arms = [b["arms"][(kind, d, s)] for s in range(tables)]
        held = [a.held_turnover for a in arms]
        incr = [h - control.held_turnover for h in held]
        caps = [a.cap_share for a in arms]
        sds = [leg_sd(a.net) for a in arms]
        return {"held": summary(held), "incr": summary(incr), "cap": summary(caps),
                "sd": summary(sds)}

    rows = {f"tilt d {d:.2f}": arm_rows("tilt", d) for d in (TILT_D, *TILT_SENSITIVITIES)}
    rows["hard switch"] = arm_rows("switch", None)
    rows["m demeaned"] = arm_rows("demeaned", TILT_D)
    rows["static K=1"] = arm_rows("static", TILT_D)

    print("   the control, the equal-weight blend of the ten:")
    print(f"      unscaled weights, full inferential sample   {control_unscaled_full:6.1f} x/yr")
    print(f"      unscaled weights, test folds                {control_unscaled_test:6.1f} x/yr")
    print(f"      HELD weights, test folds                    {control.held_turnover:6.1f} x/yr")
    gross = float(unscaled.loc[test].abs().sum(axis=1).median())
    print(f"      unscaled gross Σ|w|, median on the test folds {gross:.3f} (each signal holds 1:")
    print("      ten nearly orthogonal books net against each other before the target scales up)")
    print(f"      leverage cap binds on {control.cap_share:.1%} of test sessions; sd of the "
          f"demeaned net leg {sd:.2%} a year against the 10% target")
    whole = {bps: cost_in_sharpe(control.held_turnover, bps) for bps in COST_BPS.values()}
    print("      whole cost in Sharpe at the 10% target: " + ", ".join(
        f"{v:.3f} at {bps:.0f} bp" for bps, v in whole.items()))
    print(f"      (at the leg's own sd, {control.held_turnover * REALISTIC / 1e4 / sd:.3f} "
          "at 5 bp)")

    print("\n   the selector arms, min / median / max over the tables:")
    print(f"      {'arm':<13} {'held x/yr':<24} {'increment x/yr':<24} {'cap binds':<22} sd")
    for name, r in rows.items():
        cap = f"{r['cap'][0]:.1%} / {r['cap'][1]:.1%} / {r['cap'][2]:.1%}"
        sdr = f"{r['sd'][0]:.2%} / {r['sd'][2]:.2%}"
        print(f"      {name:<13} {fmt3(r['held']):<24} {fmt3(r['incr']):<24} {cap:<22} {sdr}")

    arms = b["arms"]
    over_static = summary(
        arms[("tilt", TILT_D, s)].held_turnover - arms[("static", TILT_D, s)].held_turnover
        for s in range(tables))
    print(f"      the conditional tilt over its static twin: {fmt3(over_static)} x/yr — the part "
          "of the\n      increment that the state, not the static tilt, trades")

    incr = rows[PRIMARY]["incr"][1]
    uncorr = b["mde"][(PRIMARY, TOY_BLOCK, "0.05")][1]
    corr = b["mde"][(PRIMARY, TOY_BLOCK, CORRECTED)][1]
    print(f"\n   the increment at d = {TILT_D:.2f} (median {incr:.2f} x/yr), priced:")
    for bps in COST_BPS.values():
        print(f"      {bps:>4.0f} bp: {cost_in_sharpe(incr, bps):.4f} Sharpe")
    be = {"0.05": breakeven_bps(uncorr, incr), CORRECTED: breakeven_bps(corr, incr)}
    print(f"   breakeven against the re-measured MDE (block {TOY_BLOCK}, median): "
          f"{be['0.05']:.0f} bp at α 0.05 ({uncorr:.3f}), {be[CORRECTED]:.0f} bp at "
          f"α {CORRECTED} ({corr:.3f})")
    kill = {
        "draft": turnover_kill(DRAFT_MDE, STRESS),
        "0.05": turnover_kill(uncorr, STRESS),
        CORRECTED: turnover_kill(corr, STRESS),
    }
    print(f"   §5 kill turnover at {STRESS:.0f} bp: {kill['draft']:.2f} x/yr against the "
          f"draft's {DRAFT_MDE}; {kill['0.05']:.2f} against {uncorr:.3f}; "
          f"{kill[CORRECTED]:.2f} against {corr:.3f}")
    over = sum(i > kill["0.05"] for i in
               (b["arms"][("tilt", TILT_D, s)].held_turnover - control.held_turnover
                for s in range(tables)))
    print(f"   {over} of {tables} RANDOM maps at d = {TILT_D:.2f} already exceed the kill "
          f"turnover at α 0.05 — a property of the clock, before any fit")

    ctl_pre, tilt_pre, n_pre = prelock
    print("\n   §5 as disclosed, reproduced (eleven signals, in-sample unorthogonalised clock,")
    print(f"   no lag, UNSCALED weights, {n_pre:,} sessions): control {ctl_pre:.1f} x/yr, "
          f"tilt {tilt_pre:.1f}, increment {tilt_pre - ctl_pre:+.2f} x/yr")
    return {
        "control_held": control.held_turnover,
        "control_unscaled_full": control_unscaled_full,
        "control_unscaled_test": control_unscaled_test,
        "control_cap": control.cap_share,
        "control_sd": sd,
        "control_gross": gross,
        "whole": whole,
        "rows": rows,
        "incr": incr,
        "over_static": over_static,
        "breakeven": be,
        "kill": kill,
        "prelock": prelock,
    }


# ---------------------------------------------------------------------- verdicts


@dataclass(frozen=True)
class Check:
    """One disclosed figure against its reproduction, at the draft's precision."""

    label: str
    disclosed: float
    decimals: int
    measured: float
    note: str = ""

    @property
    def match(self) -> bool:
        return abs(self.measured - self.disclosed) <= 0.5 * 10.0**-self.decimals + 1e-12

    def row(self) -> str:
        flag = "MATCH" if self.match else "MISMATCH"
        return (f"   {self.label:<54} {self.disclosed:>8.{self.decimals}f} "
                f"{self.measured:>10.{self.decimals + 1}f}   {flag}")


def finite(readings: Mapping[str, object]) -> list[str]:
    """Names of the readings that hold a non-finite number anywhere."""
    broken = []
    for name, value in readings.items():
        flat = np.asarray(list(_numbers(value)), dtype=float)
        if flat.size == 0 or not np.isfinite(flat).all():
            broken.append(name)
    return broken


def _numbers(value: object) -> Iterable[float]:
    if isinstance(value, Mapping):
        for v in value.values():
            yield from _numbers(v)
    elif isinstance(value, list | tuple):
        for v in value:
            yield from _numbers(v)
    elif isinstance(value, int | float | np.floating | np.integer):
        yield float(value)


def verdicts(a: dict, b: dict, c: dict, lo: dict, clock: dict) -> None:
    fix = summary(a["fix"])
    ctl_pre, tilt_pre, _ = c["prelock"]
    checks = [
        Check("P4  hard switch, block 63", 0.474, 3, a["switch"][TOY_BLOCK]),
        Check("P4  tilt d 0.50, eight draws: min", 0.221, 3, fix[0]),
        Check("P4  tilt d 0.50, eight draws: median", 0.241, 3, fix[1]),
        Check("P4  tilt d 0.50, eight draws: max", 0.294, 3, fix[2]),
        Check("§7  uncorrected, the figure 0.338 is scaled from", 0.272, 3, a["amtp"]["0.05"]),
        Check(f"P4  α = {CORRECTED}", 0.338, 3, a["amtp"][CORRECTED],
              "generator 0 (0.272) scaled, not the eight-draw median"),
        Check(f"P4  α = 0.05/{DECLARED_EVALUATIONS}", 0.376, 3,
              a["amtp"][f"0.05/{DECLARED_EVALUATIONS}"]),
        Check("measure_fix.py's 'earlier single-seed' figure", 0.238, 3, a["power2"][0.50],
              "quoted in the script's printout, not in the draft"),
        Check("P5  selector increment (x/yr)", 2.6, 1, tilt_pre - ctl_pre),
        Check("P5  blend turnover, eleven signals (x/yr)", 21.6, 1, ctl_pre),
    ]
    print(f"\n{RULE}\nREPRODUCTION — disclosed against measured, at the draft's precision\n")
    print(f"   {'':<54} {'disclosed':>8} {'measured':>10}")
    for check in checks:
        print(check.row())
        if check.note:
            print(f"        {check.note}")
    failed = [c for c in checks if not c.match]
    print(f"\n   {len(checks) - len(failed)} of {len(checks)} match.")

    corr = b["mde"][(PRIMARY, TOY_BLOCK, CORRECTED)]
    uncorr = b["mde"][(PRIMARY, TOY_BLOCK, "0.05")]
    widest = max(b["mde"][(PRIMARY, blk, CORRECTED)][1] for blk in POWER_BLOCKS)
    print(f"\n{RULE}\nTHE DECLARED OBJECT AGAINST THE DRAFT\n")
    print(f"   A-1's bar in the draft: {DRAFT_CORRECTED_MDE} (toy library, random 21-session "
          f"tilt, {a['years']:.1f} years, gross, one draw).")
    print(f"   On the declared object, block {TOY_BLOCK}, α {CORRECTED}, d {TILT_D:.2f}: "
          f"median {corr[1]:.3f}, range {corr[0]:.3f} to {corr[2]:.3f} over the tables "
          f"— {corr[1] / DRAFT_CORRECTED_MDE:.2f} times the draft's figure.")
    print(f"   The widest block's median at α {CORRECTED}: {widest:.3f}. Uncorrected, block "
          f"{TOY_BLOCK}: median {uncorr[1]:.3f} against the draft's {DRAFT_MDE}.")
    print(f"   Lo's standalone threshold over the same {clock['years']:.2f} years: "
          f"{lo[CORRECTED]:.3f} at α {CORRECTED}.")
    print(f"   The control's whole cost at 5 bp, the level-C ceiling: {c['whole'][REALISTIC]:.3f}"
          f" Sharpe (draft 0.108, on the eleven-signal unscaled blend).")
    print(f"   §5's kill turnover re-derived at {STRESS:.0f} bp against the re-measured "
          f"uncorrected MDE: {c['kill']['0.05']:.2f} x/yr (draft {DRAFT_KILL_TURNOVER:.0f}); "
          f"the random-map increment at d {TILT_D:.2f} is {c['incr']:.2f} x/yr.")


# --------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1))
    parser.add_argument("--tables", type=int, default=MIN_TABLES)
    args = parser.parse_args(argv)
    if args.tables < MIN_TABLES:
        raise SystemExit(f"at least {MIN_TABLES} random m tables")
    missing = [p for p in (INDUSTRY_PANEL, FACTORS_PANEL, FEATURES, CROSS_ASSET)
               if not p.exists()]
    if missing:
        raise SystemExit("missing input: " + ", ".join(p.name for p in missing))

    clock_start = time.perf_counter()
    lap: dict[str, float] = {}

    def timed(name: str, fn: Callable, *fargs):
        start = time.perf_counter()
        out = fn(*fargs)
        lap[name] = time.perf_counter() - start
        return out

    print(RULE)
    print("TWO SIGMA PLAN — THE POWER OF LEVEL A, BLIND")
    print(RULE)
    print("\n   Every leg is demeaned before the bootstrap. No mean, Sharpe or Sharpe")
    print("   difference is computed. The m tables are N(0,1) draws, not estimates.")

    industries = load_panel("industry_49")
    market = load_panel("factors_5")["ff_mkt-rf"]
    a = timed("a", section_a, industries, args.workers)
    inputs = timed("inputs", declared_inputs, industries, market)
    clock = clock_section(inputs)
    b = timed("b", section_b, inputs, args.tables, args.workers)
    full_years = (inputs.sessions[-1] - inputs.sessions[0]).days / 365.25
    lo = lo_section(clock["years"], full_years)
    prelock = timed("prelock", disclosed_increment, industries, market)
    c = timed("c", section_c, inputs, b, args.tables, prelock)

    readings = {
        "a": {k: v for k, v in a.items() if k != "observed"},
        "b": {k: v for k, v in b["mde"].items()},
        "c": {k: v for k, v in c.items() if k != "rows"},
        "c rows": c["rows"],
        "lo": lo,
        "clock": clock,
    }
    broken = finite({k: v for k, v in readings.items()})
    if broken or not (a["observed"] < 1e-9 and b["observed"] < 1e-9):
        print(f"\n{RULE}\n   A READING IS NOT FINITE, OR A MEAN REACHED A BOOTSTRAP: {broken}")
        print("   No verdict. A broken instrument is not a mismatch.")
        print(f"\n{RULE}")
        sys.exit(1)
    verdicts(a, b, c, lo, clock)

    total = time.perf_counter() - clock_start
    print(f"\n{RULE}\nd.  RUNTIME\n")
    for name, seconds in lap.items():
        print(f"   {name:<8} {seconds:7.1f} s")
    print(f"   total    {total:7.1f} s on {args.workers} worker(s)")
    print(f"\n{RULE}")


if __name__ == "__main__":
    main()

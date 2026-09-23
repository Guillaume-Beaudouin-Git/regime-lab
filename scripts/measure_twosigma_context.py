"""Two Sigma plan — the context partition, the walk-forward, the matched placebo.

`PRESPEC_TWOSIGMA.md` is a DRAFT. §1 P3 and P6 disclose a clock and a placebo that
the pre-lock scripts measured on ONE K-means fit over the whole context sample, with
in-sample labels. §4 declares a different object: a walk-forward refit, orthogonalised,
labels out of sample by nearest centroid. This script does two things and keeps them
apart:

    A   REPRODUCTION of P3, of the episode counts of the pre-lock power script and of
        P6, exactly as the pre-lock scripts computed them, from committed code
        (`regime_lab/selection/context.py`, `regime_lab/analysis/placebo.py`). A
        mismatch is printed as a mismatch; nothing is tuned to meet a disclosed number.
    B   THE DECLARED OBJECT: the walk-forward clock, per fold and pooled, for both
        readings of where a context training window starts, the unorthogonalised
        variant, the other anchor and the declared sensitivities K in {3, 5, 6}.
    C   The labels on the 49-industry session calendar, and the one-session lag.
    C2  THE CELLS of the lock's minimum-cell rule (§12.4), counted on the stamped
        paths at K = 4 and at every K sensitivity, the test sessions that sit at the
        abstaining row, and the 21-session smoothing sensitivity (§12.13): its clock,
        its cells, and whether the uniform placebo is exact on it.
    D   PLACEBO FEASIBILITY on the declared object, per (fold, segment) block: the
        draft's swap repair against the proposed uniform construction.
    E   Runtime.

THE BLINDNESS RULE. The pre-registration is not locked. This script reads no return of
any signal or industry: the industry panel is read for its calendar and for a
completeness count only. What it prints is counts, dates, transitions, occupancy,
episode lengths, refit agreement, and eta-squared of a partition against the realised
volatility of eq_us_large — the variable the partition is orthogonalised against.

THE GUARD. If any reading is not finite, the script prints no verdict: a broken
instrument is not a mismatch.

Usage:
    .venv/bin/python scripts/measure_twosigma_context.py [--draws 1000]
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from regime_lab.analysis.placebo import (
    JoinFreeSequences,
    check_placebos,
    matched_placebos,
    occupancy,
    placebo_feasibility,
    run_lengths,
    span_years,
    transitions,
)
from regime_lab.config import RAW
from regime_lab.selection.context import (
    LOG_RV,
    MIN_CELL_EPISODES,
    MIN_CELL_SESSIONS,
    N_STATES,
    SENSITIVITY_STATES,
    SMOOTHING_WINDOW,
    WalkForwardContext,
    as_of,
    context_panel,
    eta_squared,
    fit_context,
    load_context_features,
    load_log_realised_vol,
    placebo_groups,
    qualifying_cells,
    session_paths,
    trailing_mode,
    walk_forward_context,
)
from regime_lab.selection.folds import (
    Fold,
    describe_folds,
    inferential_sessions,
    walk_forward_folds,
)
from regime_lab.selection.protocol import map_states

RULE = "=" * 78
#: The pre-lock power script's toy library was first complete on the 1,261st session
#: of the industry calendar: long-term reversal over 1,008 sessions, skipping 252,
#: lagged one.
TOY_WARMUP = 1_260
PRE_LOCK_PLACEBO_DRAWS = 300


@dataclass(frozen=True)
class Check:
    """One disclosed figure against its reproduction, at the draft's precision."""

    label: str
    disclosed: float
    decimals: int
    measured: float
    note: str = ""

    @property
    def finite(self) -> bool:
        return math.isfinite(self.measured)

    @property
    def match(self) -> bool:
        return abs(self.measured - self.disclosed) <= 0.5 * 10.0**-self.decimals + 1e-12

    def row(self) -> str:
        shown = self.decimals + 1 if self.decimals else 0
        flag = "MATCH" if self.match else "MISMATCH"
        note = f"   ({self.note})" if self.note else ""
        return (f"   {self.label:<46} {self.disclosed:>9.{self.decimals}f} "
                f"{self.measured:>11.{shown}f}   {flag}{note}")


READINGS: list[float] = []


def reading(value: float) -> float:
    """Record a reading for the guard and hand it back."""
    READINGS.append(float(value))
    return float(value)


def episodes_by_state(labels: np.ndarray, states: int) -> np.ndarray:
    _, s = run_lengths(np.asarray(labels, dtype=np.int64))
    return np.bincount(s, minlength=states)


def fmt_counts(counts: np.ndarray) -> str:
    return "/".join(str(int(c)) for c in counts)


def fmt_occupancy(labels: pd.Series, states: int) -> str:
    occ = occupancy(labels).reindex(range(states), fill_value=0.0)
    return "/".join(f"{v:.2f}" for v in occ)


# ----------------------------------------------------------------- A. reproduction


def reproduction(panel: pd.DataFrame, rv_level: pd.Series, industry: pd.DatetimeIndex,
                 library: pd.DatetimeIndex) -> list[Check]:
    print(f"\n{RULE}\nA.  REPRODUCTION — one full-sample K-means fit, in-sample labels\n")
    years = span_years(panel.index)
    print(f"   context panel {panel.index[0]:%Y-%m-%d} to {panel.index[-1]:%Y-%m-%d}, "
          f"{len(panel):,} rows (feature calendar, weekdays), {years:.2f} years")
    print("   eta2 is printed on both scales: the pre-lock script took the unorthogonalised")
    print("   figure against rv in LEVELS and the orthogonalised one against LOG rv.\n")
    print("   K  variant   transitions  per yr  eta2(rv)  eta2(log rv)  occupancy")
    fits = {}
    for orth in (False, True):
        for k in (3, 4, 5, 6):
            labels = fit_context(panel, orthogonalise=orth, n_states=k).train_labels
            fits[(orth, k)] = labels
            tr = transitions(labels)
            e_level = reading(eta_squared(labels, rv_level))
            e_log = reading(eta_squared(labels, panel[LOG_RV]))
            print(f"   {k}  {'orth' if orth else 'raw ':<8} {tr:>11d}  {reading(tr / years):6.2f}"
                  f"  {e_level:8.3f}  {e_log:12.3f}  {fmt_occupancy(labels, k)}")

    checks: list[Check] = [Check("context span, years", 34.52, 2, reading(years))]
    # P3 and the pre-lock tables (clock.csv: raw, eta2 on rv levels; clock_resid.csv:
    # orthogonalised, eta2 on log rv).
    disclosed = {
        (False, 3): (141, 4.08, 0.371), (False, 4): (223, 6.46, 0.421),
        (False, 5): (290, 8.40, 0.487), (False, 6): (293, 8.49, 0.466),
        (True, 3): (200, 5.79, 0.008), (True, 4): (287, 8.31, 0.048),
        (True, 5): (266, 7.71, 0.098), (True, 6): (264, 7.65, 0.143),
    }
    for (orth, k), (tr, per_year, eta2) in disclosed.items():
        labels = fits[(orth, k)]
        name = f"K={k} {'orth' if orth else 'raw'}"
        scale, values = ("log rv", panel[LOG_RV]) if orth else ("rv level", rv_level)
        checks += [
            Check(f"{name}: transitions", tr, 0, transitions(labels)),
            Check(f"{name}: per year", per_year, 2, transitions(labels) / years),
            Check(f"{name}: eta2 vs {scale}", eta2, 3, eta_squared(labels, values)),
        ]
    raw4 = fits[(False, 4)]
    checks.append(Check("K=4 raw: episodes", 224, 0, len(run_lengths(raw4.to_numpy())[0])))
    checks.append(Check("K=4 orth: episodes", 288, 0,
                        len(run_lengths(fits[(True, 4)].to_numpy())[0])))
    occ = occupancy(raw4)
    for state, share in enumerate((0.261, 0.427, 0.091, 0.221)):
        checks.append(Check(f"K=4 raw: occupancy of state {state}", share, 3, occ[state]))

    # The pre-lock power script's episode counts: the raw K=4 labels read on its toy
    # library's sample.
    toy = industry[TOY_WARMUP:]
    toy = toy[toy <= library[-1]]
    on_toy = as_of(raw4, toy).astype(np.int64)
    lengths, states = run_lengths(on_toy.to_numpy())
    per_state = np.bincount(states, minlength=4)
    print(f"\n   raw K=4 labels on the pre-lock toy library's sample, {toy[0]:%Y-%m-%d} to "
          f"{toy[-1]:%Y-%m-%d}, {len(toy):,} sessions ({span_years(toy):.2f} years):")
    print(f"      {len(lengths)} episodes, per state {fmt_counts(per_state)}, median "
          f"{np.median(lengths):.0f}, mean {lengths.mean():.1f}, p90 "
          f"{np.quantile(lengths, 0.9):.0f} sessions; {transitions(on_toy)} transitions = "
          f"{transitions(on_toy) / span_years(toy):.2f}/yr")
    checks += [
        Check("toy sample: sessions", 7_952, 0, len(toy)),
        Check("toy sample: episodes", 215, 0, len(lengths)),
        *[Check(f"toy sample: episodes of state {s}", d, 0, per_state[s])
          for s, d in enumerate((46, 71, 20, 78))],
        Check("toy sample: median episode", 8, 0, float(np.median(lengths))),
        Check("toy sample: mean episode", 37.0, 1, float(lengths.mean())),
        Check("toy sample: per year (placebo.py docstring)", 6.77, 2,
              transitions(on_toy) / span_years(toy),
              "the clock §8 attempts 1-2 were measured against"),
    ]
    on_lib = as_of(raw4, library).astype(np.int64)
    lib_lengths, lib_states = run_lengths(on_lib.to_numpy())
    print(f"   the same labels on the ten-signal library's sample, {library[0]:%Y-%m-%d} to "
          f"{library[-1]:%Y-%m-%d}, {len(library):,} sessions:")
    print(f"      {len(lib_lengths)} episodes, per state "
          f"{fmt_counts(np.bincount(lib_states, minlength=4))}, median "
          f"{np.median(lib_lengths):.0f}, mean {lib_lengths.mean():.1f}")

    # P6: the swap repair on the full-sample raw clock, one block.
    used = placebo_feasibility(raw4, PRE_LOCK_PLACEBO_DRAWS)
    accepted = int((used > 0).all(axis=1).sum())
    draws = matched_placebos(raw4, PRE_LOCK_PLACEBO_DRAWS, method="swap")
    check = check_placebos(raw4, draws.draws)
    print(f"\n   P6, swap repair on the raw K=4 clock (one block, {PRE_LOCK_PLACEBO_DRAWS} "
          f"draws): {accepted} accepted, transitions {check.transitions.min()}-"
          f"{check.transitions.max()} (sd {check.transitions_sd:.1f}), max occupancy "
          f"deviation {check.max_occupancy_deviation:.4f}, episodes preserved "
          f"{check.episodes_preserved}, max tries {int(used.max())}")
    checks += [
        Check("P6: draws accepted of 300", 300, 0, accepted),
        Check("P6: transitions in every draw", 223, 0,
              float(check.transitions.mean()) if check.transitions_sd == 0 else float("nan")),
        Check("P6: max occupancy deviation", 0.0, 4, check.max_occupancy_deviation),
    ]
    return checks


# ------------------------------------------------------------ B. declared object


@dataclass(frozen=True)
class Clock:
    """Blind summary of one walk-forward partition on the industry sessions."""

    k: int
    per_year: float
    train_per_year: float
    boundary: int
    min_test_cell: int
    zero_test_cells: int
    min_train_cell: int
    median_length: float
    mean_length: float
    eta2: float
    agreement_min: float


def fold_rows(wf: WalkForwardContext, folds: tuple[Fold, ...], paths: pd.Series,
              panel: pd.DataFrame, k: int,
              full: pd.Series | None = None) -> tuple[list[dict], Clock]:
    rows, test_cells, train_cells, lengths = [], [], [], []
    for fold, agreement in zip(folds, wf.agreement, strict=True):
        test = paths.xs((fold.number, "test"), level=["fold", "segment"])
        train = paths.xs((fold.number, "train"), level=["fold", "segment"])
        model = wf.model(fold.number)
        ctx = wf.test_labels(fold.number)
        run, _ = run_lengths(test.to_numpy())
        lengths.append(run)
        test_cells.append(episodes_by_state(test.to_numpy(), k))
        train_cells.append(episodes_by_state(train.to_numpy(), k))
        rows.append({
            "fold": fold.number,
            "sessions": len(test),
            "tr_ctx": transitions(ctx),
            "tr_ind": transitions(test),
            "per_yr": reading(transitions(test) / fold.test_years),
            "episodes": fmt_counts(test_cells[-1]),
            "occupancy": fmt_occupancy(test, k),
            "eta2": reading(eta_squared(ctx, panel[LOG_RV])),
            "agree": reading(agreement),
            "train_from": f"{model.train_start:%Y-%m-%d}",
            "train_per_yr": reading(transitions(model.train_labels)
                                    / span_years(model.train_labels.index)),
            "train_episodes": fmt_counts(train_cells[-1]),
            "full_per_yr": float("nan") if full is None else reading(
                transitions(as_of(full, test.index)) / fold.test_years),
        })
    test_cells_arr, train_cells_arr = np.vstack(test_cells), np.vstack(train_cells)
    lengths_all = np.concatenate(lengths)
    total_tr = sum(r["tr_ind"] for r in rows)
    total_years = sum(f.test_years for f in folds)
    train_rates = [r["train_per_yr"] for r in rows]
    clock = Clock(
        k=k,
        per_year=reading(total_tr / total_years),
        train_per_year=reading(float(np.mean(train_rates))),
        boundary=wf.boundary_transitions(),
        min_test_cell=int(test_cells_arr.min()),
        zero_test_cells=int((test_cells_arr == 0).sum()),
        min_train_cell=int(train_cells_arr.min()),
        median_length=reading(float(np.median(lengths_all))),
        mean_length=reading(float(lengths_all.mean())),
        eta2=reading(eta_squared(wf.aligned, panel[LOG_RV])),
        agreement_min=reading(min(wf.agreement)),
    )
    return rows, clock


def print_fold_table(rows: list[dict]) -> None:
    print("   fold  sess  trans ctx/ind  per yr  full /yr  episodes/state   occupancy          "
          "eta2(log)  agree   train from   train /yr  train episodes/state")
    for r in rows:
        print(f"   {r['fold']:>4}  {r['sessions']:>4}  {r['tr_ctx']:>5}/{r['tr_ind']:<5}  "
              f"{r['per_yr']:6.2f}  {r['full_per_yr']:7.2f}  {r['episodes']:<15}  "
              f"{r['occupancy']:<18} {r['eta2']:8.3f}  {r['agree']:5.3f}   {r['train_from']}  "
              f"{r['train_per_yr']:8.2f}   {r['train_episodes']}")


def print_clock(label: str, c: Clock) -> None:
    print(f"   {label:<34} K={c.k}  {c.per_year:5.2f}/yr OOS (in-sample {c.train_per_year:5.2f})"
          f"  boundary {c.boundary}  min cell test {c.min_test_cell} (zero {c.zero_test_cells})"
          f" train {c.min_train_cell}  episode median {c.median_length:.0f} mean "
          f"{c.mean_length:.1f}  eta2 {c.eta2:.3f}  agree>={c.agreement_min:.3f}")


def declared_object(panel: pd.DataFrame, sessions: pd.DatetimeIndex,
                    starts: dict[str, pd.Timestamp | None]) -> dict:
    print(f"\n{RULE}\nB.  THE DECLARED OBJECT — walk-forward refit, labels out of sample\n")
    folds_end = walk_forward_folds(sessions, anchor="end")
    folds_start = walk_forward_folds(sessions, anchor="start")
    for name, folds in (("end", folds_end), ("start", folds_start)):
        print(f"   anchor='{name}'")
        table = describe_folds(folds, sessions)
        print("   " + table.round(3).to_string().replace("\n", "\n   "))
        print()
    print("   Clock on the industry sessions (labels read as of each session); 'ctx' counts")
    print("   the same labels on the feature calendar. Episodes are cut at fold edges.")
    print("   eta2 per fold: that fold's OOS labels against log rv; pooled: aligned labels.")
    print("   'train /yr': the fold model's in-sample labels over its whole training window.")
    print("   'full /yr': the disclosed object — ONE fit on the whole context sample, in-sample")
    print("   labels — counted on the same test window, to separate period from refit.")
    full = fit_context(panel).train_labels

    out: dict = {"folds": folds_end}
    for start_name, start in starts.items():
        print(f"\n   --- context training starts: {start_name} ---\n")
        t0 = time.perf_counter()
        wf = walk_forward_context(panel, folds_end, train_from=start)
        out[("time", start_name)] = time.perf_counter() - t0
        paths = session_paths(wf, folds_end, sessions)
        rows, clock = fold_rows(wf, folds_end, paths, panel, N_STATES, full)
        print_fold_table(rows)
        years = sum(f.test_years for f in folds_end)
        full_rate = sum(r["full_per_yr"] * f.test_years
                        for r, f in zip(rows, folds_end, strict=True)) / years
        print(f"\n   same {years:.1f} test years: disclosed object {full_rate:.2f}/yr, declared "
              f"object {clock.per_year:.2f}/yr; disclosed object over its whole "
              f"{span_years(full.index):.2f} years {transitions(full) / span_years(full.index):.2f}"
              "/yr")
        print()
        print_clock("pooled, orth, anchor end", clock)
        out[(start_name, N_STATES)] = (wf, paths)
        out[("clock", start_name)] = clock

        raw = walk_forward_context(panel, folds_end, train_from=start, orthogonalise=False)
        _, raw_clock = fold_rows(raw, folds_end, session_paths(raw, folds_end, sessions),
                                 panel, N_STATES)
        print_clock("pooled, UNORTHOGONALISED, end", raw_clock)

        other = walk_forward_context(panel, folds_start, train_from=start)
        _, other_clock = fold_rows(other, folds_start,
                                   session_paths(other, folds_start, sessions), panel, N_STATES)
        print_clock("pooled, orth, anchor START", other_clock)

        for k in SENSITIVITY_STATES:
            wf_k = walk_forward_context(panel, folds_end, train_from=start, n_states=k)
            paths_k = session_paths(wf_k, folds_end, sessions)
            _, clock_k = fold_rows(wf_k, folds_end, paths_k, panel, k)
            print_clock("sensitivity, orth, end", clock_k)
            out[(start_name, k)] = (wf_k, paths_k)
    return out


# ------------------------------------------------------------- C. session mapping


def session_mapping(wf: WalkForwardContext, folds: tuple[Fold, ...], paths: pd.Series,
                    sessions: pd.DatetimeIndex, panel: pd.DataFrame,
                    start_name: str) -> dict[str, int]:
    print(f"\n   --- K=4, orth, anchor end, context training starts: {start_name} ---\n")
    test_sessions = sessions[sessions > folds[0].train_cutoff]
    lacking = int(paths.isna().sum())
    exact = int(test_sessions.isin(panel.index).sum())
    print(f"   industry sessions in the test folds {len(test_sessions):,}; stamped on the "
          f"same date on the feature calendar {exact:,}; lacking a label {lacking}")
    train_rows = int((paths.index.get_level_values("segment") == "train").sum())
    print(f"   training segments: {train_rows:,} fold-session rows, same rule")

    one_hot = pd.DataFrame(np.eye(N_STATES), index=range(N_STATES),
                           columns=[f"s{i}" for i in range(N_STATES)])
    mismatched, own_first, raw_trap, aligned_trap = 0, 0, 0, 0
    pooled_raw = as_of(wf.labels, sessions)
    pooled_aligned = as_of(wf.aligned, sessions)
    for fold in folds:
        path = paths.xs(fold.number, level="fold").droplevel("segment")
        held = map_states(path, one_hot).to_numpy().argmax(axis=1)
        stamped_before = path.shift(1).to_numpy()
        test = path.index.isin(fold.test(sessions))
        mismatched += int((held[test] != stamped_before[test]).sum())
        first, last_train = fold.test(sessions)[0], fold.train(sessions)[-1]
        right = int(wf.model(fold.number).train_labels.loc[last_train])
        own_first += int(held[path.index.get_loc(first)] == right)
        if fold.number > 1:
            raw_trap += int(pooled_raw.loc[last_train] != right)
            mapping = wf.mappings[wf.numbers.index(fold.number)]
            aligned_trap += int(pooled_aligned.loc[last_train] != mapping[right])
    print("   per fold, training then test sessions of ONE model, lagged one session by")
    print("   protocol.map_states: test sessions whose held state is not the label stamped")
    print(f"   at the previous industry session: {mismatched}; first test sessions holding")
    print(f"   their own model's label of the last training session: {own_first} of {len(folds)}")
    print("   the trap avoided — lag the POOLED out-of-sample series instead: fold 1's first")
    print(f"   session has no label; at the {len(folds) - 1} later fold starts the held label")
    print(f"   comes from the previous model, differing from the right one in raw numbering "
          f"at {raw_trap}, in aligned numbering at {aligned_trap}")
    return {"lacking": lacking, "mismatched": mismatched, "own_first": own_first}


# ------------------------------------------------------------------ C2. cells


@dataclass(frozen=True)
class Cells:
    """The lock's cells on one stamped path: what qualifies and who abstains."""

    label: str
    k: int
    train: int
    test: int
    both: int
    per_fold: str
    abstaining: int
    test_sessions: int

    @property
    def floor(self) -> float:
        """The UNDECIDABLE floor: 10 cells at K = 4, half of the 5K cells otherwise."""
        return 10.0 if self.k == N_STATES else 5 * self.k / 2


def count_cells(label: str, paths: pd.Series, k: int) -> Cells:
    cells = qualifying_cells(paths)["qualifies"]
    train = cells.xs("train", level="segment")
    test = cells.xs("test", level="segment")
    both = train & test.reindex(train.index, fill_value=False)
    abstaining, test_sessions = 0, 0
    for fold in paths.index.get_level_values("fold").unique():
        path = paths.xs(fold, level="fold")
        lagged = path.droplevel("segment").shift(1)
        in_test = (path.index.get_level_values("segment") == "test")
        held = lagged.to_numpy()[in_test]
        allowed = set(train.xs(fold, level="fold").loc[lambda q: q].index)
        abstaining += int(sum(not (np.isfinite(h) and int(h) in allowed) for h in held))
        test_sessions += int(in_test.sum())
    per_fold = "/".join(str(int(both.xs(f, level="fold").sum()))
                        for f in paths.index.get_level_values("fold").unique())
    return Cells(label, k, int(train.sum()), int(test.sum()), int(both.sum()), per_fold,
                 abstaining, test_sessions)


def cells_section(objects: dict, sessions: pd.DatetimeIndex, panel: pd.DataFrame,
                  draws: int) -> list[Cells]:
    print(f"\n{RULE}\nC2. THE CELLS — the lock's minimum-cell rule, stamped paths, start 1995\n")
    print(f"   a state qualifies in a (fold, segment) block with at least {MIN_CELL_SESSIONS} "
          f"sessions and {MIN_CELL_EPISODES} episodes;")
    print("   an A-2 cell is a (fold, state) whose training AND test cells qualify; a test")
    print("   session abstains when its lagged state (fold path, lag 1) has no qualifying")
    print("   training cell. Counts only: no return is read.\n")
    folds = objects["folds"]
    smoothed = trailing_mode(objects[("1995", N_STATES)][1])
    rows = [count_cells(f"K={k}", objects[("1995", k)][1], k)
            for k in (N_STATES, *SENSITIVITY_STATES)]
    rows.append(count_cells(f"K={N_STATES} smoothed {SMOOTHING_WINDOW}", smoothed, N_STATES))
    print("   object              train cells  test cells  A-2 cells  per fold    floor  "
          "abstaining test sessions")
    for r in rows:
        n = len(folds) * r.k
        print(f"   {r.label:<19} {r.train:>5} of {n:<3}  {r.test:>4} of {n:<3} {r.both:>4} of "
              f"{n:<3}  {r.per_fold:<10} {r.floor:5.1f}  {r.abstaining:>5} of {r.test_sessions:,}"
              f" = {reading(r.abstaining / r.test_sessions):.1%}")

    wf = objects[("1995", N_STATES)][0]
    labelled = smoothed.dropna().astype(np.int64)
    years = sum(f.test_years for f in folds)
    print(f"\n   the smoothing sensitivity, K={N_STATES}: trailing {SMOOTHING_WINDOW}-session "
          "mode of each fold's stamped path, smallest label on ties, before the lag;")
    print(f"   the first {SMOOTHING_WINDOW - 1} sessions of each path have no label "
          f"({int(smoothed.isna().sum())} rows) and hold the fallback. Industry sessions,")
    print("   transitions inside each fold's segment; in-sample = mean of the five training")
    print("   segments' rates; eta2 of the pooled OOS labels in fold 1's numbering.")
    print("      path          OOS /yr  in-sample /yr  median ep.  mean ep.  eta2(log rv)")
    for name, path in (("stamped", objects[("1995", N_STATES)][1]), ("smoothed", labelled)):
        test = path.xs("test", level="segment")
        train = path.xs("train", level="segment")
        within = sum(transitions(test.xs(f.number, level="fold")) for f in folds)
        lengths = np.concatenate([run_lengths(test.xs(f.number, level="fold").to_numpy())[0]
                                  for f in folds])
        in_sample = np.mean([transitions(train.xs(f.number, level="fold"))
                             / span_years(train.xs(f.number, level="fold").index)
                             for f in folds])
        aligned = pd.concat([
            pd.Series(wf.mappings[wf.numbers.index(f.number)][
                test.xs(f.number, level="fold").to_numpy()],
                index=test.xs(f.number, level="fold").index) for f in folds])
        eta = eta_squared(aligned, as_of(panel[LOG_RV], aligned.index))
        print(f"      {name:<12} {reading(within / years):8.2f}  {reading(in_sample):13.2f}  "
              f"{reading(float(np.median(lengths))):10.0f}  {reading(float(lengths.mean())):8.1f}"
              f"  {reading(eta):12.3f}")
    groups = placebo_groups(labelled)
    uniform = matched_placebos(labelled, draws, groups=groups, method="uniform")
    check = check_placebos(labelled, uniform.draws, groups=groups)
    reading(check.max_occupancy_deviation)
    print(f"      uniform placebo, {draws:,} draws on (fold, segment) blocks: exact "
          f"{check.exact}, transition sd {check.transitions_sd:.1f}, max occupancy deviation "
          f"{check.max_occupancy_deviation:.4f}")
    return rows


# --------------------------------------------------------------- D. placebo


def placebo_section(objects: dict, starts: Iterable[str], draws: int) -> dict:
    print(f"\n{RULE}\nD.  PLACEBO FEASIBILITY — groups = (fold, segment), {draws:,} draws\n")
    print("   Each fold's in-sample TRAINING labels and its OUT-OF-SAMPLE test labels are a")
    print("   block of their own, on the industry sessions: m_ik is estimated on the first")
    print("   and applied to the second, so both need a null. 'tightest' is the largest")
    print("   ratio of one state's episodes to ceil(episodes/2), the most a join-free order")
    print("   allows: at 1.00 one state must sit on every other slot.\n")
    print("   start  K  swap: draws ok  blocks failing (draws of each block)          max try"
          "   uniform: exact  sd  occ.dev  s/1000   tightest")
    timing = {}
    for start_name in starts:
        for k in (N_STATES, *SENSITIVITY_STATES):
            _, paths = objects[(start_name, k)]
            groups = placebo_groups(paths)
            t0 = time.perf_counter()
            used = placebo_feasibility(paths, draws, groups=groups)
            t_swap = time.perf_counter() - t0
            ok = int((used > 0).all(axis=1).sum())
            failing = (used == 0).sum(axis=0)
            t0 = time.perf_counter()
            uniform = matched_placebos(paths, draws, groups=groups, method="uniform")
            t_uniform = time.perf_counter() - t0
            check = check_placebos(paths, uniform.draws, groups=groups)
            tight = 0.0
            for _, block in paths.groupby(groups, sort=False):
                _, s = run_lengths(block.to_numpy())
                tight = max(tight, np.bincount(s).max() / math.ceil(len(s) / 2))
            reading(check.max_occupancy_deviation)
            print(f"   {start_name:<6} {k}  {ok:>13d}  {fmt_counts(failing):<44} "
                  f"{int(used.max()):>4}   {str(check.exact):>14}  {check.transitions_sd:3.1f}"
                  f"  {check.max_occupancy_deviation:7.4f}  {t_uniform * 1000 / draws:6.1f}"
                  f"   {tight:.2f}")
            timing[(start_name, k)] = (t_swap, t_uniform, check.exact, ok)
    print("\n   blocks are ordered 1:train 1:test 2:train ... 5:test; s/1000 is the uniform")
    print("   construction's time for 1,000 draws of that object.")
    for start_name in starts:
        _, paths = objects[(start_name, N_STATES)]
        block_detail(paths, start_name, draws)
    return timing


def block_detail(paths: pd.Series, start_name: str, draws: int) -> None:
    """Per block: how much freedom a join-free order has, and where the swap fails."""
    groups = placebo_groups(paths)
    used = placebo_feasibility(paths, draws, groups=groups)
    print(f"\n   K=4, start {start_name}, block by block:")
    print("      block     sessions  episodes  per state      max/ceil   log10 orders"
          "   swap fails")
    for b, (name, block) in enumerate(paths.groupby(groups, sort=False)):
        _, s = run_lengths(block.to_numpy())
        per_state = np.bincount(s, minlength=N_STATES)
        orders = JoinFreeSequences(per_state).log_count / math.log(10)
        print(f"      {name:<9} {len(block):>8}  {len(s):>8}  {fmt_counts(per_state):<13} "
              f"{per_state.max():>3}/{math.ceil(len(s) / 2):<4}   {reading(orders):12.1f}"
              f"   {int((used[:, b] == 0).sum()):>5} of {draws:,}")


# ---------------------------------------------------------------------- main


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--draws", type=int, default=1_000)
    args = parser.parse_args()

    features = load_context_features()
    log_rv = load_log_realised_vol()
    panel = context_panel(features, log_rv)
    rv_level = as_of(np.exp(log_rv), panel.index)
    industry = inferential_sessions(start=pd.Timestamp("1900-01-01"))
    library = inferential_sessions()
    counts = pd.read_parquet(RAW / "panels" / "industry_49.parquet",
                             columns=["period", "value"]).groupby("period")["value"].count()
    if not (counts == 49).all():
        raise SystemExit("the industry panel is not complete on every session")

    checks = reproduction(panel, rv_level, industry, library)
    starts = {
        "1995": None,
        "ctx": panel.index[0],
    }
    print(f"\n   training starts compared below: '1995' = each fold's first session, "
          f"{library[0]:%Y-%m-%d} (reading ii); 'ctx' = the first date all twenty features "
          f"exist, {panel.index[0]:%Y-%m-%d} (reading i)")
    objects = declared_object(panel, library, starts)
    folds = objects["folds"]
    wf, paths = objects[("1995", N_STATES)]
    print(f"\n{RULE}\nC.  LABELS ON THE INDUSTRY CALENDAR, AND THE ONE-SESSION LAG")
    mapping = session_mapping(wf, folds, paths, library, panel, "1995")
    wf_ctx, paths_ctx = objects[("ctx", N_STATES)]
    mapping_ctx = session_mapping(wf_ctx, folds, paths_ctx, library, panel, "ctx")
    cells = cells_section(objects, library, panel, args.draws)
    timing = placebo_section(objects, starts, args.draws)

    print(f"\n{RULE}\nE.  RUNTIME\n")
    for name in starts:
        t_swap, t_uniform, _, _ = timing[(name, N_STATES)]
        print(f"   start {name:<5} one walk-forward fit (5 refits, K=4, n_init=20): "
              f"{objects[('time', name)]:.2f} s; {args.draws:,} swap attempts {t_swap:.1f} s; "
              f"{args.draws:,} uniform draws {t_uniform:.1f} s")

    print(f"\n{RULE}\nVERDICT ON THE DISCLOSED FIGURES\n")
    finite = all(c.finite for c in checks) and all(math.isfinite(r) for r in READINGS)
    if not finite:
        bad = [c.label for c in checks if not c.finite]
        print("   NO VERDICT: a reading is not finite", bad or "(declared-object reading)")
        return 1
    print(f"   {'figure':<46} {'disclosed':>9} {'measured':>11}")
    for c in checks:
        print(c.row())
    matches = sum(c.match for c in checks)
    print(f"\n   {matches} of {len(checks)} disclosed figures reproduce.")
    uniform_exact = all(v[2] for v in timing.values())
    swap_ok = {key: v[3] for key, v in timing.items()}
    print(f"   declared object: industry sessions lacking a label {mapping['lacking']} / "
          f"{mapping_ctx['lacking']}; lag mismatches {mapping['mismatched']} / "
          f"{mapping_ctx['mismatched']}")
    print("   cells (A-2 cells / floor; abstaining test sessions): " + ", ".join(
        f"{c.label} {c.both}/{c.floor:g}; {c.abstaining}" for c in cells))
    print(f"   placebo: uniform construction exact on every object: {uniform_exact}; "
          f"swap repair, draws completed of {args.draws:,}: "
          + ", ".join(f"{s} K={k} {n}" for (s, k), n in swap_ok.items()))

    print("\n   THE DECLARED OBJECT AGAINST THE PRE-LOCK ONE (K=4, orth, anchor end):")
    print(f"   {'quantity':<50} {'pre-lock':>9} {'1995':>8} {'ctx':>8}")
    c95, cctx = objects[("clock", "1995")], objects[("clock", "ctx")]
    rows = [
        ("transitions per year, out of sample", "8.31", c95.per_year, cctx.per_year, 2),
        ("transitions per year, in-sample (mean over folds)", "8.31", c95.train_per_year,
         cctx.train_per_year, 2),
        ("eta2 vs log rv, pooled aligned out-of-sample labels", "0.048", c95.eta2, cctx.eta2, 3),
        ("median episode, industry sessions", "8 (raw)", c95.median_length,
         cctx.median_length, 0),
        ("(fold, state) test cells with no episode", "n/a", c95.zero_test_cells,
         cctx.zero_test_cells, 0),
        ("fewest episodes in a (fold, state) training cell", "n/a", c95.min_train_cell,
         cctx.min_train_cell, 0),
    ]
    for label, before, a, b, decimals in rows:
        print(f"   {label:<50} {before:>9} {a:>8.{decimals}f} {b:>8.{decimals}f}")
    print("   pre-lock: one fit on the whole context sample, in-sample labels (§1 P3); the")
    print("   median episode is the raw clock's on the toy sample. Per-fold cells did not exist.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

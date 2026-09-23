"""Level C of the Two Sigma tree: the cadence channel (§12.10) and its PIT control (§8, 5).

The specification is `docs/PRESPEC_TWOSIGMA.md`, **LOCKED at commit 30f7d69**. This module
implements §6 "Level C", §12.10 (C-1 and C-2), the level-C lines of §13.1-§13.5 and, for a
C-1 that would otherwise PASS, the 18-feature point-in-time rebuild of §8 control 5. It
builds on the shared core :mod:`regime_lab.selection.tree` and changes nothing in it. A
deviation from the lock is an amendment in `docs/PROTOCOL_FREEZE.md`, never an edit here.

**What level C measures (§12.10).** The target is the control's held weights ``w*``
(after the volatility multiplier). A *cadence book* holds ``w*`` on its rebalance
sessions and keeps yesterday's weights otherwise; its interval ``h_t`` may depend on the
lagged context state. The twin is the state-blind book at h = 5; the conditional arm
takes per-state intervals chosen on each fold's training sessions to minimise turnover
under a pooled tracking budget equal to the twin's. C-1 is the held-turnover saving over
the twin on the test sessions, against the same rule re-run on 1,000 uniform placebo
draws, with a tracking gate and the W1 volatility witness. C-2 is its Sharpe translation,
a bound and never a PASS.

**Two implementations of one book.** :func:`cadence_book` is the literal rule, session by
session over the held weight vectors, the reference the engine is tested against. The
arms' returns (σ, the legs, the Sharpe of the trial row) come from ``tree.build_book`` on
the weights the engine holds, ``targeted=False``: ``w_t · (r_t − rf_t)`` net of 5 bp on
``|Δw|``, the return :func:`cadence_book` computes. :class:`CadenceSpace` is the same
book, precomputed for speed: a book holds ``w*`` of its last rebalance, so its tracking
``δ_t`` depends only
on ``t`` and the age ``a = t − r(t) ≤ 20``, and its ``Σ|Δw_t|`` only on ``r(t)`` and the
gap to the previous rebalance, ``≤ 21``. Both are read off one table
``dist[a, t] = Σ_n |w*_n,t − w*_n,t−a|`` computed once; a book is then an integer
schedule and two gathers, with no work over the 49 instruments. The rule, the null and
the real statistic ``S_C`` all go through this one engine, so a tie between the real
statistic and a placebo draw is a tie in the same code (§12.7 "ties count against").
Agreement between the two implementations is asserted at every instrument and reading.

**Blindness (§11, §13.1 step 2, §12.10 "Instrument").** :func:`instrument` builds only
the state-blind books (the twin and the menu) and the conditional arms of the placebo
draws; it reads the real partition's labels only to count cells. It returns the twin's
held turnover, mean δ and σ, the menu's turnover, the control's cap share, the counts,
the null's percentiles, ``MDE_C`` and ``S*``, and nothing else. It never builds the
conditional arm on the real partition and never computes ``h_fk``, ``S_C`` or the gate.
:func:`read` computes those, and refuses to run unless :func:`tree.verify_for_reading`
passes (thresholds committed and unmodified at HEAD, input SHA-256 and package versions
unchanged) and the recomputed instrument equals the committed one bitwise. The public
building blocks (:func:`conditional_arm`, :func:`fold_rule`) compute reading quantities:
on the real store they are called by :func:`read` only.

**The reading (§12.10, §13.1 step 4, §13.5).** :func:`read` returns the statistics, the
provisional verdicts at the Bonferroni bar 0.05/6 and the trial-row metrics; the final
verdicts after the Holm step come from :func:`lock_verdicts` with the rejected set. The
§8 control-5 rebuild on the 18-feature partition is computed whenever C-1 would pass
under either reading, so the final step never lacks it. A C-1 that is UNDECIDABLE before
its reading is not read and spends no row (§13.4).

**Non-finite readings (§13.4).** Any non-finite input of the rule makes that fold's rule
undefined and the arm's saving NaN; one NaN placebo draw makes ``p_C1`` NaN; a verdict
function returns UNDECIDABLE on any NaN. No draw is dropped, replaced or redrawn. The
instrument writes a non-finite value as the string ``non-finite``, never as a number.
"""

from __future__ import annotations

import json
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from regime_lab.config import ROOT
from regime_lab.selection import protocol, tree
from regime_lab.selection.tree import (
    BONFERRONI,
    COST_COLUMNS,
    DECIDING_BPS,
    FAIL,
    INPUT_FILES,
    N_DRAWS,
    PASS,
    PLACEBO_SEED,
    PRIMARY,
    THRESHOLDS_PATH,
    UNDECIDABLE,
    UNDERPOWERED,
    BookResult,
    Cells,
    NullSummary,
    Partition,
    PlaceboDraws,
    ReadingRefused,
    TreeData,
    Variant,
    apply_pit,
    build_book,
    cells,
    control_book,
    draw_map,
    encode_thresholds,
    level_verdict,
    null_summary,
    partition_paths,
    placebo_draws,
    placebo_verdict,
    sharpe,
    verify_for_reading,
    witness_paths,
)

NAN = float("nan")
PERIODS = protocol.PERIODS

#: §12.10: the menu of intervals, the twin's interval (the weekly twin) and the control's.
H_MENU: tuple[int, ...] = (1, 2, 3, 5, 8, 13, 21)
TWIN_H = 5
CONTROL_H = 1
MAX_H = max(H_MENU)
#: §12.10: 100 bisection steps after the doubling of λ.
BISECTION_STEPS = 100
#: §12.10 "Resolution and power": the saving worth 0.05 Sharpe at 5 bp.
S_STAR_SHARPE = 0.05
#: §12.10 "Level C's verdict is C-1's"; C-2 is a bound, entering Holm with p := 1.
LEVEL = "C"
C2_P = 1.0
#: The two implementations of a book must agree to this relative tolerance.
AGREEMENT = 1e-9
#: How the printout writes a non-finite value (the core's, as levels A and B): never as a
#: number (§13.4).
NON_FINITE = tree.NON_FINITE

_TWIN = H_MENU.index(TWIN_H)


# ---------------------------------------------------------------------------------------
# the cadence book, literally — §12.10 "Cadence books"
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False, repr=False)
class CadenceBook:
    """One cadence book of §12.10, on the target's sessions.

    ``held`` are the weights ``w_t`` (NaN before the start); ``rebalance`` flags the
    rebalance sessions; ``turnover`` is ``Σ_n |Δw_n,t|`` where ``w_t`` and ``w_{t−1}`` are
    both defined; ``delta`` is ``δ_t = Σ_n |w_n,t − w*_n,t|`` where both are defined;
    ``net`` (when excess returns were given) is ``w_t · (r_t − rf_t)`` net of ``bps`` on
    ``|Δw|`` through ``protocol.net_of_costs``, NaN where a weight or a return is missing.
    """

    held: pd.DataFrame
    rebalance: pd.Series
    turnover: pd.Series
    delta: pd.Series
    net: pd.Series | None
    start: pd.Timestamp

    def __repr__(self) -> str:
        return (f"CadenceBook({len(self.held):,} sessions from {self.start:%Y-%m-%d}, "
                f"{int(self.rebalance.sum()):,} rebalances)")


def _defined_rows(values: np.ndarray) -> np.ndarray:
    return np.isfinite(values).all(axis=1)


def _interval_array(h: pd.Series | np.ndarray | int, index: pd.Index, start: int) -> np.ndarray:
    n = len(index)
    if isinstance(h, int | np.integer):
        raw = np.full(n, float(h))
    elif isinstance(h, pd.Series):
        raw = h.reindex(index).to_numpy(float)
    else:
        raw = np.asarray(h, dtype=float)
    if raw.shape != (n,):
        raise ValueError("one interval per session")
    live = raw[start:]
    if not (np.isfinite(live).all() and (live >= 1).all() and (live == np.round(live)).all()):
        raise ValueError("every interval from the book's start must be an integer >= 1")
    out = np.ones(n, dtype=np.int64)
    out[start:] = live.astype(np.int64)
    return out


def cadence_book(
    target: pd.DataFrame,
    h: pd.Series | np.ndarray | int,
    *,
    excess: pd.DataFrame | None = None,
    bps: float = DECIDING_BPS,
) -> CadenceBook:
    """The cadence book of §12.10 "Cadence books", session by session.

    - The book starts on the first session on which ``w*`` (``target``, sessions x
      instruments) is defined on every instrument; that session is a rebalance, with
      counter 0.
    - It rebalances on *t* when the number of sessions since its last rebalance is
      ``≥ h_t``, so a shorter interval taking over after a longer count rebalances at once.
    - On a rebalance session it holds ``w*_t``; otherwise it keeps ``w_{t−1}``. A ``w*``
      row that is undefined after the start is held as it is (NaN), and the book's return
      is then missing: §13.4 makes that UNDECIDABLE, it is never filled.
    - Its return is ``w_t · (r_t − rf_t)`` net of ``bps`` on ``|Δw|`` (``excess`` holds
      ``r − rf`` per instrument, §12.3).

    ``h`` is an integer (a state-blind book: 1 is the control, 5 the twin) or one interval
    per session; intervals before the start are ignored.
    """
    values = target.to_numpy(float)
    defined = _defined_rows(values)
    if not defined.any():
        raise ValueError("w* is never defined")
    start = int(np.argmax(defined))
    hs = _interval_array(h, target.index, start)
    n = len(values)
    held = np.full_like(values, np.nan)
    rebalance = np.zeros(n, dtype=bool)
    held[start] = values[start]
    rebalance[start] = True
    last = start
    for t in range(start + 1, n):
        if t - last >= hs[t]:
            last = t
            rebalance[t] = True
            held[t] = values[t]
        else:
            held[t] = held[t - 1]
    held_df = pd.DataFrame(held, index=target.index, columns=target.columns)
    step = held_df.diff().abs()
    turnover = step.sum(axis=1).where(step.notna().all(axis=1)).rename("turnover")
    gap = (held_df - target).abs()
    delta = gap.sum(axis=1).where(gap.notna().all(axis=1)).rename("delta")
    net = None
    if excess is not None:
        product = held_df * excess.reindex(index=target.index, columns=target.columns)
        gross = product.sum(axis=1).where(product.notna().all(axis=1)).rename("book")
        net = protocol.net_of_costs(gross, held_df, bps=bps)
    return CadenceBook(held_df, pd.Series(rebalance, index=target.index, name="rebalance"),
                       turnover, delta, net, pd.Timestamp(target.index[start]))


# ---------------------------------------------------------------------------------------
# the same book, precomputed — the engine the rule, the null and S_C share
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class FoldSpan:
    """Fold *f*'s training and test sessions, as positions in the session calendar."""

    number: int
    train: np.ndarray
    test: np.ndarray


@dataclass(frozen=True, eq=False, repr=False)
class CadenceSpace:
    """Everything level C precomputes once per target (§12.10).

    ``dist[a, t] = Σ_n |w*_n,t − w*_n,t−a|`` for ``a`` in 0..21 (NaN where either row is
    undefined or ``t < a``). ``menu_*`` hold the state-blind books at each h of
    :data:`H_MENU` — computed once, since they read no label: ``menu_source[i, t]`` is the
    session whose ``w*`` the book at ``H_MENU[i]`` holds on *t* (−1 before the start),
    ``menu_tau`` its ``Σ|Δw_t|`` and ``menu_delta`` its ``δ_t``. ``test`` are the positions
    of the test sessions, which must be contiguous (§12.1: the five test windows tile
    2006-08-01 to 2026-07-31).
    """

    sessions: pd.DatetimeIndex
    columns: pd.Index
    target: np.ndarray
    start: int
    dist: np.ndarray
    test: np.ndarray
    folds: tuple[FoldSpan, ...]
    menu_source: np.ndarray
    menu_tau: np.ndarray
    menu_delta: np.ndarray

    def held(self, source: np.ndarray) -> pd.DataFrame:
        """The weights a book with this schedule holds: ``w*`` of its last rebalance."""
        values = np.full_like(self.target, np.nan)
        live = source >= 0
        values[live] = self.target[source[live]]
        return pd.DataFrame(values, index=self.sessions, columns=self.columns)

    def turnover(self, tau: np.ndarray) -> float:
        """Held turnover on the test sessions, restricted first (§12.6 "Inputs", §12.10).

        ``252 × mean Σ|Δw_t|`` over the test sessions after the first, skipping a session
        whose change is undefined — ``protocol.annual_turnover`` of the held weights
        restricted to the (contiguous) test sessions, so the entry change is not counted.
        """
        return PERIODS * _nanmean(tau[self.test[1:]])

    def mean_delta(self, delta: np.ndarray) -> float:
        """Mean ``δ_t`` over the test sessions on which it is defined (§12.10)."""
        return _nanmean(delta[self.test])

    @property
    def twin_turnover(self) -> float:
        return self.turnover(self.menu_tau[_TWIN])

    @property
    def twin_delta(self) -> float:
        return self.mean_delta(self.menu_delta[_TWIN])

    def __repr__(self) -> str:
        return (f"CadenceSpace({len(self.sessions):,} sessions, {len(self.columns)} "
                f"instruments, start {self.sessions[self.start]:%Y-%m-%d}, "
                f"{len(self.test):,} test sessions, menu {H_MENU})")


def _nanmean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else NAN


def _schedule(h: np.ndarray, start: int, stop: int | None = None) -> np.ndarray:
    """``source[t]``: the last rebalance at or before *t*, for *t* < ``stop``; −1 before
    ``start``. The rule of :func:`cadence_book`; a constant prefix of ``h`` is laid out
    arithmetically, which is what the session loop does on it."""
    stop = len(h) if stop is None else stop
    if not 0 <= start < stop <= len(h):
        raise ValueError("the book must start before it stops")
    marks = np.full(stop, -1, dtype=np.int64)
    first = int(h[start])
    changes = np.flatnonzero(h[start:stop] != first)
    prefix = start + int(changes[0]) if changes.size else stop
    marks[start:prefix:first] = np.arange(start, prefix, first)
    last = start + first * ((prefix - 1 - start) // first)
    rebalances = []
    append = rebalances.append
    for t, ht in enumerate(h[prefix:stop].tolist(), start=prefix):
        if t - last >= ht:
            last = t
            append(t)
    if rebalances:
        marks[rebalances] = rebalances
    return np.maximum.accumulate(marks)


def _book_tau(dist: np.ndarray, source: np.ndarray, start: int) -> np.ndarray:
    """``Σ_n |w_t − w_{t−1}|`` of the book with this schedule; NaN up to its start."""
    n = len(source)
    tau = np.full(n, np.nan)
    if n > start + 1:
        now = source[start + 1:]
        gap = now - source[start:-1]
        if gap.min() < 0 or gap.max() > MAX_H:
            raise AssertionError("a rebalance gap outside 0..21")  # pragma: no cover
        tau[start + 1:] = dist[gap, now]
    return tau


def _book_delta(dist: np.ndarray, source: np.ndarray, start: int) -> np.ndarray:
    """``δ_t = Σ_n |w_t − w*_t|`` of the book with this schedule; NaN before its start."""
    n = len(source)
    delta = np.full(n, np.nan)
    t = np.arange(start, n)
    age = t - source[start:]
    if age.min() < 0 or age.max() > MAX_H:
        raise AssertionError("a holding age outside 0..21")  # pragma: no cover
    delta[start:] = dist[age, t]
    return delta


def cadence_space(
    target: pd.DataFrame, sessions: pd.DatetimeIndex, folds: Sequence[Any]
) -> CadenceSpace:
    """Precompute :class:`CadenceSpace` for ``target`` (``w*`` on ``sessions``)."""
    sessions = pd.DatetimeIndex(sessions)
    if not target.index.equals(sessions):
        raise ValueError("the target must sit on the sessions, in order")
    values = np.ascontiguousarray(target.to_numpy(float))
    defined = _defined_rows(values)
    if not defined.any():
        raise ValueError("w* is never defined")
    start = int(np.argmax(defined))
    n = len(values)
    dist = np.full((MAX_H + 1, n), np.nan)
    for a in range(MAX_H + 1):
        dist[a, a:] = np.abs(values[a:] - values[:n - a]).sum(axis=1)
    spans = tuple(
        FoldSpan(f.number, sessions.get_indexer(f.train(sessions)),
                 sessions.get_indexer(f.test(sessions)))
        for f in folds
    )
    test = np.sort(np.concatenate([s.test for s in spans]))
    if test.size < 2 or (np.diff(test) != 1).any():
        raise ValueError("the test sessions must be contiguous (§12.1)")
    if test[0] <= start:
        raise ValueError("w* must be defined before the first test session")
    sources = np.stack([_schedule(np.full(n, h, dtype=np.int64), start) for h in H_MENU])
    return CadenceSpace(
        sessions=sessions,
        columns=target.columns,
        target=values,
        start=start,
        dist=dist,
        test=test,
        folds=spans,
        menu_source=sources,
        menu_tau=np.stack([_book_tau(dist, s, start) for s in sources]),
        menu_delta=np.stack([_book_delta(dist, s, start) for s in sources]),
    )


def target_weights(data: TreeData) -> pd.DataFrame:
    """§12.10 "Target": the control's held weights ``w*_t``, after the multiplier."""
    return control_book(data).book.weights


# ---------------------------------------------------------------------------------------
# the pooled-budget rule — §12.10 "Per-state intervals"
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Intervals:
    """The pooled-budget solution of one fold: ``h`` per qualifying state, λ* and the
    budget ``B_f`` with what the chosen intervals spend of it (§12.10)."""

    h: tuple[int, ...]
    lam: float
    budget: float
    spend: float


def pooled_budget_intervals(
    pi: np.ndarray,
    turnover: np.ndarray,
    tracking: np.ndarray,
    *,
    menu: Sequence[int] = H_MENU,
    twin: int = TWIN_H,
    steps: int = BISECTION_STEPS,
) -> Intervals | None:
    """§12.10's λ procedure on one fold's training quantities, exactly as written.

    ``pi[k]`` is ``π_k``; ``turnover[k, i]`` is ``T_k(menu[i])`` and ``tracking[k, i]``
    is ``δ_k(menu[i])``, for the qualifying states k.

    - ``B_f = Σ_k π_k δ_k(twin)``;
    - ``h_k(λ) = argmin_h [T_k(h) + λ δ_k(h)]``, exact ties to the larger h;
    - λ* = 0 if ``Σ_k π_k δ_k(h_k(0)) ≤ B_f``; otherwise ``λ_hi`` starts at 1 and doubles
      until the budget holds, then 100 bisection steps run between the last failing λ
      and ``λ_hi``, and the intervals are ``h_k(λ_hi)``.

    The spend and the budget go through one function, so the twin's own intervals spend
    exactly ``B_f``. Returns ``None`` if any input is not finite (the rule is undefined,
    §13.4). No qualifying state gives empty intervals.
    """
    pi = np.asarray(pi, dtype=float)
    cost = np.asarray(turnover, dtype=float)
    track = np.asarray(tracking, dtype=float)
    k = len(pi)
    if cost.shape != (k, len(menu)) or track.shape != (k, len(menu)):
        raise ValueError("one row per state and one column per interval of the menu")
    if k == 0:
        return Intervals((), 0.0, 0.0, 0.0)
    if not (np.isfinite(pi).all() and np.isfinite(cost).all() and np.isfinite(track).all()):
        return None
    rows = np.arange(k)
    last = len(menu) - 1

    def choose(lam: float) -> np.ndarray:
        return last - np.argmin((cost + lam * track)[:, ::-1], axis=1)

    def spend(idx: np.ndarray) -> float:
        return float(np.sum(pi * track[rows, idx]))

    budget = spend(np.full(k, list(menu).index(twin)))
    lam_hi = 0.0
    if spend(choose(0.0)) > budget:
        lam_lo, lam_hi = 0.0, 1.0
        while spend(choose(lam_hi)) > budget:
            lam_lo, lam_hi = lam_hi, 2.0 * lam_hi
            if not np.isfinite(lam_hi):
                raise RuntimeError("λ doubled past the largest float without meeting the budget")
        for _ in range(steps):
            mid = 0.5 * (lam_lo + lam_hi)
            if spend(choose(mid)) <= budget:
                lam_hi = mid
            else:
                lam_lo = mid
    idx = choose(lam_hi)
    return Intervals(tuple(int(menu[i]) for i in idx), lam_hi, budget, spend(idx))


@dataclass(frozen=True, eq=False)
class FoldRule:
    """The level-C rule of one fold, fitted on its training sessions (§12.10).

    ``states`` are fold *f*'s qualifying training states and ``intervals`` their
    ``h_fk`` (``None``: a non-finite input left the rule undefined). ``holds_twin`` says
    the fold trades the twin (every h = 5): no qualifying state, every ``h_fk`` at 5, or
    the training check — turnover not strictly below the twin's, or mean δ above it.
    ``check`` holds the candidate's and the twin's training turnover (per session) and
    mean δ, NaN when the check did not run. ``pi``, ``turnover`` (``T_k(h)``, states x
    menu) and ``tracking`` (``δ_k(h)``) are the rule's inputs. On the real partition every
    field is a reading quantity (§12.10 "Instrument").
    """

    fold: int
    states: tuple[int, ...]
    intervals: Intervals | None
    holds_twin: bool
    reason: str
    check: tuple[float, float, float, float] = (NAN, NAN, NAN, NAN)
    pi: np.ndarray | None = None
    turnover: np.ndarray | None = None
    tracking: np.ndarray | None = None

    @property
    def defined(self) -> bool:
        return self.intervals is not None

    def mapping(self) -> dict[int, int]:
        """``{state: h}`` as traded: 5 everywhere when the fold holds the twin."""
        if self.intervals is None:
            return {}
        h = (TWIN_H,) * len(self.states) if self.holds_twin else self.intervals.h
        return dict(zip(self.states, h, strict=True))


def _group_means(values: np.ndarray, groups: np.ndarray, k: int) -> np.ndarray:
    """Mean of each row of ``values`` (menu x sessions) per group (0..k−1) over its finite
    entries, as a (k, menu) array; NaN for an empty group. One ``bincount`` for all rows:
    each (row, group) bin still sums its sessions in session order."""
    m = values.shape[0]
    flat = values.ravel()
    bins = (groups[None, :] + k * np.arange(m)[:, None]).ravel()
    ok = np.isfinite(flat)
    sums = np.bincount(bins[ok], weights=flat[ok], minlength=k * m)
    counts = np.bincount(bins[ok], minlength=k * m)
    means = np.divide(sums, counts, out=np.full(k * m, np.nan), where=counts > 0)
    return means.reshape(m, k).T


def _lookup(
    states: Sequence[int], codes: np.ndarray, values: Sequence[int], fill: int
) -> np.ndarray:
    """An array ``a`` with ``a[code + 1]`` = the value of that state, ``fill`` otherwise
    (code −1, an unlabelled session, reads ``a[0] = fill``)."""
    top = max(int(codes.max(initial=-1)), max(states, default=-1))
    table = np.full(top + 2, fill, dtype=np.int64)
    if len(states):
        table[np.asarray(states, dtype=np.int64) + 1] = np.asarray(values, dtype=np.int64)
    return table


def fold_rule(
    space: CadenceSpace, fold_index: int, lag_train: np.ndarray, states: Sequence[int]
) -> FoldRule:
    """§12.10's per-state intervals for one fold, and its training check.

    ``lag_train`` is the lagged state of each of fold *f*'s training sessions along its own
    path (−1: none; the first training session never has one), ``states`` its qualifying
    training states (§12.4). On the training sessions that have a lagged state:
    ``π_k`` is the share whose lagged state is *k*; ``T_k(h)`` and ``δ_k(h)`` are the
    means of ``Σ|Δw_t|`` and ``δ_t`` of the state-blind book at *h* over those whose
    lagged state is *k*, over the sessions on which each is defined — the books run over
    the full path and are never restarted. Then :func:`pooled_budget_intervals` and the
    check: a cadence book at ``h_f,k(t)`` along the lagged training path (h = 5 for a
    missing or non-qualifying state), started with every book on the first session on
    which ``w*`` is defined, is read over the same training sessions; if its mean
    ``Σ|Δw_t|`` is not strictly below the twin's, or its mean δ exceeds the twin's, the
    fold holds the twin. A fold whose intervals are all 5 holds the twin without running
    the check, whose candidate would be the twin itself.
    """
    span = space.folds[fold_index]
    states = tuple(sorted(int(s) for s in states))
    if len(lag_train) != len(span.train):
        raise ValueError("one lagged state per training session")
    if not states:
        return FoldRule(span.number, states, Intervals((), 0.0, 0.0, 0.0), True,
                        "no qualifying training state")
    index = _lookup(states, lag_train, range(len(states)), -1)[lag_train + 1]
    labelled = lag_train >= 0
    k = len(states)
    pi = np.bincount(index[index >= 0], minlength=k) / max(int(labelled.sum()), 1)
    chosen = index >= 0
    at, groups = span.train[chosen], index[chosen]
    cost = _group_means(space.menu_tau[:, at], groups, k)
    track = _group_means(space.menu_delta[:, at], groups, k)
    intervals = pooled_budget_intervals(pi, cost, track)

    def outcome(ruled: Intervals | None, holds: bool, reason: str,
                check: tuple[float, float, float, float] = (NAN, NAN, NAN, NAN)) -> FoldRule:
        return FoldRule(span.number, states, ruled, holds, reason, check, pi, cost, track)

    if intervals is None:
        return outcome(None, False, "undefined: a non-finite input")
    if all(h == TWIN_H for h in intervals.h):
        return outcome(intervals, True, "every interval at the twin's")
    stop = int(span.train[-1]) + 1
    if stop <= space.start + 1:
        return outcome(None, False, "undefined: no book on training")
    h = np.full(stop, TWIN_H, dtype=np.int64)
    h[span.train] = _lookup(states, lag_train, intervals.h, TWIN_H)[lag_train + 1]
    source = _schedule(h, space.start, stop)
    read_at = span.train[labelled]
    check = (
        _nanmean(_book_tau(space.dist, source, space.start)[read_at]),
        _nanmean(_book_delta(space.dist, source, space.start)[read_at]),
        _nanmean(space.menu_tau[_TWIN, read_at]),
        _nanmean(space.menu_delta[_TWIN, read_at]),
    )
    if not np.isfinite(check).all():
        return outcome(None, False, "undefined: a non-finite check", check)
    turnover_c, delta_c, turnover_twin, delta_twin = check
    if not turnover_c < turnover_twin:
        return outcome(intervals, True, "training check: turnover not below the twin's", check)
    if delta_c > delta_twin:
        return outcome(intervals, True, "training check: tracking above the twin's", check)
    return outcome(intervals, False, "rule", check)


# ---------------------------------------------------------------------------------------
# paths, the conditional arm and its saving — §12.2, §12.10
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class PathLayout:
    """Where each fold's rows sit in a ``session_paths``-format Series (§12.2).

    ``rows[i]`` is the row range of the fold ``space.folds[i]`` and ``n_train[i]`` its
    number of training rows; the rows run training then test, in session order.
    """

    index: pd.MultiIndex
    rows: tuple[tuple[int, int], ...]
    n_train: tuple[int, ...]


def path_layout(paths: pd.Series, space: CadenceSpace) -> PathLayout:
    """Check that ``paths`` has the ``session_paths`` layout on ``space`` and locate it."""
    if list(paths.index.names) != ["fold", "segment", "session"]:
        raise ValueError("paths must be indexed by (fold, segment, session)")
    fold = paths.index.get_level_values("fold").to_numpy()
    segment = paths.index.get_level_values("segment").to_numpy()
    session = pd.DatetimeIndex(paths.index.get_level_values("session"))
    rows, n_train = [], []
    for span in space.folds:
        where = np.flatnonzero(fold == span.number)
        if where.size == 0 or (np.diff(where) != 1).any():
            raise ValueError(f"fold {span.number}: its rows must be present and contiguous")
        a, b = int(where[0]), int(where[-1]) + 1
        n = len(span.train)
        expected = space.sessions[np.concatenate([span.train, span.test])]
        if (b - a != n + len(span.test) or not session[a:b].equals(expected)
                or (segment[a:a + n] != "train").any() or (segment[a + n:b] != "test").any()):
            raise ValueError(f"fold {span.number}: its path must be its training then test "
                             "sessions (§12.2)")
        rows.append((a, b))
        n_train.append(n)
    return PathLayout(paths.index, tuple(rows), tuple(n_train))


def state_codes(paths: pd.Series) -> np.ndarray:
    """A stamped path as integer codes, −1 where no label is held."""
    values = paths.to_numpy(float)
    labelled = np.isfinite(values)
    if (values[labelled] < 0).any() or (values[labelled] != np.round(values[labelled])).any():
        raise ValueError("states must be non-negative integers")
    return np.where(labelled, values, -1).astype(np.int64)


def _lagged(codes: np.ndarray, rows: tuple[int, int]) -> np.ndarray:
    """One fold's path lagged one session inside the fold (§12.2): −1 on its first row."""
    a, b = rows
    lag = np.empty(b - a, dtype=np.int64)
    lag[0] = -1
    lag[1:] = codes[a:b - 1]
    return lag


def train_states(fold_cells: Cells, space: CadenceSpace) -> tuple[tuple[int, ...], ...]:
    """Each fold's qualifying training states (§12.4), in the order of ``space.folds``."""
    return tuple(tuple(sorted(fold_cells.train.get(s.number, frozenset()))) for s in space.folds)


@dataclass(frozen=True, eq=False, repr=False)
class ArmReading:
    """The conditional arm on one labelling, read on the test sessions (§12.10).

    ``saving`` is ``S = turnover(twin) − turnover(arm)`` on the test sessions, each
    restricted first, in x/yr; NaN when a fold's rule is undefined. ``fallback`` counts
    the test sessions of folds running their rule that hold h = 5 for want of a lagged
    state or a qualifying one; ``twin_sessions`` the test sessions of folds holding the
    twin. A reading quantity: never printed by the instrument for the real partition.
    """

    rules: tuple[FoldRule, ...]
    source: np.ndarray | None
    h: np.ndarray | None
    turnover: float
    mean_delta: float
    saving: float
    fallback: int
    twin_sessions: int

    def __repr__(self) -> str:
        return f"ArmReading({len(self.rules)} folds)"


def conditional_arm(
    space: CadenceSpace,
    layout: PathLayout,
    codes: np.ndarray,
    states: Sequence[Sequence[int]],
    *,
    detail: bool = True,
) -> ArmReading:
    """§12.10's conditional arm on one labelling (the real partition, a witness or a draw).

    Each fold's rule is fitted on its own lagged training path (:func:`fold_rule`). The
    arm then runs as one book over the full path: h = 5 on every session before the first
    test session; on fold *f*'s test sessions ``h_t = h_f,k(t)`` with *k(t)* the lagged
    state along fold *f*'s path (the first test session holds the label stamped on the
    last training session), h = 5 for a missing or non-qualifying state or a fold holding
    the twin. The counter carries across fold boundaries. ``codes`` are
    :func:`state_codes` of the stamped path (or a placebo draw's codes, same rows);
    ``states`` the qualifying training states per fold (:func:`train_states`).
    ``detail=False`` skips the tracking and the counts (the null needs the saving only).
    """
    rules, h = [], np.full(len(space.sessions), TWIN_H, dtype=np.int64)
    fallback = twin_sessions = 0
    for i, span in enumerate(space.folds):
        lag = _lagged(codes, layout.rows[i])
        n = layout.n_train[i]
        rule = fold_rule(space, i, lag[:n], states[i])
        rules.append(rule)
        if not rule.defined:
            continue
        if rule.holds_twin:
            twin_sessions += len(span.test)
            continue
        lag_test = lag[n:]
        h[span.test] = _lookup(rule.states, lag_test, rule.intervals.h, TWIN_H)[lag_test + 1]
        if detail:
            fallback += int((~np.isin(lag_test, rule.states)).sum())
    if not all(r.defined for r in rules):
        return ArmReading(tuple(rules), None, None, NAN, NAN, NAN, fallback, twin_sessions)
    source = _schedule(h, space.start)
    turnover = space.turnover(_book_tau(space.dist, source, space.start))
    mean_delta = space.mean_delta(_book_delta(space.dist, source, space.start)) if detail else NAN
    return ArmReading(tuple(rules), source, h, turnover, mean_delta,
                      space.twin_turnover - turnover, fallback, twin_sessions)


@dataclass(frozen=True, eq=False)
class _NullJob:
    space: CadenceSpace
    layout: PathLayout
    codes: np.ndarray
    states: tuple[tuple[int, ...], ...]


def _null_draw(shared: _NullJob, j: int) -> float:
    codes = shared.codes[:, j].astype(np.int64)
    return conditional_arm(shared.space, shared.layout, codes, shared.states,
                           detail=False).saving


def null_savings(
    space: CadenceSpace,
    layout: PathLayout,
    draws: PlaceboDraws,
    states: Sequence[Sequence[int]],
    *,
    workers: int | None = 1,
) -> np.ndarray:
    """``S^(j)`` on every placebo draw (§12.10 "Null"): the whole rule — λ, the
    intervals, the training check — re-run on the draw's training labels and the arm run
    on its test labels, through :func:`conditional_arm`, in the core's deterministic pool.

    ``states`` are the real path's qualifying training states: every uniform draw keeps
    each block's per-state sessions and episodes exactly, so it qualifies exactly the
    same cells (§12.4, §12.11), which ``draws.exact`` certifies.
    """
    if not draws.index.equals(layout.index):
        raise ValueError("the draws were not made on these paths")
    job = _NullJob(space, layout, draws.codes, tuple(tuple(s) for s in states))
    return np.asarray(draw_map(_null_draw, draws.n, job, workers=workers), dtype=float)


# ---------------------------------------------------------------------------------------
# verdicts — §12.10, §13.2, §8 control 5
# ---------------------------------------------------------------------------------------


def _finite(*values: float | None) -> bool:
    return all(v is not None and np.isfinite(v) for v in values)


def s_star(sigma_twin: float) -> float:
    """``S* = 0.05 × σ_twin × 10,000 / 5``: the saving worth 0.05 Sharpe at 5 bp (§12.10),
    ``σ_twin`` in decimal (7.88% gives 7.88x/yr). NaN on a non-finite σ."""
    return S_STAR_SHARPE * sigma_twin * 10_000.0 / DECIDING_BPS if _finite(sigma_twin) else NAN


def c2_bound(saving: float, sigma_twin: float, bps: float) -> float:
    """C-2: ``S_C × bps / 10,000 / σ_twin``, a Sharpe bound, never a PASS (§12.10)."""
    if not _finite(saving, sigma_twin) or not sigma_twin > 0:
        return NAN
    return saving * bps / 10_000.0 / sigma_twin


def c1_verdict(
    *,
    s_c: float,
    p: float,
    exact: bool,
    gate: bool | None,
    s_w: float,
    mde_c: float,
    s_star: float,
    leg_missing: bool,
    holm_rejected: bool | None = None,
) -> str:
    """C-1's verdict before the PIT line (§12.10 "Verdict"), in the lock's order.

    - UNDECIDABLE: ``S_C`` or any ``S^(j)`` not finite (``p`` is then NaN), the placebo
      check not exact, a leg missing on a test session, or a non-finite instrument or
      real reading (``MDE_C``, ``S*``, a mean δ of the gate, ``S_W``) (§13.4);
    - FAIL: the gate fails (the conditional arm's mean test δ exceeds the twin's);
    - the lock holds if ``p ≤ 0.01`` and Holm rejects (default: the provisional
      ``p ≤ 0.05/6`` of §13.3); if not, FAIL when ``MDE_C ≤ S*``, else NOT SHOWN;
    - DOMINATED: the lock holds and ``S_W ≥ S_C``;
    - PASS otherwise, to be passed through ``tree.apply_pit`` (the lock rebuilt on the
      18-feature partition must hold with its own witness and gate).
    """
    instrument_finite = _finite(mde_c, s_star)
    return placebo_verdict(
        statistic=s_c,
        p=p,
        exact=exact,
        dominated=bool(s_w >= s_c) if _finite(s_w, s_c) else None,
        holm_rejected=holm_rejected,
        gate_failed=None if gate is None else not gate,
        powered=bool(instrument_finite and mde_c <= s_star),
        other_undecidable=bool(leg_missing or not instrument_finite),
    )


def c2_verdict(bound5: float) -> str:
    """The C-2 row's verdict. The lock gives C-2 no verdict line, only "reported as a bound
    and never as a PASS" with ``p_C2 := 1`` (§6, §12.10, §13.3); this is its plainest
    reading: UNDECIDABLE on a non-finite bound, FAIL when the saving is not positive (as
    §12.6 line 3 reads Δ ≤ 0), UNDERPOWERED otherwise — a positive Sharpe bound that §6
    declared below any decidable bar (the control's whole cost, 0.18 to 0.23, is below
    0.338). It enters no level verdict (§12.10 "Level C's verdict is C-1's")."""
    if not _finite(bound5):
        return UNDECIDABLE
    return FAIL if bound5 <= 0 else UNDERPOWERED


def pit_variant(variant: Variant) -> Variant:
    """The §8 control-5 rebuild of ``variant`` (``tree.pit_variant``): the 18 features
    without ``fin_nfci`` and ``fin_nfci_chg13w``, the same K and smoothing (``n_init``,
    seed, folds and training start are fixed in ``tree.partition_paths``);
    :data:`tree.PIT18` for the primary, ``pit18[K=3]`` for a sensitivity. Refuses the
    rebuild itself."""
    return tree.pit_variant(variant)


# ---------------------------------------------------------------------------------------
# the instrument — §12.10 "Instrument", §13.1 steps 1-3
# ---------------------------------------------------------------------------------------


def section_name(variant: Variant) -> str:
    """The threshold file's section for level C at ``variant`` (``tree.section_name``):
    ``C`` for the primary, ``C:<name>`` for a §12.13 sensitivity. The PIT rebuild has none:
    its thresholds are computed at the reading, never committed (§8 control 5)."""
    return tree.section_name(LEVEL, variant)


def _check_variant(variant: Variant) -> None:
    if LEVEL not in variant.levels:
        raise ValueError(f"{variant.name} is not evaluated at level C (§7)")
    if variant.role == "control":
        raise ValueError(f"{variant.name} is the PIT rebuild (§8 control 5): it runs inside a "
                         "reading, never instrumented or read on its own")


def _resolve_draws(draws: PlaceboDraws | int | None, paths: pd.Series) -> PlaceboDraws:
    """§12.11's placebo on ``paths``: a count (``None``: 1,000) is drawn at seed 0; draws
    passed in must be this path's, uniform, at seed 0 (``tree.check_draws``)."""
    if draws is None or isinstance(draws, int | np.integer):
        return placebo_draws(paths, N_DRAWS if draws is None else int(draws), PLACEBO_SEED)
    if not isinstance(draws, PlaceboDraws):
        raise TypeError("draws must be a PlaceboDraws, a count or None")
    return tree.check_draws(draws, paths)


_clean = tree.printable


@dataclass(frozen=True, eq=False, repr=False)
class _Setup:
    control: BookResult
    space: CadenceSpace
    twin: BookResult


def _agree(a: float, b: float, what: str) -> None:
    both_nan = not (np.isfinite(a) or np.isfinite(b))
    if not both_nan and not abs(a - b) <= AGREEMENT * max(1.0, abs(a), abs(b)):
        raise RuntimeError(f"the two implementations of the book disagree on {what}: "
                           f"{a!r} against {b!r}")


def _setup(data: TreeData, target: pd.DataFrame | None = None) -> _Setup:
    """The control, the precomputed space of its held weights and the twin (h = 5) as a
    book on the traded path (§12.3, §12.10). Reads no label."""
    control = control_book(data)
    target = control.book.weights if target is None else target
    space = cadence_space(target, data.sessions, data.folds)
    twin = build_book(data, weights=space.held(space.menu_source[_TWIN]), targeted=False)
    _agree(twin.turnover, space.twin_turnover, "the twin's turnover")
    return _Setup(control, space, twin)


@dataclass(frozen=True, eq=False, repr=False)
class _Instrument:
    """What the instrument built. ``printout`` is all it shows (and what the threshold
    file records); the raw numbers are kept for the reading."""

    variant: Variant
    printout: dict[str, Any]
    null: np.ndarray
    summary: NullSummary
    sigma_twin: float
    s_star: float
    exact: bool
    undecidable: tuple[str, ...]
    layout: PathLayout
    states: tuple[tuple[int, ...], ...]


def _instrument(
    setup: _Setup,
    variant: Variant,
    partition: Partition,
    draws: PlaceboDraws,
    workers: int | None,
) -> _Instrument:
    space, twin = setup.space, setup.twin
    paths = partition.paths
    fold_cells = cells(paths)
    layout = path_layout(paths, space)
    states = train_states(fold_cells, space)
    null = null_savings(space, layout, draws, states, workers=workers)
    summary = null_summary(null)
    sigma_twin = twin.sigma
    star = s_star(sigma_twin)
    powered = bool(_finite(summary.mde_c, star) and summary.mde_c <= star)
    non_finite = int((~np.isfinite(null)).sum())
    reasons = []
    if not draws.exact:
        reasons.append("the placebo check is not exact (§12.11)")
    if non_finite:
        reasons.append(f"{non_finite} of {draws.n} placebo draws of S are not finite (§13.4)")
    if twin.missing:
        reasons.append(f"the twin has no net return on {twin.missing} test sessions (§13.4)")
    readings = (("the twin's turnover", space.twin_turnover), ("the twin's mean δ",
                space.twin_delta), ("σ_twin", sigma_twin), ("S*", star), ("MDE_C", summary.mde_c))
    bad = [name for name, value in readings if not _finite(value)]
    if bad:
        reasons.append(f"non-finite instrument reading: {', '.join(bad)} (§13.4)")
    printout = {
        "variant": variant.name,
        "K": int(variant.K),
        "placebo": {"method": draws.method, "draws": draws.n, "seed": draws.seed,
                    "exact": bool(draws.exact)},
        "cells": {
            "train": fold_cells.n_train,
            "train_per_fold": "/".join(str(len(s)) for s in states),
            "abstaining": fold_cells.abstaining,
            "test_sessions": fold_cells.test_sessions,
        },
        "twin": {
            "h": TWIN_H,
            "turnover": space.twin_turnover,
            "mean_delta": space.twin_delta,
            "sigma": sigma_twin,
            "missing": twin.missing,
        },
        "menu_turnover": {str(h): space.turnover(space.menu_tau[i])
                          for i, h in enumerate(H_MENU)},
        "control_cap_share": setup.control.cap_share,
        "null": {**summary.as_dict(), "non_finite": non_finite},
        "MDE_C": summary.mde_c,
        "S_star": star,
        "powered": powered,
        "undecidable": reasons,
    }
    return _Instrument(variant, _clean(printout), null, summary, sigma_twin, star,
                       bool(draws.exact), tuple(reasons), layout, states)


def instrument(
    data: TreeData,
    variant: Variant = PRIMARY,
    draws: PlaceboDraws | int | None = None,
    *,
    partition: Partition | None = None,
    workers: int | None = 1,
) -> dict[str, Any]:
    """Level C's instrument (§12.10 "Instrument", §13.1 steps 1-2): the thresholds and
    the only printout allowed before the reading.

    Builds the state-blind books (the twin at h = 5 and the menu) and the conditional arms
    of the placebo draws, and nothing on the real partition but its cell counts. Returns
    a JSON-serialisable mapping, every entry of which may be printed and committed as the
    level-C section of the threshold file (:func:`section_name`):

    - ``placebo``: method, draws, seed, and whether ``check_placebos`` is exact (§12.11);
    - ``cells``: qualifying training cells, per fold, abstaining test sessions (labels);
    - ``twin``: its held turnover and mean δ on the test sessions, ``σ_twin`` (the realised
      sd of its 5 bp net daily excess returns there) and its test sessions without one;
    - ``menu_turnover``: the held turnover of each state-blind book, test sessions;
    - ``control_cap_share``: the share of test sessions on which the control's cap binds;
    - ``null``: q20, q50, q95, q99, ``q99 − q50`` and ``q99 − q20`` of ``S^(·)``, and the
      count of non-finite draws;
    - ``MDE_C = q99 − q20``, ``S_star = 0.05 × σ_twin × 10,000 / 5`` and ``powered``
      (``MDE_C ≤ S*``: a C-1 that does not hold then reads FAIL, otherwise NOT SHOWN);
    - ``undecidable``: why C-1 is UNDECIDABLE before its reading (§13.4), if it is.

    A non-finite value is written ``non-finite``, never as a number. It computes no
    ``h_fk``, ``S_C``, gate, turnover or δ of the conditional arm on the real partition,
    no Sharpe and no mean.

    ``draws``: the variant's placebo (``tree.placebo_draws`` on its paths; a count or
    ``None`` builds it, seed 0, 1,000 by default). ``partition``: the variant's paths
    (default ``tree.partition_paths``). Refuses a variant not evaluated at level C and the
    PIT rebuild. The result is bitwise the same whatever ``workers`` is; the default runs
    in-process, since a draw takes about 7 ms at the real sample's size (measured on
    synthetic data of that size, 8,090 sessions x 49 industries), less than a spawned
    pool costs to start.
    """
    _check_variant(variant)
    with threadpool_limits(limits=1):
        partition = partition_paths(data, variant) if partition is None else partition
        draws = _resolve_draws(draws, partition.paths)
        return _instrument(_setup(data), variant, partition, draws, workers).printout


# ---------------------------------------------------------------------------------------
# the reading — §12.10, §13.1 step 4, §13.5
# ---------------------------------------------------------------------------------------


def _encoded(value: Any) -> Any:
    return json.loads(json.dumps(encode_thresholds(value), sort_keys=True))


def _arm_book(data: TreeData, space: CadenceSpace, arm: ArmReading) -> BookResult | None:
    if arm.source is None:
        return None
    book = build_book(data, weights=space.held(arm.source), targeted=False)
    _agree(book.turnover, arm.turnover, "the conditional arm's turnover")
    return book


def _gate(arm: ArmReading, space: CadenceSpace) -> bool | None:
    if not _finite(arm.mean_delta, space.twin_delta):
        return None
    return bool(arm.mean_delta <= space.twin_delta)


def _rules_report(arm: ArmReading) -> dict[str, Any]:
    return {
        str(rule.fold): {
            "states": list(rule.states),
            "h": {str(k): v for k, v in rule.mapping().items()},
            "lambda": rule.intervals.lam if rule.intervals is not None else NAN,
            "holds_twin": rule.holds_twin,
            "reason": rule.reason,
        }
        for rule in arm.rules
    }


def _witness_saving(data: TreeData, space: CadenceSpace, K: int) -> float:
    """``S_W``: the same rule on witness W1 of §12.8 at ``K`` bins, its cells qualified on
    its own stamped path, the same λ procedure per fold (§12.10 "Volatility witness")."""
    witness = witness_paths(data, "W1", K)
    arm = conditional_arm(space, path_layout(witness, space), state_codes(witness),
                          train_states(cells(witness), space), detail=False)
    return arm.saving


def _c1_reading(
    data: TreeData, setup: _Setup, partition: Partition, inst: _Instrument, *, s_w: float
) -> dict[str, Any]:
    """C-1 on one partition against its own null: the rule per fold, the conditional arm,
    ``S_C``, ``p_C1``, the gate, and the inputs of :func:`c1_verdict` (§12.10). ``s_w`` is
    the witness's saving (:func:`_witness_saving`), reported after ``S_C``; nothing is
    printed while a reading computes."""
    space = setup.space
    arm = conditional_arm(space, inst.layout, state_codes(partition.paths), inst.states)
    book = _arm_book(data, space, arm)
    leg_missing = bool(setup.twin.missing or book is None or book.missing)
    p = protocol.placebo_p_value(arm.saving, inst.null)
    gate = _gate(arm, space)
    causes = []
    if not _finite(arm.saving):
        undefined = [f"fold {r.fold}: {r.reason}" for r in arm.rules if not r.defined]
        causes.append("S_C is not finite (" + "; ".join(undefined or ["no finite turnover"])
                      + ")")
    elif not _finite(p):
        causes.append("a placebo draw of S is not finite")
    if gate is None:
        causes.append("a mean δ of the gate is not finite")
    if not _finite(s_w):
        causes.append("S_W on witness W1 is not finite")
    if leg_missing:
        causes.append("a leg is missing on a test session")
    inputs = {"s_c": arm.saving, "p": p, "exact": inst.exact, "gate": gate, "s_w": s_w,
              "mde_c": inst.summary.mde_c, "s_star": inst.s_star, "leg_missing": leg_missing}
    return {
        "read": True,
        "S_C": arm.saving,
        "p": p,
        "placebo_pct": protocol.placebo_percentile(arm.saving, inst.null),
        "gate": gate,
        "turnover": {"twin": space.twin_turnover, "conditional": arm.turnover},
        "mean_delta": {"twin": space.twin_delta, "conditional": arm.mean_delta},
        "S_W": s_w,
        "rules": _rules_report(arm),
        "fallback_sessions": arm.fallback,
        "twin_sessions": arm.twin_sessions,
        "leg_missing": leg_missing,
        "sigma": {"twin": setup.twin.sigma, "conditional": book.sigma if book else NAN},
        "sharpe": sharpe(book.net5) if book is not None else NAN,
        "q99": inst.summary.q99,
        "MDE_C": inst.summary.mde_c,
        "S_star": inst.s_star,
        "undecidable": causes,
        "verdict_inputs": inputs,
    }


def _would_pass(inputs: Mapping[str, Any]) -> bool:
    """Would C-1 pass at the provisional Bonferroni reading or under a Holm rejection?"""
    return PASS in (c1_verdict(**inputs), c1_verdict(**inputs, holm_rejected=True))


def _pit_rebuild(
    data: TreeData,
    setup: _Setup,
    variant: Variant,
    *,
    s_w: float,
    pit: tuple[Partition, PlaceboDraws] | None,
    workers: int | None,
) -> dict[str, Any]:
    """§8 control 5 at C-1: the lock rebuilt on the 18-feature partition (same K, n_init,
    seed, folds, training start), its own 1,000 uniform draws (seed 0), its thresholds by
    the same procedure (``MDE′_C`` on its own null; ``S*`` is the twin's, which reads no
    partition), its own gate, and the W1 witness at the same K (W1 reads no feature, so
    ``S_W`` is the primary's). Its p is compared with 0.05/6 and never enters Holm (§8).
    Its thresholds are computed here and never committed."""
    rebuilt = pit_variant(variant)
    if pit is None:
        partition = partition_paths(data, rebuilt)
        draws = placebo_draws(partition.paths, N_DRAWS, PLACEBO_SEED)
    else:
        partition, draws = pit
        draws = _resolve_draws(draws, partition.paths)
    inst = _instrument(setup, rebuilt, partition, draws, workers)
    lock = _c1_reading(data, setup, partition, inst, s_w=s_w)
    verdict = c1_verdict(**lock["verdict_inputs"], holm_rejected=None)
    return {
        "variant": rebuilt.name,
        "instrument": inst.printout,
        **{k: lock[k] for k in ("S_C", "p", "placebo_pct", "gate", "turnover", "mean_delta",
                                "rules", "leg_missing", "undecidable")},
        "verdict": verdict,
    }


def lock_verdicts(
    reading: Mapping[str, Any], rejected: Collection[str] | None = None
) -> dict[str, str]:
    """C-1's, C-2's and level C's verdicts from a reading (§12.10, §8 control 5, §13.2).

    ``rejected`` is the set of primaries Holm rejects (§13.3), known once C is read;
    ``None`` gives the provisional Bonferroni reading ``p ≤ 0.05/6`` that the results
    file states and the trial rows log (§13.5). A sensitivity never enters Holm and
    refuses ``rejected``. A would-be PASS goes through ``tree.apply_pit`` with the
    reading's PIT rebuild, which :func:`read` computed whenever C-1 could pass under
    either reading. A C-1 that was UNDECIDABLE before its reading (§13.4) stays so, and
    C-2, its translation, with it.
    """
    if rejected is not None and reading["role"] != "primary":
        raise ValueError("Holm runs over the six primaries only (§13.3); a sensitivity reads "
                         "its locks at 0.05/6")
    c1 = reading["C-1"]
    if not c1["read"]:
        return {"C-1": UNDECIDABLE, "C-2": UNDECIDABLE, LEVEL: UNDECIDABLE}
    holm = None if rejected is None else "C-1" in rejected
    raw = c1_verdict(**c1["verdict_inputs"], holm_rejected=holm)
    pit = reading.get("pit")
    verdict = apply_pit(raw, None if pit is None else pit["verdict"])
    return {"C-1": verdict, "C-2": c2_verdict(reading["C-2"]["bound"][_key(DECIDING_BPS)]),
            LEVEL: level_verdict([verdict])}


def _key(bps: float) -> str:
    return str(float(bps))


def trial_rows(
    reading: Mapping[str, Any],
    variant: Variant,
    verdicts: Mapping[str, str] | None = None,
    *,
    primary_level: str | None = None,
) -> dict[str, dict[str, Any]]:
    """The §13.5 metrics of the level-C rows of one variant, keyed by ``test``, ready for
    ``tree.log_trial(test, variant, **row)``. Nothing is logged here.

    - Primary: ``C-1`` (sharpe = the conditional arm's 5 bp net Sharpe, delta = ``S_C`` in
      x/yr, threshold = the null's q99, which §6 says ``S_C`` must lie above, p = ``p_C1``,
      placebo_pct = ``S_C``'s percentile in the null) and ``C-2`` (sharpe NaN, delta = the
      bound at 5 bp, threshold NaN, p = 1, the 10 and 20 bp bounds in the note).
    - K and smoothing rows: one row ``C`` with C-1's metrics, ``p2`` NaN, verdict = the
      level verdict (§13.2). If ``primary_level`` is PASS and delta ≤ 0, the note says
      PASS (not robust) (§12.13).
    - The PIT rebuild spends no row; a C-1 UNDECIDABLE before its reading spends none
      (§13.4), and its translation C-2 none either.

    ``verdicts`` default to the reading's provisional ones, which §13.5 logs. An
    UNDECIDABLE row's note carries the reading that caused it (§13.4).
    """
    verdicts = reading["verdicts"] if verdicts is None else verdicts
    c1, c2 = reading["C-1"], reading["C-2"]
    if variant.role == "control" or not c1["read"]:
        return {}
    n = int(reading["test_sessions"])
    row = {"sharpe": c1["sharpe"], "delta": c1["S_C"], "threshold": c1["q99"], "p": c1["p"],
           "placebo_pct": c1["placebo_pct"], "sessions": n}

    def note(verdict: str, base: str) -> str:
        causes = c1["undecidable"] if verdict == UNDECIDABLE else []
        return " | ".join([base, *causes])

    c1_note = "delta = S_C (x/yr), threshold = the null's q99, p = p_C1 (§12.10)"
    if variant.role == "primary":
        bounds = ", ".join(f"{b} bp {c2['bound'][_key(b)]!r}" for b in COST_COLUMNS)
        return {
            "C-1": {**row, "verdict": verdicts["C-1"], "note": note(verdicts["C-1"], c1_note)},
            "C-2": {"sharpe": NAN, "delta": c2["bound"][_key(DECIDING_BPS)], "threshold": NAN,
                    "p": C2_P, "placebo_pct": NAN, "sessions": n, "verdict": verdicts["C-2"],
                    "note": note(verdicts["C-2"], "bound S_C x bps / 10,000 / sigma_twin, "
                                 f"never a PASS, p := 1 (§12.10): {bounds}")},
        }
    base = note(verdicts[LEVEL], c1_note + ", p2 NaN at C (§13.5)")
    delta = c1["S_C"]
    if primary_level == PASS and _finite(delta) and delta <= 0:
        base += " | PASS (not robust): delta <= 0 at this sensitivity (§12.13)"
    return {LEVEL: {**row, "p2": NAN, "verdict": verdicts[LEVEL], "note": base}}


def read(
    data: TreeData,
    variant: Variant = PRIMARY,
    draws: PlaceboDraws | int | None = None,
    thresholds: Mapping[str, Any] | None = None,
    *,
    partition: Partition | None = None,
    pit: tuple[Partition, PlaceboDraws] | None = None,
    workers: int | None = 1,
    root: Path = ROOT,
    path: Path = THRESHOLDS_PATH,
    inputs: Sequence[str] = INPUT_FILES,
) -> dict[str, Any]:
    """Level C's reading (§12.10, §13.1 step 4, §8 control 5, §13.5). **Refuses to run**.

    1. ``tree.ReadingRefused`` unless ``tree.verify_for_reading`` passes for
       :func:`section_name`'s section: the threshold file is tracked and unmodified at
       HEAD, the input files' SHA-256 and the package versions are those it records, and
       the instrument recomputed here equals the committed one bitwise. ``thresholds``,
       if given, must equal the committed section too, and is checked first, before any
       computation. No statistic is computed before these checks pass.
    2. A C-1 that is UNDECIDABLE before its reading (the instrument's ``undecidable``,
       §13.4) is not read: no statistic of the real partition is computed and no row is
       spent.
    3. Otherwise, on the real partition: the rule per fold (``h_fk``, reported), the
       conditional arm, ``S_C``, ``p_C1 = placebo_p_value(S_C, S^(·))``, the gate, the
       arm's σ and 5 bp net Sharpe; then ``S_W``, the same rule on witness W1 at the
       variant's K. If C-1 would pass at the provisional reading or under a Holm
       rejection, the PIT rebuild (``pit`` may pass its partition and draws). C-2's bound
       at 5, 10 and 20 bp beside the control's whole cost (turnover x bps / 10,000 / its
       realised sd, §6, §12.3), with ``p_C2 = 1``.

    Returns the statistics, the provisional verdicts (``verdicts``; final ones after Holm:
    :func:`lock_verdicts` with the rejected set), the Holm p-values and the trial-row
    metrics (:func:`trial_rows`, provisional). Nothing is logged here.
    """
    _check_variant(variant)
    section = section_name(variant)
    if thresholds is not None:
        verify_for_reading(thresholds, section=section, path=path, root=root, inputs=inputs)
    with threadpool_limits(limits=1):
        partition = partition_paths(data, variant) if partition is None else partition
        draws = _resolve_draws(draws, partition.paths)
        setup = _setup(data)
        inst = _instrument(setup, variant, partition, draws, workers)
    stored = verify_for_reading(inst.printout, section=section, path=path, root=root,
                                inputs=inputs)
    if thresholds is not None and _encoded(thresholds) != _encoded(stored):
        raise ReadingRefused("the thresholds passed are not the committed ones")
    with threadpool_limits(limits=1):
        return _reading(data, setup, variant, partition, inst, pit=pit, workers=workers)


def _reading(
    data: TreeData,
    setup: _Setup,
    variant: Variant,
    partition: Partition,
    inst: _Instrument,
    *,
    pit: tuple[Partition, PlaceboDraws] | None,
    workers: int | None,
) -> dict[str, Any]:
    """The reading proper; :func:`read` calls it only after the verification."""
    control = setup.control
    control_cost = {_key(b): (control.turnover * b / 10_000.0 / control.sigma
                              if _finite(control.turnover, control.sigma) and control.sigma > 0
                              else NAN) for b in COST_COLUMNS}
    reading: dict[str, Any] = {
        "level": LEVEL,
        "variant": variant.name,
        "role": variant.role,
        "instrument": inst.printout,
        "test_sessions": int(len(data.test_sessions)),
        "bonferroni": BONFERRONI,
    }
    if inst.undecidable:
        reading["C-1"] = {"read": False, "S_C": NAN, "p": NAN,
                          "undecidable": list(inst.undecidable), "verdict_inputs": None}
        reading["C-2"] = {"read": False, "bound": {_key(b): NAN for b in COST_COLUMNS},
                          "control_cost": control_cost, "p": C2_P}
        reading["pit"] = None
    else:
        s_w = _witness_saving(data, setup.space, variant.K)
        c1 = _c1_reading(data, setup, partition, inst, s_w=s_w)
        c1["verdict_before_pit"] = c1_verdict(**c1["verdict_inputs"])
        reading["C-1"] = c1
        sigma_twin = setup.twin.sigma
        reading["C-2"] = {
            "read": True,
            "bound": {_key(b): c2_bound(c1["S_C"], sigma_twin, b) for b in COST_COLUMNS},
            "control_cost": control_cost,
            "p": C2_P,
        }
        reading["pit"] = (_pit_rebuild(data, setup, variant, s_w=s_w, pit=pit, workers=workers)
                          if _would_pass(c1["verdict_inputs"]) else None)
    reading["verdicts"] = lock_verdicts(reading)
    reading["holm_p"] = {"C-1": reading["C-1"]["p"], "C-2": C2_P}
    reading["trial_rows"] = trial_rows(reading, variant)
    return reading

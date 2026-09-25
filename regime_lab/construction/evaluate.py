"""Evaluation machinery of the Bridgewater tree: level A, its placebos, B2's map, level C.

The draft is `pilotage/plans_de_recherche/bridgewater/PRESPEC_BRIDGEWATER.md`
(2026-09-22); every § below is that draft's. The lock text, `docs/PRESPEC_BRIDGEWATER.md`,
settles the open choices listed here (its §12) and wins wherever the two differ; the
OC-* defaults below record the state of this module when it was built, not the lock's
decisions. The declared objects are assembled by `construction.declared` (§12.4 of the
lock), and the lock's inference helpers live in `construction.inference`. Nothing in
this module reads `data/`. Every function takes returns and labels as arguments, and the
tests run it on synthetic data only.

**Not the lock's level C.** :func:`cadence_book`, :func:`level_c_compare`,
:func:`level_c_null` and :func:`placebo_rebalances` implement a level C with a constant
target (Two Sigma's C-1 shape). The lock's level C rebalances `sleeves.blind_book` on
stamp subsets and calls none of them. :func:`placebo_rebalances` returns sessions that
are already lagged, so passing them to `sleeves.blind_book` would lag twice.

**Blindness.** Until the lock is committed, nothing here may be run on a real return
series together with the real quadrant labels, or the real B1 labels, except through
the null. :func:`null_statistics` and :func:`null_reductions` draw content-free
partitions with the real clock and occupancy, and :class:`NullDistribution` summarises
them. The functions `LevelA.run`, `reduction_bootstrap`, `reduction_se`,
`exposure_matched`, `p3_reductions` and `level_c_compare` all compute cell-conditional
quantities when they are given real labels. :func:`d_statistic` and
:func:`cell_variances` do too.

**What the draft fixes.**
- The statistic `D = log var(worst cell) − log var(best cell)` (§3 A).
- A blind leg (unconditional risk parity) against a balanced leg, whose cell
  covariances are estimated on training folds only (§3 A).
- Four PASS conditions: a reduction of at least 0.204, the 95th percentile of P1,
  survival of P2, and sign stability across cost columns (§3 A).
- The placebos P1, P2 and P3 (§6), B2's map (§3 B2), level C (§3 C), and Šidák over
  five tests (§3).

**What it leaves open.** Each item names the default taken here. None of these
defaults is a decision: the lock must choose.

*Level A*
- OC-A1 **The series D is read on.** The default is the volatility-targeted book, the
  object §2 declares. The alternative is the unscaled constant-weight book, the object
  the balanced weights are fitted to. The target equalises variance through time, so it
  compresses D on both legs; hence the witness (OC-W).
- OC-A2 **"Worst" and "best".** The default (`extremes="own"`) takes each leg's own
  highest- and lowest-variance cell, so `D ≥ 0`. The alternative (`extremes="blind"`)
  picks the two cells on the blind leg's test variances and reads both legs on those
  cells. A third reading picks them on training. D is a max − min, so it is biased
  upward and non-smooth near ties.
- OC-A3 **Variance centred or about zero.** The default is centred, the literal
  "variance" and "covariance". `about_zero=True` keeps every cell mean out, as the Two
  Sigma lock did at §12.8, and makes R2's decomposition of total variance exact. The
  difference is of the order of the squared daily Sharpe.
- OC-A4 **The balanced reading.**
  - *R1* `equal_variance`: `w'Σ_k w` equal across k. This is the reading consistent
    with D.
  - *R2* `equal_share`: `π_k w'Σ_k w` equal. This is §3 A's literal text, "the four
    cells contribute equal shares of portfolio variance".
  - R2 sets the training cell variances proportional to `1/π_k`. At the draft's
    occupancy (0.143 to 0.308) its **training optimum has D = log(0.308/0.143) = 0.767**.
    R2 is therefore built to widen D, not to shrink it. `occupancy_weighted=True` gives
    the D that matches R2: the dispersion of `log π_k v_k`.
- OC-A5 **The objective when exact balance is infeasible.** The default is least
  squares on the log cell variances. The alternative is minimax, the training D.
  Infeasibility is the usual case, not an edge. On the tests' planted five-sleeve
  panel, the long-only R1 solution sits in a corner (two sleeves at zero) and its
  training D stays near 0.3.
- OC-A6 **The tie-break among exact equalisers.** R1 has three equalities over four
  free weights, so a feasible solution is a curve, not a point. The default is the
  point nearest the blind weights: a penalty `TIE_BREAK × ‖w − w_blind‖²`, with a
  deterministic start at `w_blind`, solved by SLSQP. The objective is not convex, so
  the solution is a local optimum from that start. Alternatives: minimum pooled
  variance, or maximum entropy.
- OC-A7 **The blind leg.**
  - The default is ERC on the full training covariance (`protocol.risk_parity_mix`).
    The alternative is inverse volatility.
  - The default estimates it on every finite training session, whatever its label, so
    it is one fixed leg across the quadrant, the witness and every placebo draw.
  - The sibling module `construction.sleeves` re-estimates its blind leg at every
    rebalance on a trailing 252 sessions. The balanced leg's cells must pool the whole
    training fold. The two legs would then differ in their estimation window as well
    as in the conditioning.
- OC-A8 **Cell qualification.** The draft sets none. The default follows Two Sigma
  §12.4: at least 63 sessions and 3 episodes on the stamped training path, and a
  positive-definite matrix. With fewer than two qualifying cells, the balanced leg
  holds the blind weights, and this is counted as a fallback.
- OC-A9 **D pooled over the union of test sessions** (default), or per fold then
  averaged. The (growth−, inflation−) cell is concentrated around 2008–2015 (§7 killer
  6), so a fold can hold none of it.
- OC-A10 **The traded path.**
  - Before the first test session, both legs hold the blind fold-1 weights, the Two
    Sigma §12.3 convention.
  - Weights are constant within a fold, and the book is rebalanced back to them every
    session without charge, as in `construction.sleeves`.
  - The default leg charges costs at sleeve level only. :class:`InstrumentLegBuilder`,
    passed as `leg_builder`, is the declared book instead: the sleeve weights spread
    over the 30 instruments by the within-sleeve weights held on each session, built
    by `construction.sleeves.build_book` and charged per instrument.
- OC-A11 **Label timing.**
  - Session t holds the label stamped at or before session t, shifted one session. This
    is `quadrant.daily_labels`, and `protocol.map_states` in Two Sigma.
  - `construction.sleeves.hold` holds a schedule dated τ from the first session
    *strictly after* τ. The two agree when τ is a session, and differ by one session
    when τ falls on a non-trading day.
  - The draft's cell counts use this **stamped** label. A contemporaneous label (the
    quadrant of the quarter a session lies in) is the other reading of an evaluation
    partition. It is not known on the last training sessions.
- OC-A12 **The walk-forward.** "5 folds = 18.2 quarters and 1,187 sessions each" (§1.3,
  §7 killer 7) tiles the whole sample: 5 × 1,187 = 5,935 of 5,938 sessions. Under a
  walk-forward, that leaves fold 1 no training window. An initial training window is
  needed, taken from the sample or from before it. :func:`stamp_folds` builds the
  proposal measured before the lock, from label counts only: the first training
  window ends once every cell has held four quarters, and the remaining quarters form
  five test folds cut at the stamps.
- OC-A13 **The bootstrap of the reduction.** The (blind, balanced, label) triplets of
  the pooled test sessions are resampled jointly. A bootstrap of a max − min is biased
  where two cells nearly tie.

*P1*
- OC-P1a **Blocks.** One block over the sample (the default), or one per calendar
  segment: the pre-test training span, then each test fold (`calendar_blocks`). Unlike
  Two Sigma, the label is not refitted per fold, so one path serves every fold.
- OC-P1b **The grid.** The default is the quarterly stamps. Quarter-level occupancy,
  clock and episodes are then exact. Session-level occupancy is only approximate,
  because quarters hold different numbers of sessions
  (`session_occupancy_deviation`).
- OC-P1c The quarter in force at the sample start is a truncated episode. It is
  redrawn as a whole.

*P2*
- OC-P2a **D is scale-free.** Matching the two legs on *mean* gross exposure, a
  constant, cannot move D. Only a rescaling that changes session by session can.
  Readings: (i) `on="gross"`, the balanced leg rescaled to the blind leg's gross on
  each session; (ii) `on="vol"`, both legs rescaled to one ex-ante volatility, from a
  common trailing 63-session sleeve covariance lagged one session; (iii) the T3 test,
  the real leg's exposure percentile among the P1 placebo legs (`null_statistics`).
  The survival rule belongs to the lock.

*B2 and P3*
- OC-B2a **Weights "from a sign constraint".** The default gives each of the four
  tags {g+, g−, i+, i−} a risk budget of 1/4, split equally among the sleeves carrying
  that tag, with `w ∝ b / σ` (training σ, inverse-volatility budgeting). Alternatives:
  ERC with budgets, or notional weights `w ∝ b`.
- OC-B2b **The space of maps.** Counted here, with the number of maps that cover all
  four tags in brackets:
  - `permutation`: 120 (120).
  - `profile` (every sleeve keeps its number of tags): 1,024 (752).
  - `axis_consistent` (at most one sign per axis): 32,768 (21,300).
  - `any_subset`: 759,375 (693,601).
- **A map acts only through its budget vector, and the budgets ignore the signs.**
  - Flipping every sign of the draft map gives the same budgets, so the same book.
  - `permutation` has **60 distinct books**; the draft map ties with the one that
    swaps commodity and precious metals.
  - `profile` (covering) has **61 distinct books**, and the draft map ties with 32
    maps.
  - Under any budget rule that treats the four tags alike, the map's economics (which
    direction each sleeve is exposed to) cannot reach the weights.
  - Exact enumeration beats 2,000 random draws in every space that small.
- OC-B2c **P3's convention.** Under exact enumeration the real map is in its own null,
  and ties count against it. The smallest attainable p is then:
  - `permutation`: 2/120 = 0.0167;
  - `profile`: 32/752 = 0.0426.

  Both are above the Šidák 0.0102. The 95th percentile of §3 B2 remains reachable:
  0.9917 and 0.9787 at most, with ties counted half.

*Level C*
- OC-C1 **The calendar.** §2 rebalances "quarterly, on the availability stamp". The
  transition dates are then a **subset** of the calendar dates, and the conditioned
  book is the calendar book minus the stamps at which the label repeated. The
  alternative is calendar quarter-ends.
- OC-C2 **The target the rebalance moves to.** If the target is constant within a fold
  and does not drift, every cadence holds the same weights, and **the null is an atom
  at zero**, the Two Sigma C-1 trap (tested). The target must move between rebalances:
  re-estimated weights, a daily multiplier, or drift.
- OC-C3 **The statistic, which the draft leaves unnamed.** Candidates: the paired net
  Sharpe difference, the turnover saving under a tracking gate (Two Sigma C-1), the
  tracking difference, or a reduction in D.
- OC-C4 **The placebo.** The default takes the transition dates of P1 draws, whose
  count and episode multiset are exact. The alternative is a renewal process on the
  empirical spacings.
- OC-C5 No drift between rebalances, the Two Sigma C-1 convention.

*Witness*
- OC-W1 **The driver series.** Candidates: the equity sleeve or the S&P (the
  programme's standing adversary, Two Sigma W1), or the blind leg's own unscaled return
  (Two Sigma W2).
- OC-W2 The window: 21 sessions (Two Sigma W1) or 63.
- OC-W3 **The grid.** Daily bins, or bins sampled at the availability stamps, which
  gives the quadrant's own clock.
- OC-W4 **The downgrade.** If the witness's reduction is at least the quadrant's, the
  result reads DOMINATED. It never passes anything.

*Power*
- OC-N1 **The MDE of the declared statistic.** It is read off the null of the reduction
  (`null_reductions`). One reading is a location shift, `q(1−α) − q(1−power)`, whose
  meaning fails on an atom. The other is `(z_{1−α} + z_power) × sd`. α is 0.05 (P1's
  95th percentile) or the Šidák 0.0102.
- **Why 0.204 is not that MDE.** The draft's 0.204 is `2.8016 / √(2 · 5938/63)`, the
  MDE of **one log σ̂** on the whole panel (`draft_mde_log_sigma`). D differs on four
  counts:
  1. D is a log *variance*, `2 log σ`, whose standard error is twice as large.
  2. It is a difference of two cells.
  3. It is a max − min.
  4. The reduction is a paired difference between two legs.

  :func:`analytic_mde_log_variance_difference` gives the unpaired two-cell figure for
  comparison.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
import pandas as pd
from scipy import optimize, stats

from regime_lab.analysis.bootstrap import stationary_indices
from regime_lab.analysis.placebo import (
    JoinFreeSequences,
    MatchedPlacebo,
    check_placebos,
    matched_placebos,
    run_lengths,
)
from regime_lab.construction import sleeves as sleeve_module
from regime_lab.extensions.trend import MAX_LEVERAGE, VOL_TARGET
from regime_lab.extensions.vehicle import charge_by_instrument
from regime_lab.selection.folds import Fold
from regime_lab.selection.protocol import (
    annual_turnover,
    placebo_p_value,
    placebo_percentile,
    risk_parity_mix,
    scale_to_target,
)

PERIODS = 252
#: §1.1: ``label = 2·1[growth surprise > 0] + 1[inflation surprise > 0]``.
N_CELLS = 4
#: §2: the volatility target's window, and the witness/exposure covariance window.
VOL_WINDOW = 63
#: OC-A8, Two Sigma §12.4: a cell enters the balance objective only above both floors.
MIN_CELL_SESSIONS = 63
MIN_CELL_EPISODES = 3
#: OC-A6: weight of the pull toward the blind weights among exact equalisers.
TIE_BREAK = 1e-4
#: OC-A8: a cell matrix whose smallest eigenvalue is at or below this share of its
#: trace is not positive definite (Two Sigma `level_b.PD_RELATIVE_FLOOR`).
PD_FLOOR = 1e-12
#: §2 inference: stationary block bootstrap; the task's three mean blocks.
POWER_BLOCKS = (21, 63, 126)
BOOTSTRAP_DRAWS = 2_000
#: §3 A condition 2 and §6: 2,000 draws for P1 and P3.
PLACEBO_DRAWS = 2_000
#: §3: five declared tests, Šidák at family α 0.05.
DECLARED_TESTS = 5
FAMILY_ALPHA = 0.05
POWER = 0.80
#: §3 A condition 1 and §4: the draft's MDE on the log-ratio.
DRAFT_MDE = 0.204
#: OC-W2: Two Sigma W1's window.
WITNESS_WINDOW = 21
#: Two Sigma `level_c.ATOM_SHARE`: no single value may hold 20% of a null's draws.
ATOM_SHARE = 0.20

#: §1.3, in the order and with the names of `construction.sleeves.HEADLINE_SLEEVES`.
SLEEVES = ("equity", "duration", "inflation_linked", "commodity", "precious")
#: §3 B2: the four one-sided environments a sleeve is mapped to.
TAGS = ("g+", "g-", "i+", "i-")
#: §3 B2, fixed there "and never adjusted".
DRAFT_MAP: dict[str, frozenset[str]] = {
    "equity": frozenset({"g+"}),
    "duration": frozenset({"g-", "i-"}),
    "inflation_linked": frozenset({"i+"}),
    "commodity": frozenset({"i+", "g+"}),
    "precious": frozenset({"i+", "g-"}),
}

Reading = Literal["equal_variance", "equal_share"]
Scaling = Literal["vol_target", "none"]
Extremes = Literal["own", "blind"]
BlindRule = Literal["erc", "inverse_vol"]
BudgetRule = Literal["inverse_vol", "erc", "notional"]
MapSpace = Literal["permutation", "profile", "axis_consistent", "any_subset"]
CStatistic = Literal["sharpe_difference", "turnover_saving", "tracking_difference"]


class SplitLike(Protocol):
    """A walk-forward fold, as `regime_lab.selection.folds.Fold` provides it."""

    number: int

    def train(self, index: pd.DatetimeIndex) -> pd.DatetimeIndex: ...

    def test(self, index: pd.DatetimeIndex) -> pd.DatetimeIndex: ...


def sidak_alpha(n_tests: int = DECLARED_TESTS, family: float = FAMILY_ALPHA) -> float:
    """Per-test α under Šidák (§3 table): 0.01021 for the tree's five tests."""
    return float(1.0 - (1.0 - family) ** (1.0 / n_tests))


def draft_mde_log_sigma(
    n_sessions: int, *, block: int = 63, alpha: float = FAMILY_ALPHA, power: float = POWER
) -> float:
    """The draft's §4 figure: MDE of ONE log σ̂, ``z · √(1/(2 n_eff))``, ``n_eff = n/block``.

    ``draft_mde_log_sigma(5938)`` gives 0.204. It is the resolution of a single
    log-volatility on the whole panel, not of D (see the module docstring, OC-N1).
    """
    z = stats.norm.ppf(1.0 - alpha / 2.0) + stats.norm.ppf(power)
    return float(z * np.sqrt(1.0 / (2.0 * n_sessions / block)))


def analytic_mde_log_variance_difference(
    n_a: int, n_b: int, *, block: int = 63, alpha: float = FAMILY_ALPHA, power: float = POWER
) -> float:
    """Unpaired MDE of ``log v_a − log v_b`` for two independent cells, Gaussian, blocked.

    ``SE(log v̂) ≈ √(2 / n_eff)``: twice the SE of ``log σ̂``. Reference only. The
    declared reduction is a paired max − min, whose null is measured by
    :func:`null_reductions`.
    """
    z = stats.norm.ppf(1.0 - alpha / 2.0) + stats.norm.ppf(power)
    return float(z * np.sqrt(2.0 * block / n_a + 2.0 * block / n_b))


# ----------------------------------------------------------------------------- labels


def _check_calendar(index: pd.Index, what: str) -> None:
    if not (index.is_monotonic_increasing and index.is_unique):
        raise ValueError(f"{what} must be sorted and without duplicates")


def _asof_positions(stamps: pd.DatetimeIndex, sessions: pd.DatetimeIndex, lag: int) -> np.ndarray:
    """Position in ``stamps`` of the label in force on each session, -1 where none."""
    if lag < 0:
        raise ValueError("lag must be non-negative")
    known = stamps.searchsorted(sessions, side="right") - 1
    out = np.full(len(sessions), -1, dtype=np.int64)
    if lag == 0:
        out[:] = known
    elif lag < len(sessions):
        out[lag:] = known[:-lag]
    return out


def expand_to_sessions(
    stamped: pd.Series | pd.DataFrame, sessions: pd.DatetimeIndex, *, lag: int = 1
) -> pd.Series | pd.DataFrame:
    """The label in force on each session: last stamp at or before it, then ``lag`` sessions.

    §1.1: a label is "held from its availability stamp until the next. Signal at T−1,
    trade at T". This is the convention of `construction.quadrant.daily_labels`
    (OC-A11). ``stamped`` is indexed by ``available_at``. A DataFrame, such as one column
    per placebo draw, is expanded column by column. The result is float, NaN before
    the first stamp and on the first ``lag`` sessions.
    """
    _check_calendar(stamped.index, "stamps")
    if len(stamped) == 0:
        raise ValueError("no stamp to expand")
    sessions = pd.DatetimeIndex(sessions)
    _check_calendar(sessions, "sessions")
    pos = _asof_positions(pd.DatetimeIndex(stamped.index), sessions, lag)
    values = stamped.to_numpy(dtype=float)
    taken = values[np.clip(pos, 0, None)]
    if isinstance(stamped, pd.Series):
        return pd.Series(np.where(pos >= 0, taken, np.nan), index=sessions, name=stamped.name)
    taken = taken.copy()
    taken[pos < 0] = np.nan
    return pd.DataFrame(taken, index=sessions, columns=stamped.columns)


def stamps_in_force(
    stamped: pd.Series, sessions: pd.DatetimeIndex, *, lag: int = 1
) -> pd.Series:
    """The stamps whose label is in force on at least one session, in order (OC-P1c).

    The first of them is usually stamped before the first session. Its episode is
    truncated by the sample start, and P1 redraws it as a whole.
    """
    _check_calendar(stamped.index, "stamps")
    pos = _asof_positions(pd.DatetimeIndex(stamped.index), pd.DatetimeIndex(sessions), lag)
    used = pos[pos >= 0]
    if used.size == 0:
        raise ValueError("no stamp is in force on any session")
    return stamped.iloc[int(used.min()) : int(used.max()) + 1]


def stamps_to_sessions(
    dates: Sequence[pd.Timestamp] | pd.DatetimeIndex,
    sessions: pd.DatetimeIndex,
    *,
    lag: int = 1,
) -> pd.DatetimeIndex:
    """The session on which a label stamped on each date first applies (§1.1, OC-A11).

    It is the first session at or after the date, plus ``lag`` sessions: the same
    session on which :func:`expand_to_sessions` first shows the label. Dates whose
    session falls beyond the sample are dropped.
    """
    sessions = pd.DatetimeIndex(sessions)
    pos = sessions.searchsorted(pd.DatetimeIndex(dates), side="left") + lag
    return sessions[np.unique(pos[pos < len(sessions)])]


def transition_stamps(stamped: pd.Series) -> pd.DatetimeIndex:
    """Stamps at which the label differs from the previous stamp's (§3 C: "the day it changes")."""
    _check_calendar(stamped.index, "stamps")
    if stamped.isna().any():
        raise ValueError("stamped labels contain missing values")
    values = stamped.to_numpy()
    change = np.r_[False, values[1:] != values[:-1]]
    return pd.DatetimeIndex(stamped.index[change])


def stamp_folds(
    stamped: pd.Series,
    sessions: pd.DatetimeIndex,
    *,
    n_folds: int = 5,
    min_quarters: int = 4,
    usable: pd.DatetimeIndex | None = None,
    lag: int = 1,
    n_cells: int = N_CELLS,
) -> tuple[Fold, ...]:
    """OC-A12: an expanding walk-forward cut at the stamps, from label COUNTS only.

    The draft's "5 folds = 18.2 quarters and 1,187 sessions each" tiles the whole
    sample and leaves fold 1 no training window. This is the proposal measured before
    the lock, not a decision:
    - Each stamp holds the sessions on which its label is in force
      (:func:`expand_to_sessions` with ``lag``), so no quarter is ever split by a fold.
    - Quarters are counted, in order, from the first whose holding window starts on or
      after the first ``usable`` session (default: the first session), for example the
      first session on which the sleeve returns exist.
    - The first training window ends with the first counted quarter at which every
      cell has been held by at least ``min_quarters`` counted quarters.
    - The later quarters form ``n_folds`` contiguous test folds of equal numbers of
      quarters (`numpy.array_split`: the first folds take the remainder).
    - Fold ``k`` trains on every session before its test window (expanding).

    Label counts only: no return is read. Returns `selection.folds.Fold` objects.

    Raises:
        ValueError: if the rule is never met, or leaves fewer quarters than folds.
    """
    sessions = pd.DatetimeIndex(sessions)
    _check_calendar(sessions, "sessions")
    used = stamps_in_force(stamped, sessions, lag=lag)
    pos = _asof_positions(pd.DatetimeIndex(used.index), sessions, lag)
    held = np.flatnonzero(np.bincount(pos[pos >= 0], minlength=len(used)) > 0)
    first = np.array([int(np.argmax(pos == k)) for k in held])
    last = np.array([len(pos) - 1 - int(np.argmax(pos[::-1] == k)) for k in held])
    start = 0 if usable is None or len(usable) == 0 else int(
        sessions.searchsorted(pd.DatetimeIndex(usable)[0], side="left"))
    counted = first >= start
    codes = _codes(used.to_numpy(float)[held], n_cells)
    tally = np.zeros(n_cells, dtype=np.int64)
    k0 = None
    for i in np.flatnonzero(counted):
        tally[codes[i]] += 1
        if (tally >= min_quarters).all():
            k0 = int(i)
            break
    if k0 is None:
        raise ValueError(f"no cell count reaches {min_quarters} quarters in every cell")
    rest = np.arange(k0 + 1, len(held))
    if rest.size < n_folds:
        raise ValueError("fewer test quarters than folds")
    folds = []
    for number, group in enumerate(np.array_split(rest, n_folds), start=1):
        a, b = int(first[group[0]]), int(last[group[-1]])
        folds.append(Fold(number, sessions[0], sessions[a - 1], sessions[a], sessions[b],
                          sessions[a - 1], sessions[b]))
    return tuple(folds)


def _codes(values: np.ndarray, n_cells: int) -> np.ndarray:
    """Float labels (NaN = none) to integer codes (-1 = none), refusing anything else."""
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    kept = values[finite]
    if kept.size and (
        (kept != np.round(kept)).any() or kept.min() < 0 or kept.max() >= n_cells
    ):
        raise ValueError(f"labels must be integers in [0, {n_cells})")
    codes = np.full(values.shape, -1, dtype=np.int64)
    codes[finite] = kept.astype(np.int64)
    return codes


def cell_counts(codes: np.ndarray, n_cells: int = N_CELLS) -> tuple[np.ndarray, np.ndarray]:
    """Sessions and episodes of each cell on a path of codes, -1 separating episodes."""
    codes = np.asarray(codes, dtype=np.int64)
    labelled = codes[codes >= 0]
    sessions = np.bincount(labelled, minlength=n_cells)[:n_cells]
    _, states = run_lengths(codes)
    states = np.asarray(states, dtype=np.int64)
    episodes = np.bincount(states[states >= 0], minlength=n_cells)[:n_cells]
    return sessions.astype(np.int64), episodes.astype(np.int64)


# ------------------------------------------------------------------------ statistic D


def _cell_variances(
    x: np.ndarray, codes: np.ndarray, n_cells: int, about_zero: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Sessions and variance of ``x`` per cell; NaN variance below two sessions."""
    keep = (codes >= 0) & np.isfinite(x)
    c, v = codes[keep], x[keep]
    n = np.bincount(c, minlength=n_cells)[:n_cells].astype(float)
    s2 = np.bincount(c, weights=v * v, minlength=n_cells)[:n_cells]
    with np.errstate(divide="ignore", invalid="ignore"):
        if about_zero:
            var = s2 / n
        else:
            s1 = np.bincount(c, weights=v, minlength=n_cells)[:n_cells]
            var = (s2 - s1 * s1 / n) / (n - 1.0)
    var = np.where(n >= 2, var, np.nan)
    return n, var


def _usable(n: np.ndarray, var: np.ndarray, min_sessions: int) -> np.ndarray:
    return (n >= max(min_sessions, 2)) & np.isfinite(var) & (var > 0)


def _extremes(n: np.ndarray, var: np.ndarray, min_sessions: int) -> tuple[int, int] | None:
    ok = _usable(n, var, min_sessions)
    if ok.sum() < 2:
        return None
    masked_hi = np.where(ok, var, -np.inf)
    masked_lo = np.where(ok, var, np.inf)
    return int(masked_hi.argmax()), int(masked_lo.argmin())


def _d_from(
    n: np.ndarray,
    var: np.ndarray,
    *,
    occupancy_weighted: bool,
    cells: tuple[int, int] | None,
    min_sessions: int,
) -> float:
    ok = _usable(n, var, min_sessions)
    logs = np.log(np.where(ok, var, np.nan))
    if occupancy_weighted:
        logs = logs + np.log(np.where(ok, n, np.nan) / n[ok].sum())
    if cells is not None:
        worst, best = cells
        if not (ok[worst] and ok[best]):
            return float("nan")
        return float(logs[worst] - logs[best])
    if ok.sum() < 2:
        return float("nan")
    return float(np.nanmax(logs) - np.nanmin(logs))


@dataclass(frozen=True)
class DSettings:
    """How D is read: OC-A2 (``extremes``), OC-A3 (``about_zero``), OC-A4 (weighting)."""

    n_cells: int = N_CELLS
    about_zero: bool = False
    occupancy_weighted: bool = False
    extremes: Extremes = "own"
    min_sessions: int = 2

    def pair(self, blind: np.ndarray, other: np.ndarray, codes: np.ndarray) -> tuple[
        float, float, tuple[int, int] | None
    ]:
        """D of the blind leg and of the other leg on the same labelled sessions."""
        nb, vb = _cell_variances(blind, codes, self.n_cells, self.about_zero)
        no, vo = _cell_variances(other, codes, self.n_cells, self.about_zero)
        cells = _extremes(nb, vb, self.min_sessions) if self.extremes == "blind" else None
        if self.extremes == "blind" and cells is None:
            return float("nan"), float("nan"), None
        kw = {"occupancy_weighted": self.occupancy_weighted, "cells": cells,
              "min_sessions": self.min_sessions}
        return _d_from(nb, vb, **kw), _d_from(no, vo, **kw), cells


def cell_variances(
    returns: pd.Series, labels: pd.Series, *, n_cells: int = N_CELLS, about_zero: bool = False
) -> pd.DataFrame:
    """Sessions and variance of ``returns`` in each cell of ``labels`` (§3 A).

    Blindness: on real labels this is a per-cell variance, forbidden before the lock.
    """
    codes = _codes(labels.reindex(returns.index).to_numpy(float), n_cells)
    n, var = _cell_variances(returns.to_numpy(float), codes, n_cells, about_zero)
    return pd.DataFrame({"sessions": n.astype(np.int64), "variance": var},
                        index=pd.RangeIndex(n_cells, name="cell"))


def d_statistic(
    returns: pd.Series,
    labels: pd.Series,
    *,
    n_cells: int = N_CELLS,
    about_zero: bool = False,
    occupancy_weighted: bool = False,
    cells: tuple[int, int] | None = None,
    min_sessions: int = 2,
) -> float:
    """``D = log var(worst cell) − log var(best cell)`` (§3 A, primary statistic).

    - By default, "worst" and "best" are the highest- and lowest-variance cells of
      ``returns`` itself (OC-A2).
    - ``cells=(worst, best)`` fixes them instead, for example to the blind leg's.
    - ``occupancy_weighted`` reads `log π_k v_k`, the share reading R2 implies (OC-A4).
    - Cells with fewer than ``min_sessions`` labelled sessions are left out.
    - NaN if fewer than two cells remain.

    Blindness: on real labels this is D, forbidden before the lock.
    """
    codes = _codes(labels.reindex(returns.index).to_numpy(float), n_cells)
    n, var = _cell_variances(returns.to_numpy(float), codes, n_cells, about_zero)
    return _d_from(n, var, occupancy_weighted=occupancy_weighted, cells=cells,
                   min_sessions=min_sessions)


# ---------------------------------------------------------------------------- weights


def _moment(x: np.ndarray, about_zero: bool) -> np.ndarray:
    if about_zero:
        return x.T @ x / len(x)
    return np.atleast_2d(np.cov(x, rowvar=False, ddof=1))


def _positive_definite(m: np.ndarray) -> bool:
    trace = float(np.trace(m))
    return bool(np.isfinite(m).all() and trace > 0
                and np.linalg.eigvalsh(m).min() > PD_FLOOR * trace)


def blind_weights(moment: np.ndarray, *, rule: BlindRule = "erc") -> np.ndarray:
    """§3 A leg (i): unconditional risk parity across sleeves, long only, summing to one.

    ``"erc"`` is `selection.protocol.risk_parity_mix`, equal risk contributions on the
    full matrix. ``"inverse_vol"`` is its diagonal special case (OC-A7).
    """
    moment = np.asarray(moment, dtype=float)
    if rule == "erc":
        return np.asarray(risk_parity_mix(moment), dtype=float)
    if rule == "inverse_vol":
        inverse = 1.0 / np.sqrt(np.diag(moment))
        return inverse / inverse.sum()
    raise ValueError(f"rule must be 'erc' or 'inverse_vol', not {rule!r}")


@dataclass(frozen=True)
class BalanceSolution:
    """The balanced weights for one set of cell matrices, and how well they balance."""

    weights: np.ndarray
    #: max − min over the cells of ``log(w'Σ_k w) − offset_k`` at the solution (training).
    dispersion: float
    converged: bool


def balanced_weights(
    moments: np.ndarray,
    start: np.ndarray,
    *,
    offsets: np.ndarray | None = None,
    tie_break: float = TIE_BREAK,
) -> BalanceSolution:
    """§3 A leg (ii): long-only weights, summing to one, that equalise the cells.

    The objective is ``Σ_k (ℓ_k − mean ℓ)² + tie_break · ‖w − start‖²``, with
    ``ℓ_k = log(w'Σ_k w) − offset_k``.
    - ``offsets = 0`` is R1: equal variance in every cell.
    - ``offsets = −log π_k`` is R2: equal occupancy-weighted shares (OC-A4).
    - The objective is least squares on the log variances (OC-A5), and it is
      scale-free, as the volatility target is.
    - The penalty picks, among exact equalisers, the one nearest ``start`` (OC-A6).

    SLSQP with an analytic gradient runs from ``start``, so the same input always gives
    the same bits. The objective is not convex: the solution is the local optimum
    reached from ``start``.
    """
    m = np.asarray(moments, dtype=float)
    if m.ndim != 3 or m.shape[1] != m.shape[2]:
        raise ValueError("moments must be (cells, sleeves, sleeves)")
    k, p, _ = m.shape
    t = np.zeros(k) if offsets is None else np.asarray(offsets, dtype=float)
    if t.shape != (k,):
        raise ValueError("one offset per cell")
    w0 = np.asarray(start, dtype=float)
    if w0.shape != (p,) or (w0 < 0).any() or not w0.sum() > 0:
        raise ValueError("start must be a non-negative weight per sleeve")
    w0 = w0 / w0.sum()

    def fun(w: np.ndarray) -> tuple[float, np.ndarray]:
        mw = m @ w
        v = mw @ w
        ell = np.log(v) - t
        e = ell - ell.mean()
        gap = w - w0
        value = float(e @ e + tie_break * gap @ gap)
        grad = 4.0 * (e / v) @ mw + 2.0 * tie_break * gap
        return value, grad

    result = optimize.minimize(
        fun, w0, jac=True, method="SLSQP", bounds=[(0.0, 1.0)] * p,
        constraints=({"type": "eq", "fun": lambda w: w.sum() - 1.0,
                      "jac": lambda w: np.ones_like(w)},),
        options={"ftol": 1e-15, "maxiter": 1_000},
    )
    w = np.clip(result.x, 0.0, None)
    w = w / w.sum()
    ell = np.log(np.einsum("i,kij,j->k", w, m, w)) - t
    return BalanceSolution(w, float(ell.max() - ell.min()), bool(result.success))


@dataclass(frozen=True)
class BalancedFit:
    """One fold's balanced leg, with the training counts that qualified its cells."""

    weights: np.ndarray
    #: Cells that entered the objective (OC-A8).
    cells: tuple[int, ...]
    #: Sessions and episodes of every cell on the stamped training path.
    sessions: np.ndarray
    episodes: np.ndarray
    dispersion: float
    converged: bool
    #: Fewer than two cells qualified: the blind weights are held.
    fallback: bool


def fit_balanced(
    returns: np.ndarray,
    codes: np.ndarray,
    start: np.ndarray,
    *,
    reading: Reading = "equal_variance",
    about_zero: bool = False,
    n_cells: int = N_CELLS,
    min_cell_sessions: int = MIN_CELL_SESSIONS,
    min_cell_episodes: int = MIN_CELL_EPISODES,
    tie_break: float = TIE_BREAK,
) -> BalancedFit:
    """Fit §3 A's balanced leg on one training window.

    Arguments:
    - ``returns`` is sessions × sleeves on the fold's training sessions, in order.
    - ``codes`` is the stamped label on those sessions, -1 where there is none.

    Steps:
    - Cells are qualified on the stamped path (OC-A8).
    - Each cell's matrix comes from its sessions with finite returns only.
    - R2's ``π_k`` is the cell's share of the qualifying cells' estimation sessions.
    """
    x = np.asarray(returns, dtype=float)
    codes = np.asarray(codes, dtype=np.int64)
    if x.ndim != 2 or len(x) != len(codes):
        raise ValueError("returns must be sessions x sleeves, aligned with codes")
    if reading not in ("equal_variance", "equal_share"):
        raise ValueError(f"unknown reading {reading!r}")
    p = x.shape[1]
    sessions_k, episodes_k = cell_counts(codes, n_cells)
    rows = np.isfinite(x).all(axis=1) & (codes >= 0)
    c, xr = codes[rows], x[rows]
    counts = np.bincount(c, minlength=n_cells)[:n_cells]
    qualify = (sessions_k >= min_cell_sessions) & (episodes_k >= min_cell_episodes) & (counts > p)
    moments: dict[int, np.ndarray] = {}
    for k in np.flatnonzero(qualify):
        m = _moment(xr[c == k], about_zero)
        if _positive_definite(m):
            moments[int(k)] = m
    cells = tuple(sorted(moments))
    start = np.asarray(start, dtype=float)
    if len(cells) < 2:
        return BalancedFit(start / start.sum(), cells, sessions_k, episodes_k,
                           float("nan"), True, True)
    stacked = np.stack([moments[k] for k in cells])
    offsets = None
    if reading == "equal_share":
        share = counts[list(cells)] / counts[list(cells)].sum()
        offsets = -np.log(share)
    solution = balanced_weights(stacked, start, offsets=offsets, tie_break=tie_break)
    return BalancedFit(solution.weights, cells, sessions_k, episodes_k,
                       solution.dispersion, solution.converged, False)


# ------------------------------------------------------------------------- the legs


@dataclass(frozen=True)
class Leg:
    """One leg on the traded path: its returns and the weights it holds.

    ``held`` is whatever level the builder charges costs on: sleeves for the default
    builder, instruments for a builder built on `construction.sleeves`.
    """

    returns: pd.Series
    held: pd.DataFrame
    multiplier: pd.Series
    cap_binds: pd.Series


#: Maps the unscaled sleeve-weight path (sessions × sleeves, NaN where not held) to a leg.
LegBuilder = Callable[[pd.DataFrame], Leg]


@dataclass(frozen=True)
class SleeveLegBuilder:
    """The default leg (OC-A1, OC-A10): sleeve-level book, optional target, sleeve costs.

    ``scaling="vol_target"`` is §2's portfolio: `selection.protocol.scale_to_target`,
    which uses `extensions.vehicle.portfolio_scalar`. It targets 10% on the 63 sessions
    of the unscaled book lagged one session, with the multiplier capped at 3, and the
    warm-up left NaN. ``sleeve_bps`` charges ``|Δ held|`` at sleeve level through
    `vehicle.charge_by_instrument`.
    """

    returns: pd.DataFrame
    scaling: Scaling = "vol_target"
    sleeve_bps: pd.Series | None = None
    target: float = VOL_TARGET
    window: int = VOL_WINDOW
    cap: float = MAX_LEVERAGE

    def __call__(self, path: pd.DataFrame) -> Leg:
        x = self.returns.reindex(index=path.index, columns=path.columns).to_numpy(float)
        w = path.to_numpy(float)
        ok = np.isfinite(x).all(axis=1) & np.isfinite(w).all(axis=1)
        unscaled = pd.Series(
            np.where(ok, np.where(ok[:, None], w * x, 0.0).sum(axis=1), np.nan),
            index=path.index, name="leg",
        )
        if self.scaling == "vol_target":
            scaled, multiplier, binds = scale_to_target(
                unscaled, target=self.target, window=self.window, cap=self.cap
            )
        elif self.scaling == "none":
            scaled = unscaled
            multiplier = pd.Series(np.where(ok, 1.0, np.nan), index=path.index)
            binds = pd.Series(np.where(ok, 0.0, np.nan), index=path.index)
        else:
            raise ValueError(f"scaling must be 'vol_target' or 'none', not {self.scaling!r}")
        held = path.mul(multiplier, axis=0)
        out = scaled
        if self.sleeve_bps is not None:
            out = charge_by_instrument(scaled, held, self.sleeve_bps)
        return Leg(out.rename("leg"), held, multiplier.rename("multiplier"),
                   pd.Series(binds, index=path.index, dtype=float).rename("cap_binds"))


@dataclass(frozen=True)
class InstrumentLegBuilder:
    """The declared leg of §2, at instrument level: the builder the power script runs.

    A sleeve-weight path ``S_s(t)`` becomes instrument weights ``S_s(t) · w_i|s(t)``,
    where ``w_i|s(t)`` are the within-sleeve weights HELD on t (for example
    `sleeves.hold` of `sleeves.within_sleeve_weights`, re-estimated at every stamp and
    held from the next session). The book is `sleeves.build_book`, so it is the same
    object as the state-blind book of `construction.sleeves`:
    - the 10% target on 63 sessions, lagged one, with the multiplier capped at 3;
    - NaN during the warm-up, never zero;
    - net of `sleeves.cost_drag` at ``cost_column`` on ``|Δw|`` of the held instrument
      weights, entry included (``None`` charges nothing).

    ``excess`` must be instrument excess returns, funded where the instrument is a
    total-return ETF (`sleeves.excess_returns`). The unscaled leg return equals
    ``Σ_s S_s · r_s`` with ``r_s`` = `sleeves.sleeve_returns` on the same held
    within-sleeve weights, which is what `LevelA` fits its weights on (tested). A sleeve
    at weight zero holds nothing, even where its within weights are undefined.

    ``foreign_lag`` (§12.4, §12.18 of the lock) is a diagnostic of time-zone causality:
    the scaled weights of the ``foreign`` columns (markets that close before the US
    close of the same date) are held ``foreign_lag`` sessions later before the book
    return is formed, and costs are charged on the weights so held. The multiplier is
    the one of the book at ``foreign_lag = 0``. At 0 the builder is the declared book,
    bit for bit.
    """

    excess: pd.DataFrame
    within: pd.DataFrame
    sleeves: Mapping[str, Sequence[str]]
    cost_column: str | None = "headline"
    target: float = VOL_TARGET
    window: int = VOL_WINDOW
    cap: float = MAX_LEVERAGE
    foreign_lag: int = 0
    foreign: tuple[str, ...] = ()

    def __call__(self, path: pd.DataFrame) -> Leg:
        names = sleeve_module.instruments(self.sleeves)
        owner = [s for s, members in self.sleeves.items() for _ in members]
        scale = path.reindex(columns=list(self.sleeves)).to_numpy(float)[
            :, [list(self.sleeves).index(s) for s in owner]]
        inner = self.within.reindex(index=path.index, columns=names).to_numpy(float)
        held = np.where(scale == 0.0, 0.0, inner * scale)
        held_frame = pd.DataFrame(held, index=path.index, columns=names)
        book = sleeve_module.build_book(self.excess, held_frame, target=self.target,
                                        window=self.window, cap=self.cap)
        if self.foreign_lag < 0:
            raise ValueError("foreign_lag must be non-negative")
        if self.foreign_lag == 0:
            out = book.returns
            weights = book.weights
        else:
            late = [c for c in names if c in set(self.foreign)]
            if not late:
                raise ValueError("foreign_lag needs at least one foreign column of the book")
            weights = book.weights.copy()
            weights[late] = book.weights[late].shift(self.foreign_lag)
            r = self.excess.reindex(index=weights.index, columns=names).fillna(0.0)
            out = (weights * r).sum(axis=1).where(weights.notna().all(axis=1)).rename("book")
        if self.cost_column is not None:
            out = out - sleeve_module.cost_drag(weights, self.cost_column)
        return Leg(out.rename("leg"), weights, book.multiplier, book.cap_binds)


@dataclass(frozen=True)
class LevelAResult:
    """Level A on one partition: both legs, their fold weights, and D on the test sessions.

    Every return and label field is restricted to the pooled test sessions. The
    ``blind_leg`` and ``balanced_leg`` fields keep the whole traded path, which P2 needs.
    """

    blind_weights: pd.DataFrame
    balanced_weights: pd.DataFrame
    fits: tuple[BalancedFit, ...] | None
    blind_leg: Leg
    balanced_leg: Leg
    test_labels: pd.Series
    d_blind: float
    d_balanced: float
    cells: tuple[int, int] | None
    #: Test sessions on which either leg is missing, or the label is: none enter D.
    missing_sessions: int
    settings: DSettings

    @property
    def reduction(self) -> float:
        """§3 A condition 1: ``D(blind) − D(balanced)``, positive when balancing helps."""
        return self.d_blind - self.d_balanced

    @property
    def blind(self) -> pd.Series:
        """The blind leg's returns on the pooled test sessions."""
        return self.blind_leg.returns.reindex(self.test_labels.index)

    @property
    def balanced(self) -> pd.Series:
        """The balanced leg's returns on the pooled test sessions."""
        return self.balanced_leg.returns.reindex(self.test_labels.index)

    @property
    def fallback_folds(self) -> tuple[int, ...]:
        """Folds whose balanced leg holds the blind weights (OC-A8)."""
        if self.fits is None:
            return ()
        return tuple(int(n) for n, f in zip(self.balanced_weights.index, self.fits, strict=True)
                     if f.fallback)


class LevelA:
    """§3 A on one return panel and one walk-forward: fit once, run on any partition.

    Inputs:
    - ``returns`` are the sleeves' daily excess returns (§2), sessions × sleeves.
    - ``folds`` are objects with ``number``, ``train(index)`` and ``test(index)``, as
      `selection.folds.walk_forward_folds` returns. Their test windows must be disjoint
      and increasing, and each must start after its training window ends.

    What is fitted once, and what per run:
    - The blind leg reads no label, so it is fitted once, on every finite training
      session (OC-A7).
    - :meth:`run` fits the balanced leg on a partition and reads D.
    - Every argument is one of the open choices listed in the module docstring.
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        folds: Sequence[SplitLike],
        *,
        reading: Reading = "equal_variance",
        about_zero: bool = False,
        blind_rule: BlindRule = "erc",
        extremes: Extremes = "own",
        occupancy_weighted: bool = False,
        n_cells: int = N_CELLS,
        min_cell_sessions: int = MIN_CELL_SESSIONS,
        min_cell_episodes: int = MIN_CELL_EPISODES,
        tie_break: float = TIE_BREAK,
        scaling: Scaling = "vol_target",
        sleeve_bps: pd.Series | None = None,
        leg_builder: LegBuilder | None = None,
    ):
        _check_calendar(returns.index, "returns")
        if reading not in ("equal_variance", "equal_share"):
            raise ValueError(f"unknown reading {reading!r}")
        if extremes not in ("own", "blind"):
            raise ValueError(f"extremes must be 'own' or 'blind', not {extremes!r}")
        self.returns = returns.astype(float)
        self.sessions = pd.DatetimeIndex(returns.index)
        self.sleeves = pd.Index(returns.columns)
        self.reading: Reading = reading
        self.about_zero = about_zero
        self.min_cell_sessions = min_cell_sessions
        self.min_cell_episodes = min_cell_episodes
        self.tie_break = tie_break
        self.settings = DSettings(n_cells, about_zero, occupancy_weighted, extremes)
        self.builder: LegBuilder = leg_builder or SleeveLegBuilder(
            self.returns, scaling=scaling, sleeve_bps=sleeve_bps
        )
        self._x = self.returns.to_numpy()
        self._finite = np.isfinite(self._x).all(axis=1)
        self.folds = tuple(folds)
        if not self.folds:
            raise ValueError("at least one fold")
        self._train: list[np.ndarray] = []
        self._test: list[np.ndarray] = []
        for fold in self.folds:
            train = np.flatnonzero(self.sessions.isin(fold.train(self.sessions)))
            test = np.flatnonzero(self.sessions.isin(fold.test(self.sessions)))
            if not (train.size and test.size):
                raise ValueError(f"fold {fold.number} has an empty training or test window")
            if train.max() >= test.min():
                raise ValueError(f"fold {fold.number} trains on or after its first test session")
            self._train.append(train)
            self._test.append(test)
        self._test_positions = np.concatenate(self._test)
        if (np.diff(self._test_positions) <= 0).any():
            raise ValueError("test windows must be disjoint and in increasing order")
        self._first_test = int(self._test_positions[0])
        self._last_test = int(self._test_positions[-1])
        self._blind = np.vstack([
            blind_weights(_moment(self._x[t][self._finite[t]], about_zero), rule=blind_rule)
            for t in self._train
        ])
        self._blind_leg = self.builder(self._path(self._blind))

    # -- construction ------------------------------------------------------------------
    @property
    def fold_numbers(self) -> pd.Index:
        return pd.Index([f.number for f in self.folds], name="fold")

    @property
    def blind_weights(self) -> pd.DataFrame:
        """The blind leg's sleeve weights per fold. They read no label."""
        return pd.DataFrame(self._blind, index=self.fold_numbers, columns=self.sleeves)

    @property
    def blind_leg(self) -> Leg:
        return self._blind_leg

    @property
    def test_sessions(self) -> pd.DatetimeIndex:
        return self.sessions[self._test_positions]

    def _path(self, fold_weights: np.ndarray) -> pd.DataFrame:
        """Unscaled sleeve weights held on each session (OC-A10)."""
        n, p = self._x.shape
        path = np.full((n, p), np.nan)
        path[: self._first_test] = self._blind[0]
        for f, test in enumerate(self._test):
            path[test] = fold_weights[f]
        path = pd.DataFrame(path).ffill().to_numpy(copy=True)
        path[self._last_test + 1 :] = np.nan
        return pd.DataFrame(path, index=self.sessions, columns=self.sleeves)

    def _label_paths(self, labels: pd.Series | Mapping[int, pd.Series]) -> np.ndarray:
        """One row of codes per fold: a common path, or each fold's own (the witness)."""
        n_cells = self.settings.n_cells

        def one(series: pd.Series) -> np.ndarray:
            _check_calendar(series.index, "labels")
            return _codes(series.reindex(self.sessions).to_numpy(float), n_cells)

        if isinstance(labels, Mapping):
            return np.vstack([one(labels[f.number]) for f in self.folds])
        row = one(labels)
        return np.vstack([row] * len(self.folds))

    def fit(self, labels: pd.Series | Mapping[int, pd.Series]) -> tuple[BalancedFit, ...]:
        """Each fold's balanced leg, from that fold's training sessions only."""
        codes = self._label_paths(labels)
        return tuple(
            fit_balanced(
                self._x[train], codes[f][train], self._blind[f], reading=self.reading,
                about_zero=self.about_zero, n_cells=self.settings.n_cells,
                min_cell_sessions=self.min_cell_sessions,
                min_cell_episodes=self.min_cell_episodes, tie_break=self.tie_break,
            )
            for f, train in enumerate(self._train)
        )

    # -- reading -----------------------------------------------------------------------
    def run(
        self,
        labels: pd.Series | Mapping[int, pd.Series],
        *,
        weights: np.ndarray | pd.DataFrame | None = None,
    ) -> LevelAResult:
        """Fit the balanced leg on ``labels``, or hold ``weights``, and read D on test.

        - ``labels`` is the label in force on each session (§1.1, lagged), or one such
          path per fold number.
        - ``weights`` (folds × sleeves) replaces the fitted leg: B2's map, or any
          static schedule.
        - Blindness: on real labels this computes D. Before the lock it may run only
          inside :func:`null_statistics`.
        """
        codes = self._label_paths(labels)
        if weights is None:
            fits: tuple[BalancedFit, ...] | None = self.fit(labels)
            w = np.vstack([fit.weights for fit in fits])
        else:
            fits = None
            w = np.asarray(weights, dtype=float)
            if w.shape != self._blind.shape:
                raise ValueError(f"weights must be {self._blind.shape} (folds x sleeves)")
        leg = self.builder(self._path(w))
        test_codes = np.concatenate([codes[f][t] for f, t in enumerate(self._test)])
        xb = self._blind_leg.returns.to_numpy(float)[self._test_positions]
        xo = leg.returns.to_numpy(float)[self._test_positions]
        missing = int((~(np.isfinite(xb) & np.isfinite(xo)) | (test_codes < 0)).sum())
        both = np.isfinite(xb) & np.isfinite(xo)
        codes_used = np.where(both, test_codes, -1)
        d_blind, d_other, cells = self.settings.pair(xb, xo, codes_used)
        test_labels = pd.Series(np.where(test_codes >= 0, test_codes, np.nan).astype(float),
                                index=self.test_sessions, name="label")
        return LevelAResult(
            blind_weights=self.blind_weights,
            balanced_weights=pd.DataFrame(w, index=self.fold_numbers, columns=self.sleeves),
            fits=fits,
            blind_leg=self._blind_leg,
            balanced_leg=leg,
            test_labels=test_labels,
            d_blind=d_blind,
            d_balanced=d_other,
            cells=cells,
            missing_sessions=missing,
            settings=self.settings,
        )


# --------------------------------------------------------------------------- bootstrap


def reduction_bootstrap(
    blind: pd.Series,
    balanced: pd.Series,
    labels: pd.Series,
    *,
    mean_block: int,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = 0,
    settings: DSettings | None = None,
) -> np.ndarray:
    """Stationary-block-bootstrap draws of the reduction in D (§2 inference, OC-A13).

    - The (blind, balanced, label) triplets of the labelled test sessions with both
      legs finite are resampled jointly, on one index path per draw
      (`analysis.bootstrap.stationary_indices`, wrapping). The comparison therefore
      stays paired.
    - A draw in which a cell falls below two sessions gives NaN. It is kept, so that
      the standard error is visibly undefined.
    - Blindness: on real labels this computes cell variances. Before the lock it may
      run only on placebo or synthetic labels.
    """
    s = settings or DSettings()
    frame = pd.concat([blind.rename("b"), balanced.rename("o"),
                       labels.reindex(blind.index).rename("k")], axis=1).dropna()
    xb = frame["b"].to_numpy(float)
    xo = frame["o"].to_numpy(float)
    codes = _codes(frame["k"].to_numpy(float), s.n_cells)
    n = len(frame)
    if n < 2:
        raise ValueError("fewer than two usable test sessions")
    rng = np.random.default_rng(seed)
    out = np.empty(draws)
    for d in range(draws):
        idx = stationary_indices(n, mean_block, rng)
        d_blind, d_other, _ = s.pair(xb[idx], xo[idx], codes[idx])
        out[d] = d_blind - d_other
    return out


def reduction_se(
    blind: pd.Series,
    balanced: pd.Series,
    labels: pd.Series,
    *,
    blocks: Sequence[int] = POWER_BLOCKS,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = 0,
    settings: DSettings | None = None,
) -> pd.Series:
    """Bootstrap standard error of the reduction at each mean block. NaN if any draw is.

    The Two Sigma convention (§12.12 there) reads the largest standard error among
    the blocks.
    """
    out = {}
    for block in blocks:
        values = reduction_bootstrap(blind, balanced, labels, mean_block=block, draws=draws,
                                     seed=seed, settings=settings)
        out[int(block)] = float(np.std(values, ddof=1)) if np.isfinite(values).all() else np.nan
    return pd.Series(out, name="se").rename_axis("mean_block")


# ---------------------------------------------------------------------------------- P1


def p1_draws(
    labels: pd.Series,
    n_draws: int = PLACEBO_DRAWS,
    *,
    seed: int = 0,
    blocks: pd.Series | None = None,
) -> MatchedPlacebo:
    """§6 P1: random partitions keeping occupancy, clock and every episode, exactly.

    The draws come from `analysis.placebo.matched_placebos` with ``method="uniform"``:
    uniform over the join-free orders, with each state's own lengths permuted, and no
    retry. That is the construction Two Sigma adopted after the swap repair failed. It
    is run on the label sequence it is given, which is the quarterly stamps (OC-P1b).
    With ``blocks`` it runs within each contiguous block (OC-P1a).

    The draws are checked with `check_placebos`, and a draw that is not exact raises.
    Draw ``d`` depends on ``seed`` and ``d`` only.
    """
    placebo = matched_placebos(labels, n_draws, seed=seed, groups=blocks, method="uniform")
    check = check_placebos(labels, placebo.draws, groups=blocks)
    if not check.exact:
        raise RuntimeError("P1 draws do not reproduce the clock, occupancy and episodes")
    return placebo


def calendar_blocks(
    stamps: pd.DatetimeIndex,
    sessions: pd.DatetimeIndex,
    folds: Sequence[SplitLike],
    *,
    lag: int = 1,
) -> pd.Series:
    """The calendar segment of each stamp, for P1 per block (OC-P1a).

    A stamp belongs to the segment holding the first session on which its label is in
    force (:func:`stamps_to_sessions`). The segments are ``"0:train"`` before the first
    test session, then ``"<fold>:test"``. Stamps in force only after the last test
    session join the last fold.
    """
    sessions = pd.DatetimeIndex(sessions)
    stamps = pd.DatetimeIndex(stamps)
    first = sessions.searchsorted(stamps, side="left") + lag
    starts = np.array([sessions.get_loc(f.test(sessions)[0]) for f in folds])
    if (np.diff(starts) <= 0).any():
        raise ValueError("folds must be in increasing order")
    which = np.searchsorted(starts, first, side="right")
    names = ["0:train"] + [f"{f.number}:test" for f in folds]
    return pd.Series([names[i] for i in which], index=stamps, name="block")


def p1_feasibility(labels: pd.Series, blocks: pd.Series | None = None) -> pd.DataFrame:
    """Per block: points, episodes per state, and log10 of the number of join-free orders.

    The number of orders comes from `analysis.placebo.JoinFreeSequences`. A block is
    feasible when at least one join-free order exists. The real sequence is one, so a
    real partition is always feasible. ``log10_orders`` near zero means the placebo
    can barely move the block's states.
    """
    codes, uniques = pd.factorize(labels, sort=True)
    if (codes < 0).any():
        raise ValueError("labels contain missing values")
    ids = (np.zeros(len(labels), dtype=object) if blocks is None
           else blocks.reindex(labels.index).to_numpy())
    starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
    bounds = np.r_[starts, len(labels)]
    rows = []
    for b in range(len(starts)):
        segment = codes[bounds[b] : bounds[b + 1]]
        lengths, states = run_lengths(segment)
        counts = np.bincount(np.asarray(states, dtype=np.int64), minlength=len(uniques))
        log_count = JoinFreeSequences(counts).log_count
        row: dict[str, object] = {
            "block": "all" if blocks is None else ids[bounds[b]],
            "points": int(segment.size),
            "episodes": int(len(states)),
            "transitions": int(len(states) - 1),
            "longest": int(lengths.max()),
        }
        row.update({f"episodes_{u}": int(c) for u, c in zip(uniques, counts, strict=True)})
        row["log10_orders"] = float(log_count / np.log(10.0))
        row["feasible"] = bool(np.isfinite(log_count))
        rows.append(row)
    return pd.DataFrame(rows).set_index("block")


def session_occupancy_deviation(real: pd.Series, draws: pd.DataFrame) -> float:
    """Largest gap between a draw's session-level occupancy and the real one (OC-P1b)."""
    truth = real.dropna().value_counts(normalize=True)
    worst = 0.0
    for column in draws.columns:
        drawn = draws[column].dropna().value_counts(normalize=True)
        cells = truth.index.union(drawn.index)
        gap = (drawn.reindex(cells, fill_value=0.0) - truth.reindex(cells, fill_value=0.0))
        worst = max(worst, float(gap.abs().max()))
    return worst


# ----------------------------------------------------------------------- null / power


@dataclass(frozen=True)
class NullDistribution:
    """Draws of a statistic under a content-free partition, and what they can resolve."""

    values: np.ndarray

    @property
    def n(self) -> int:
        return int(self.values.size)

    @property
    def finite(self) -> bool:
        """Every draw finite. The programme never drops a draw (Two Sigma §13.4)."""
        return bool(self.values.size and np.isfinite(self.values).all())

    @property
    def sd(self) -> float:
        return float(np.std(self.values, ddof=1)) if self.finite else float("nan")

    @property
    def skewness(self) -> float:
        if not self.finite or np.ptp(self.values) == 0:
            return float("nan")
        return float(stats.skew(self.values, bias=False))

    @property
    def n_distinct(self) -> int:
        return int(np.unique(self.values).size)

    @property
    def atom_share(self) -> float:
        """Largest share of draws taken by one single value (Two Sigma `atom_share`)."""
        if not self.finite:
            return float("nan")
        _, counts = np.unique(self.values, return_counts=True)
        return float(counts.max() / self.values.size)

    @property
    def is_atom(self) -> bool:
        """One value holds at least `ATOM_SHARE` of the draws: shift MDEs mean nothing."""
        return bool(self.atom_share >= ATOM_SHARE) if self.finite else True

    def quantile(self, q: float | Sequence[float]) -> float | np.ndarray:
        if not self.finite:
            return np.full(np.shape(q), np.nan) if np.ndim(q) else float("nan")
        out = np.quantile(self.values, q)
        return float(out) if np.ndim(q) == 0 else out

    def mde_shift(self, alpha: float = FAMILY_ALPHA, power: float = POWER) -> float:
        """One-sided location-shift MDE: ``q(1 − α) − q(1 − power)`` (OC-N1).

        This is the shift of the null that puts ``power`` of it above its own
        ``1 − α`` quantile (Two Sigma's ``MDE_C = q99 − q20``). NaN on an atom.
        """
        if self.is_atom:
            return float("nan")
        return float(self.quantile(1.0 - alpha) - self.quantile(1.0 - power))

    def mde_sd(self, alpha: float = FAMILY_ALPHA, power: float = POWER) -> float:
        """One-sided Gaussian MDE: ``(z_{1−α} + z_power) × sd`` (OC-N1)."""
        return float((stats.norm.ppf(1.0 - alpha) + stats.norm.ppf(power)) * self.sd)

    def percentile(self, value: float) -> float:
        """Share of draws below ``value``, ties half (`selection.protocol`)."""
        return placebo_percentile(value, self.values)

    def p_value(self, value: float) -> float:
        """``(1 + #{draws ≥ value}) / (1 + n)`` (`selection.protocol`)."""
        return placebo_p_value(value, self.values)

    def describe(self) -> dict[str, float]:
        """Label-free summary of the null, safe to print before the lock."""
        q = self.quantile([0.05, 0.20, 0.50, 0.80, 0.95, 0.99])
        return {
            "draws": float(self.n),
            "finite": float(self.finite),
            "mean": float(np.mean(self.values)) if self.finite else float("nan"),
            "sd": self.sd,
            "skewness": self.skewness,
            "atom_share": self.atom_share,
            "n_distinct": float(self.n_distinct),
            "q05": float(q[0]), "q20": float(q[1]), "q50": float(q[2]),
            "q80": float(q[3]), "q95": float(q[4]), "q99": float(q[5]),
            "mde_shift_05": self.mde_shift(0.05),
            "mde_shift_sidak": self.mde_shift(sidak_alpha()),
            "mde_sd_05": self.mde_sd(0.05),
            "mde_sd_sidak": self.mde_sd(sidak_alpha()),
        }


def _placebo_paths(
    engine: LevelA,
    labels: pd.Series,
    n_draws: int,
    *,
    seed: int,
    blocks: pd.Series | None,
    lag: int,
    stamped: bool,
) -> Iterator[pd.Series]:
    if stamped:
        stamps = stamps_in_force(labels, engine.sessions, lag=lag)
        grouped = None if blocks is None else blocks.reindex(stamps.index)
        placebo = p1_draws(stamps, n_draws, seed=seed, blocks=grouped)
        pos = _asof_positions(pd.DatetimeIndex(stamps.index), engine.sessions, lag)
        for column in placebo.draws.columns:
            values = placebo.draws[column].to_numpy(float)[np.clip(pos, 0, None)]
            yield pd.Series(np.where(pos >= 0, values, np.nan), index=engine.sessions)
        return
    in_force = labels.reindex(engine.sessions)
    defined = np.flatnonzero(in_force.notna().to_numpy())
    if defined.size == 0:
        raise ValueError("no session carries a label")
    if defined.size != defined[-1] - defined[0] + 1:
        raise ValueError("session labels may be missing only before or after the labelled span")
    span = in_force.iloc[defined[0] : defined[-1] + 1]
    grouped = None if blocks is None else blocks.reindex(span.index)
    placebo = p1_draws(span, n_draws, seed=seed, blocks=grouped)
    for column in placebo.draws.columns:
        yield placebo.draws[column].astype(float).reindex(engine.sessions)


def null_statistics(
    engine: LevelA,
    labels: pd.Series,
    n_draws: int = PLACEBO_DRAWS,
    *,
    statistics: Mapping[str, Callable[[LevelAResult], float]],
    seed: int = 0,
    blocks: pd.Series | None = None,
    lag: int = 1,
    stamped: bool = True,
) -> dict[str, NullDistribution]:
    """Run level A on P1 draws and record each statistic (§3 A condition 2, §4, OC-N1).

    Input:
    - With ``stamped=True``, ``labels`` is the stamped quarterly sequence (index
      ``available_at``). The draws are made on the stamps in force over the engine's
      sessions, then expanded with the same lag as the real label.
    - With ``stamped=False``, ``labels`` is already a session path in force, for example
      B1's daily labels, and the draws are made on it directly.

    Each draw:
    - The draw refits the balanced leg on its own training cells and reads D on its
      own test cells, the whole pipeline.
    - The draws carry no content: each one keeps the real clock and occupancy and
      nothing else. This is the null computation the blindness rule allows before the
      lock.
    - ``statistics`` maps a name to a function of the draw's `LevelAResult`. Examples:
      the reduction, a leg's mean gross exposure for P2's T3 reading, or the share of
      sessions on which the cap binds.
    """
    out: dict[str, list[float]] = {name: [] for name in statistics}
    for path in _placebo_paths(engine, labels, n_draws, seed=seed, blocks=blocks, lag=lag,
                               stamped=stamped):
        result = engine.run(path)
        for name, fn in statistics.items():
            out[name].append(float(fn(result)))
    return {name: NullDistribution(np.asarray(v, dtype=float)) for name, v in out.items()}


def null_reductions(
    engine: LevelA,
    labels: pd.Series,
    n_draws: int = PLACEBO_DRAWS,
    *,
    seed: int = 0,
    blocks: pd.Series | None = None,
    lag: int = 1,
    stamped: bool = True,
) -> NullDistribution:
    """The null of the reduction in D under P1: its percentile rule and its MDE (§3 A, §4).

    The same distribution serves the P1 percentile of §3 A condition 2 (after the lock)
    and the MDE of the declared statistic (before it). This function draws only
    content-free partitions, so it may run on real returns before the lock.
    """
    return null_statistics(
        engine, labels, n_draws, statistics={"reduction": lambda r: r.reduction}, seed=seed,
        blocks=blocks, lag=lag, stamped=stamped,
    )["reduction"]


def placebo_session_paths(
    engine: LevelA,
    labels: pd.Series,
    n_draws: int = PLACEBO_DRAWS,
    *,
    seed: int = 0,
    blocks: pd.Series | None = None,
    lag: int = 1,
    stamped: bool = True,
) -> pd.DataFrame:
    """The P1 draws of :func:`null_statistics`, as session paths (sessions × draws).

    Column ``d`` is exactly the path that :func:`null_statistics` runs as its draw ``d``
    with the same arguments, so a caller can split the draws across processes, or
    check them, without re-deriving the construction. Content-free by construction.
    """
    paths = list(_placebo_paths(engine, labels, n_draws, seed=seed, blocks=blocks, lag=lag,
                                stamped=stamped))
    return pd.DataFrame(np.column_stack([p.to_numpy(float) for p in paths]),
                        index=engine.sessions, columns=pd.RangeIndex(n_draws, name="draw"))


# ---------------------------------------------------------------------------------- P2


def gross_exposure(held: pd.DataFrame) -> pd.Series:
    """``Σ|w|`` of the weights held on each session, NaN where nothing is held."""
    return held.abs().sum(axis=1, min_count=1).rename("gross")


def ex_ante_volatility(
    held: pd.DataFrame,
    asset_returns: pd.DataFrame,
    *,
    window: int = VOL_WINDOW,
    missing_as_zero: bool = False,
) -> pd.Series:
    """Annualised ``√(w_t' Σ_{t−1} w_t)``. ``Σ_{t−1}`` is the sample covariance of
    ``asset_returns`` over the ``window`` sessions ending at t−1 (OC-P2a).

    Every leg is read through the same estimator, and it is causal: the weights held
    on t are known at t−1, and so is the covariance. NaN until a full window of finite
    returns exists. With ``missing_as_zero`` a missing return counts as a zero return,
    the convention of the book itself (`sleeves.build_book`), so a single missing price
    does not leave the next ``window`` sessions undefined (the lock's P2, §12.11).
    """
    x = asset_returns.reindex(index=held.index, columns=held.columns).to_numpy(float)
    if missing_as_zero:
        x = np.where(np.isfinite(x), x, 0.0)
    n, p = x.shape
    ok = np.isfinite(x).all(axis=1)
    z = np.where(ok[:, None], x, 0.0)
    c1 = np.vstack([np.zeros((1, p)), np.cumsum(z, axis=0)])
    c2 = np.concatenate([np.zeros((1, p, p)),
                         np.cumsum(z[:, :, None] * z[:, None, :], axis=0)])
    count = np.r_[0, np.cumsum(ok)]
    end = np.arange(1, n + 1)
    begin = np.clip(end - window, 0, None)
    full = (end - window >= 0) & (count[end] - count[begin] == window)
    s1 = c1[end] - c1[begin]
    s2 = c2[end] - c2[begin]
    cov = (s2 - s1[:, :, None] * s1[:, None, :] / window) / (window - 1)
    cov[~full] = np.nan
    lagged = np.full_like(cov, np.nan)
    lagged[1:] = cov[:-1]
    w = held.to_numpy(float)
    var = np.einsum("ti,tij,tj->t", w, lagged, w)
    vol = np.sqrt(np.where(var >= 0, var, np.nan) * PERIODS)
    return pd.Series(vol, index=held.index, name="ex_ante_vol")


@dataclass(frozen=True)
class ExposureMatch:
    """D of both legs after the P2 rescaling (§6 P2), and the factors applied."""

    on: str
    d_blind: float
    d_balanced: float
    blind_scale: pd.Series
    balanced_scale: pd.Series

    @property
    def reduction(self) -> float:
        return self.d_blind - self.d_balanced


def exposure_matched(
    result: LevelAResult,
    asset_returns: pd.DataFrame,
    *,
    on: Literal["gross", "vol"],
    target: float = VOL_TARGET,
    window: int = VOL_WINDOW,
    missing_as_zero: bool = False,
) -> ExposureMatch:
    """§6 P2, the plainest operational readings: rescale session by session, reread D.

    Readings:
    - ``on="gross"`` rescales the balanced leg on every session to the blind leg's
      gross exposure. The blind leg is untouched.
    - ``on="vol"`` rescales both legs to ``target`` ex-ante volatility, through one
      common covariance estimator (:func:`ex_ante_volatility` on ``asset_returns``,
      which must match the level of ``held``).

    Why a constant scale is not enough: D is scale-free, so matching on MEAN gross
    cannot move it (OC-P2a). Each factor is known at t−1. Blindness: on real labels
    this reads D.
    """
    idx = result.test_labels.index
    rb = result.blind_leg.returns
    ro = result.balanced_leg.returns
    ones = pd.Series(1.0, index=rb.index)
    if on == "gross":
        gb = gross_exposure(result.blind_leg.held)
        go = gross_exposure(result.balanced_leg.held)
        blind_scale = ones
        balanced_scale = (gb / go.replace(0.0, np.nan)).reindex(rb.index)
    elif on == "vol":
        vb = ex_ante_volatility(result.blind_leg.held, asset_returns, window=window,
                                missing_as_zero=missing_as_zero)
        vo = ex_ante_volatility(result.balanced_leg.held, asset_returns, window=window,
                                missing_as_zero=missing_as_zero)
        blind_scale = (target / vb.replace(0.0, np.nan)).reindex(rb.index)
        balanced_scale = (target / vo.replace(0.0, np.nan)).reindex(rb.index)
    else:
        raise ValueError(f"on must be 'gross' or 'vol', not {on!r}")
    xb = (rb * blind_scale).reindex(idx).to_numpy(float)
    xo = (ro * balanced_scale).reindex(idx).to_numpy(float)
    codes = _codes(result.test_labels.to_numpy(float), result.settings.n_cells)
    codes = np.where(np.isfinite(xb) & np.isfinite(xo), codes, -1)
    d_blind, d_other, _ = result.settings.pair(xb, xo, codes)
    return ExposureMatch(on, d_blind, d_other, blind_scale.rename("scale"),
                         balanced_scale.rename("scale"))


# ----------------------------------------------------------------------------- witness


def trailing_log_vol(returns: pd.Series, *, window: int = WITNESS_WINDOW) -> pd.Series:
    """``log(√252 · sd)`` over the ``window`` sessions ending at t (through t, not lagged)."""
    sd = returns.rolling(window).std(ddof=1) * np.sqrt(PERIODS)
    return np.log(sd.where(sd > 0)).rename("log_rv")


def volatility_witness_labels(
    driver: pd.Series,
    folds: Sequence[SplitLike],
    *,
    sessions: pd.DatetimeIndex | None = None,
    window: int = WITNESS_WINDOW,
    n_bins: int = N_CELLS,
    lag: int = 1,
    grid: pd.DatetimeIndex | None = None,
) -> dict[int, pd.Series]:
    """The volatility witness: ``n_bins`` quantile bins of trailing log realised volatility.

    This is the programme's standing one-line adversary, set in the quadrant's place
    (Two Sigma W1, OC-W).
    - Each fold's cut-offs are the quantiles of ``log_rv`` over its training sessions.
    - A bin is stamped at t on volatility through t and held from t + ``lag``.
    - With ``grid``, for example the availability stamps, the bin is sampled at each
      grid date and held to the next, which gives the quadrant's own clock (OC-W3).

    Returns one path per fold number, for `LevelA.run`. Downgrade only (OC-W4).
    """
    sessions = pd.DatetimeIndex(driver.index if sessions is None else sessions)
    log_rv = trailing_log_vol(driver.reindex(sessions), window=window)
    values = log_rv.to_numpy(float)
    out: dict[int, pd.Series] = {}
    for fold in folds:
        reference = log_rv.reindex(fold.train(sessions)).dropna().to_numpy(float)
        if reference.size < n_bins:
            raise ValueError(f"fold {fold.number}: too few training sessions for {n_bins} bins")
        cuts = np.quantile(reference, np.arange(1, n_bins) / n_bins)
        bins = np.where(np.isfinite(values), np.searchsorted(cuts, values, side="right"),
                        np.nan)
        stamped = pd.Series(bins, index=sessions, name="witness")
        if grid is None:
            path = stamped.shift(lag)
        else:
            at_grid = expand_to_sessions(stamped, pd.DatetimeIndex(grid), lag=0).dropna()
            path = expand_to_sessions(at_grid, sessions, lag=lag)
        out[fold.number] = path.rename("witness")
    return out


def witness_dominates(real_reduction: float, witness_reduction: float) -> bool:
    """OC-W4: the one-line volatility partition does at least as well. Downgrade only."""
    return bool(np.isfinite(real_reduction) and np.isfinite(witness_reduction)
                and witness_reduction >= real_reduction)


# ------------------------------------------------------------------------------ B2 / P3


def map_risk_budgets(
    mapping: Mapping[str, frozenset[str] | set[str]],
    *,
    sleeves: Sequence[str] = SLEEVES,
    tags: Sequence[str] = TAGS,
    empty: Literal["raise", "renormalise"] = "raise",
) -> pd.Series:
    """§3 B2: the risk budget a sleeve → tags map implies (OC-B2a).

    Each tag receives ``1/len(tags)`` of the risk, split equally among the sleeves
    mapped to it. A sleeve's budget is the sum of its shares. For the draft's map:
    - equity 1/8;
    - duration 3/8;
    - inflation-linked 1/12;
    - commodity 5/24;
    - precious metals 5/24.

    A tag no sleeve carries raises, or with ``empty="renormalise"`` is dropped and the
    rest rescaled.
    """
    carriers = {tag: [s for s in sleeves if tag in mapping.get(s, frozenset())] for tag in tags}
    unknown = set().union(*[set(mapping.get(s, ())) for s in sleeves]) - set(tags)
    if unknown:
        raise ValueError(f"unknown tags {sorted(unknown)}")
    covered = [tag for tag in tags if carriers[tag]]
    if len(covered) < len(tags) and empty == "raise":
        raise ValueError(f"no sleeve carries {sorted(set(tags) - set(covered))}")
    if not covered:
        raise ValueError("no tag is carried")
    budget = pd.Series(0.0, index=list(sleeves), name="budget")
    for tag in covered:
        budget[carriers[tag]] += 1.0 / len(covered) / len(carriers[tag])
    return budget


def budget_weights(
    budgets: np.ndarray, moment: np.ndarray, *, rule: BudgetRule = "inverse_vol"
) -> np.ndarray:
    """Weights from risk budgets, long only, summing to one (OC-B2a).

    Rules:
    - ``"inverse_vol"``: ``w ∝ b/σ``, which uses no covariance, as §3 B2 asks.
    - ``"erc"``: `risk_parity_mix` with budgets.
    - ``"notional"``: ``w ∝ b``.

    A zero budget gives a zero weight.
    """
    b = np.asarray(budgets, dtype=float)
    m = np.asarray(moment, dtype=float)
    if (b < 0).any() or not b.sum() > 0:
        raise ValueError("budgets must be non-negative with a positive sum")
    if rule == "notional":
        w = b.copy()
    elif rule == "inverse_vol":
        w = b / np.sqrt(np.diag(m))
    elif rule == "erc":
        live = b > 0
        w = np.zeros_like(b)
        if live.sum() == 1:
            w[live] = 1.0
        else:
            w[live] = np.asarray(risk_parity_mix(m[np.ix_(live, live)], budgets=b[live]))
    else:
        raise ValueError(f"unknown budget rule {rule!r}")
    return w / w.sum()


def axis_consistent_sets(tags: Sequence[str] = TAGS) -> tuple[frozenset[str], ...]:
    """The eight non-empty tag sets with at most one sign per axis (OC-B2b)."""
    growth = [None, *[t for t in tags if t.startswith("g")]]
    inflation = [None, *[t for t in tags if t.startswith("i")]]
    sets = [frozenset(x for x in (g, i) if x) for g in growth for i in inflation]
    return tuple(s for s in sets if s)


def _allowed_sets(
    space: MapSpace, sleeves: Sequence[str], reference: Mapping[str, frozenset[str]],
    tags: Sequence[str],
) -> list[list[frozenset[str]]]:
    consistent = axis_consistent_sets(tags)
    if space == "axis_consistent":
        return [list(consistent) for _ in sleeves]
    if space == "profile":
        return [[s for s in consistent if len(s) == len(reference[name])] for name in sleeves]
    if space == "any_subset":
        every = [frozenset(c) for r in range(1, len(tags) + 1)
                 for c in itertools.combinations(tags, r)]
        return [every for _ in sleeves]
    raise ValueError(f"unknown map space {space!r}")


def enumerate_maps(
    space: MapSpace,
    *,
    sleeves: Sequence[str] = SLEEVES,
    reference: Mapping[str, frozenset[str]] = DRAFT_MAP,
    tags: Sequence[str] = TAGS,
    require_all_tags: bool = True,
) -> Iterator[dict[str, frozenset[str]]]:
    """Every sleeve → tags map of one space, lazily and in a fixed order (OC-B2b).

    The spaces:
    - ``permutation``: the reference's tag sets reassigned among the sleeves.
    - ``profile``: every sleeve keeps its number of tags, at most one sign per axis.
    - ``axis_consistent``: any non-empty set with at most one sign per axis.
    - ``any_subset``: any non-empty set.

    ``require_all_tags`` keeps only the maps that leave no tag uncovered.
    """
    if space == "permutation":
        assignments: Iterator[tuple[frozenset[str], ...]] = itertools.permutations(
            [frozenset(reference[s]) for s in sleeves])
    else:
        assignments = itertools.product(*_allowed_sets(space, sleeves, reference, tags))
    everything = set(tags)
    for sets in assignments:
        if require_all_tags and set().union(*sets) != everything:
            continue
        yield dict(zip(sleeves, sets, strict=True))


def map_space_size(
    space: MapSpace,
    *,
    sleeves: Sequence[str] = SLEEVES,
    reference: Mapping[str, frozenset[str]] = DRAFT_MAP,
    tags: Sequence[str] = TAGS,
    require_all_tags: bool = True,
) -> int:
    """Number of maps in a space, by inclusion–exclusion over uncovered tags (no listing)."""
    if space == "permutation":
        return int(np.prod(np.arange(1, len(sleeves) + 1)))
    allowed = _allowed_sets(space, sleeves, reference, tags)
    if not require_all_tags:
        return int(np.prod([len(a) for a in allowed], dtype=object))
    total = 0
    for r in range(len(tags) + 1):
        for missing in itertools.combinations(tags, r):
            gone = set(missing)
            count = 1
            for options in allowed:
                count *= sum(1 for s in options if not (s & gone))
            total += (-1) ** r * count
    return int(total)


@dataclass(frozen=True)
class MapSpaceSummary:
    """How finely P3 can rank the draft's map within one space (OC-B2b, OC-B2c)."""

    space: str
    maps: int
    budget_vectors: int
    reference_ties: int

    @property
    def min_p_value(self) -> float:
        """Smallest attainable p, with the real map in its own null and ties against."""
        return self.reference_ties / self.maps

    @property
    def max_percentile(self) -> float:
        """Largest attainable percentile, with ties counted half."""
        return 1.0 - self.reference_ties / (2.0 * self.maps)


def _budget_key(budget: pd.Series) -> tuple[float, ...]:
    return tuple(np.round(budget.to_numpy(float), 12).tolist())


def map_space_summary(
    space: MapSpace,
    *,
    sleeves: Sequence[str] = SLEEVES,
    reference: Mapping[str, frozenset[str]] = DRAFT_MAP,
    tags: Sequence[str] = TAGS,
    require_all_tags: bool = True,
) -> MapSpaceSummary:
    """Count the maps, their distinct budget vectors, and the reference's ties.

    Every map is enumerated. Each budget vector is computed with ``empty="renormalise"``,
    so this also runs when ``require_all_tags=False``.
    """
    counts: dict[tuple[float, ...], int] = {}
    total = 0
    for mapping in enumerate_maps(space, sleeves=sleeves, reference=reference, tags=tags,
                                  require_all_tags=require_all_tags):
        key = _budget_key(map_risk_budgets(mapping, sleeves=sleeves, tags=tags,
                                           empty="renormalise"))
        counts[key] = counts.get(key, 0) + 1
        total += 1
    ref = _budget_key(map_risk_budgets(reference, sleeves=sleeves, tags=tags))
    return MapSpaceSummary(space, total, len(counts), counts.get(ref, 0))


def b2_fold_weights(
    engine: LevelA,
    mapping: Mapping[str, frozenset[str]],
    *,
    rule: BudgetRule = "inverse_vol",
    tags: Sequence[str] = TAGS,
    empty: Literal["raise", "renormalise"] = "raise",
) -> np.ndarray:
    """B2's sleeve weights per fold, from the map's budgets and training moments only.

    The budgets come from the map, with ``empty`` deciding what an uncovered tag does.
    The volatilities (or the matrix, for ``rule="erc"``) come from every finite training
    session of the fold, whatever its label: no cell is estimated (§3 B2).
    """
    budget = map_risk_budgets(mapping, sleeves=list(engine.sleeves), tags=tags,
                              empty=empty).to_numpy(float)
    return np.vstack([
        budget_weights(budget, _moment(engine._x[t][engine._finite[t]], engine.about_zero),
                       rule=rule)
        for t in engine._train
    ])


def p3_reductions(
    engine: LevelA,
    labels: pd.Series | Mapping[int, pd.Series],
    maps: Sequence[Mapping[str, frozenset[str]]],
    *,
    rule: BudgetRule = "inverse_vol",
    tags: Sequence[str] = TAGS,
) -> np.ndarray:
    """§6 P3: the reduction in D of every map in ``maps``, against the same blind leg.

    Maps with the same budget vector give the same book, so each distinct vector is
    run once. A map that leaves a tag uncovered is renormalised. Blindness: on real
    labels this reads D.
    """
    cache: dict[tuple[float, ...], float] = {}
    out = np.empty(len(maps))
    for i, mapping in enumerate(maps):
        budget = map_risk_budgets(mapping, sleeves=list(engine.sleeves), tags=tags,
                                  empty="renormalise")
        key = _budget_key(budget)
        if key not in cache:
            weights = b2_fold_weights(engine, mapping, rule=rule, tags=tags, empty="renormalise")
            cache[key] = engine.run(labels, weights=weights).reduction
        out[i] = cache[key]
    return out


# ------------------------------------------------------------------------------ level C


def cadence_weights(target: pd.DataFrame, rebalance: pd.DatetimeIndex) -> pd.DataFrame:
    """Hold ``target`` on each rebalance session and keep the previous weights otherwise.

    This is the Two Sigma C-1 convention, with no drift (OC-C5). The first session on
    which the target is fully defined is a rebalance. Before it the book holds nothing
    (NaN). A rebalance on a session where the target is undefined keeps the previous
    weights.
    """
    _check_calendar(target.index, "target")
    defined = target.notna().all(axis=1).to_numpy()
    if not defined.any():
        raise ValueError("the target is never fully defined")
    first = int(np.flatnonzero(defined)[0])
    marks = np.asarray(target.index.isin(pd.DatetimeIndex(rebalance))) & defined
    marks[first] = True
    held = target.where(np.broadcast_to(marks[:, None], target.shape)).ffill()
    held.iloc[:first] = np.nan
    return held


@dataclass(frozen=True)
class CadenceBook:
    """A book rebalanced on given sessions: held weights, net returns, rebalances used."""

    weights: pd.DataFrame
    returns: pd.Series
    rebalances: pd.DatetimeIndex


def cadence_book(
    target: pd.DataFrame,
    returns: pd.DataFrame,
    rebalance: pd.DatetimeIndex,
    *,
    bps: float | pd.Series,
    vol_target: bool = False,
    target_vol: float = VOL_TARGET,
    window: int = VOL_WINDOW,
    cap: float = MAX_LEVERAGE,
) -> CadenceBook:
    """§3 C: the construction held at a cadence, net of costs on ``|Δw|`` of held weights.

    - ``target`` is the construction's causal weight on each session, sessions ×
      assets, already lagged.
    - With ``vol_target``, the cadence mix is scaled daily to §2's target (the
      multiplier is not a rebalance), and costs are charged on mix × multiplier.
    """
    mix = cadence_weights(target, rebalance)
    r = returns.reindex(index=mix.index, columns=mix.columns)
    ok = mix.notna().all(axis=1) & r.notna().all(axis=1)
    gross = (mix * r).sum(axis=1).where(ok).rename("book")
    held = mix
    if vol_target:
        gross, multiplier, _ = scale_to_target(gross, target=target_vol, window=window, cap=cap)
        held = mix.mul(multiplier, axis=0)
    rates = (bps.reindex(mix.columns) if isinstance(bps, pd.Series)
             else pd.Series(float(bps), index=mix.columns))
    net = charge_by_instrument(gross, held, rates)
    used = mix.index[np.asarray(mix.index.isin(pd.DatetimeIndex(rebalance)))]
    return CadenceBook(held, net.rename("book"), pd.DatetimeIndex(used))


def tracking_distance(weights: pd.DataFrame, target: pd.DataFrame) -> pd.Series:
    """``δ_t = Σ|w_t − w*_t|`` where both are defined (Two Sigma §12.10)."""
    w = weights.reindex_like(target)
    ok = w.notna().all(axis=1) & target.notna().all(axis=1)
    return (w - target).abs().sum(axis=1).where(ok).rename("tracking")


def _sharpe(x: pd.Series) -> float:
    x = x.dropna()
    sd = x.std(ddof=1)
    return float(np.sqrt(PERIODS) * x.mean() / sd) if len(x) > 1 and sd > 0 else float("nan")


@dataclass(frozen=True)
class CadenceComparison:
    """§3 C: a book rebalanced on transition dates against a calendar (OC-C3 candidates)."""

    sharpe_conditional: float
    sharpe_calendar: float
    turnover_conditional: float
    turnover_calendar: float
    tracking_conditional: float
    tracking_calendar: float
    rebalances_per_year_conditional: float
    rebalances_per_year_calendar: float

    @property
    def sharpe_difference(self) -> float:
        """Paired net Sharpe, conditional minus calendar."""
        return self.sharpe_conditional - self.sharpe_calendar

    @property
    def turnover_saving(self) -> float:
        """Calendar minus conditional turnover, ×/yr (Two Sigma C-1's ``S_C``)."""
        return self.turnover_calendar - self.turnover_conditional

    @property
    def tracking_difference(self) -> float:
        """Conditional minus calendar mean tracking distance. Positive means it tracks worse."""
        return self.tracking_conditional - self.tracking_calendar

    def statistic(self, name: CStatistic) -> float:
        return float(getattr(self, name))


def level_c_compare(
    target: pd.DataFrame,
    returns: pd.DataFrame,
    conditional: pd.DatetimeIndex,
    calendar: pd.DatetimeIndex,
    *,
    bps: float | pd.Series,
    evaluate: pd.DatetimeIndex,
    vol_target: bool = False,
) -> CadenceComparison:
    """§3 C on the ``evaluate`` sessions (the pooled test folds).

    - Sharpe and turnover are read on the book as held (after the multiplier when
      ``vol_target`` is set). Turnover is restricted to the sessions first, then
      differenced, as in Two Sigma §12.10.
    - Tracking compares the cadence mix with the target mix. The daily multiplier is
      not a rebalance, so it is left out of that comparison.
    - Blindness: with the real transition dates this reads a performance, which is
      forbidden before the lock.
    """
    evaluate = pd.DatetimeIndex(evaluate)
    years = len(evaluate) / PERIODS
    values: dict[str, float] = {}
    for name, dates in (("conditional", conditional), ("calendar", calendar)):
        book = cadence_book(target, returns, dates, bps=bps, vol_target=vol_target)
        mix = cadence_weights(target, dates)
        values[f"sharpe_{name}"] = _sharpe(book.returns.reindex(evaluate))
        values[f"turnover_{name}"] = annual_turnover(book.weights.reindex(evaluate))
        values[f"tracking_{name}"] = float(
            tracking_distance(mix, target).reindex(evaluate).mean())
        values[f"rebalances_per_year_{name}"] = float(
            np.asarray(book.rebalances.isin(evaluate)).sum() / years)
    return CadenceComparison(**values)


def placebo_rebalances(
    draws: pd.DataFrame, sessions: pd.DatetimeIndex, *, lag: int = 1
) -> list[pd.DatetimeIndex]:
    """§3 C placebo: the transition sessions of each P1 draw of the stamped labels (OC-C4).

    A P1 draw keeps the transition count and the multiset of episode lengths in
    quarters, which is the spacing law, exactly within each block.
    """
    return [stamps_to_sessions(transition_stamps(draws[c]), sessions, lag=lag)
            for c in draws.columns]


def level_c_null(
    target: pd.DataFrame,
    returns: pd.DataFrame,
    placebos: Sequence[pd.DatetimeIndex],
    calendar: pd.DatetimeIndex,
    *,
    bps: float | pd.Series,
    evaluate: pd.DatetimeIndex,
    statistic: CStatistic = "sharpe_difference",
    vol_target: bool = False,
) -> NullDistribution:
    """The level-C statistic on every placebo set of dates, against the same calendar."""
    return NullDistribution(np.array([
        level_c_compare(target, returns, dates, calendar, bps=bps, evaluate=evaluate,
                        vol_target=vol_target).statistic(statistic)
        for dates in placebos
    ], dtype=float))

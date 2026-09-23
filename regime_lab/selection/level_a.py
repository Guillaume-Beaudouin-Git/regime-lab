"""Level A of the Two Sigma tree — the first-moment channel (`docs/PRESPEC_TWOSIGMA.md`).

The specification is `docs/PRESPEC_TWOSIGMA.md`, **LOCKED at commit 30f7d69**. This module
implements, on the shared core :mod:`regime_lab.selection.tree`:

- the level-A map ``m_ik`` of §12.5 (:func:`training_profile`, :func:`estimate_m`);
- the selector and its traded path, §4, §12.3, §12.6 "Arms" (:func:`selector_mix`);
- the A-1 instrument, §12.6 steps 1-6, including the NFCI diagnostic of step 6
  (:func:`nfci_diagnostic`), and the A-2 instrument, §12.7 (:func:`instrument`);
- the readings of A-1 and A-2, their placebo arms and nulls (§12.6 step 4, §12.7,
  §12.11), the volatility witness W1 (§8 control 4), the point-in-time rebuild on the
  18-feature partition (§8 control 5) and the d sensitivity (§12.13) (:func:`read`);
- the verdict lines of §12.6 (first line that applies) and §12.7 (:func:`a1_verdict`,
  :func:`a2_verdict`), and the §13.5 trial-row metrics (:func:`trial_rows`).

**Blindness (§11, §13.1 step 2).** Nothing here prints. :func:`instrument` returns only
what §13.1 step 2 allows at level A: counts (cells, abstentions, fallbacks), the turnover
of the selector and of the control with ΔT, the kill line, the realised sd and cap share
of both arms, the demeaned-leg MDEs, T_A1 and SE*, the 50th/95th/99th percentiles of the
A-2 null with q99 − q50, and, for the primary, the NFCI diagnostic. It computes ``m``
from returns on the training folds to build the selector's legs and never returns it or
any per-state quantity; it never computes a Sharpe, a mean, the real ``R``, or any
placebo arm's Δ or β (§12.6 step 4: the reading builds those). The ``repr`` of every
object holding ``m`` shows counts only.

**The reading cannot run by accident (§13.1 step 4).** :func:`read` first recomputes the
instrument and passes it to ``tree.verify_for_reading``: the threshold file must be
tracked by git and identical to HEAD, every input's SHA-256 must match, the package
versions stored, pinned by ``uv.lock`` and installed must agree, and the recomputed
thresholds must equal the committed ones bitwise. Otherwise it raises
``tree.ReadingRefused`` before any statistic is computed. Trial-row metrics are
returned only at the lock's constants (1,000 placebo draws, 2,000 bootstrap draws).

**Non-finite readings (§13.4).** A non-finite map, profile, threshold, statistic or
placebo draw propagates as NaN; every verdict function returns UNDECIDABLE on it and no
verdict is read off a NaN. No draw is dropped.

**Choices this module had to make where the lock's text did not fix the computation**
are stated in the docstring of the function that makes them and repeated in the
build's report; none eases a PASS.
"""

from __future__ import annotations

import json
import os
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from threadpoolctl import threadpool_limits

from regime_lab.analysis.placebo import transitions
from regime_lab.config import ROOT
from regime_lab.selection import protocol, tree
from regime_lab.selection.context import LOG_RV, align_labels, eta_squared
from regime_lab.selection.tree import (
    BONFERRONI,
    BOOT_DRAWS,
    BOOT_SEED,
    COST_COLUMNS,
    DECIDING_BPS,
    INPUT_FILES,
    N_DRAWS,
    NAN,
    PASS,
    PIT18,
    PLACEBO_METHOD,
    PLACEBO_SEED,
    POWER_BLOCKS,
    PRIMARY,
    THRESHOLDS_PATH,
    UNDECIDABLE,
    BookResult,
    Cells,
    Kill,
    PairThreshold,
    Partition,
    PlaceboDraws,
    TradedMix,
    TreeData,
    Variant,
)

#: The threshold-file section of the primary's level-A instrument (§13.1 step 3).
SECTION = "A"
SQRT_PERIODS = float(np.sqrt(protocol.PERIODS))


def section_name(variant: Variant) -> str:
    """The threshold-file section of a variant's level-A instrument (``tree.section_name``):
    ``A`` for the primary, ``A:<variant>`` for a §12.13 sensitivity (``A:K=3``,
    ``A:d=0.25``). The PIT rebuild (§8 control 5) has no section: its thresholds are
    computed at the reading, never committed."""
    return tree.section_name(SECTION, variant)


def pit_variant(variant: Variant) -> Variant:
    """§8 control 5: the variant rebuilt on the 18 features without the NFCI
    (``tree.pit_variant``): the same K, smoothing and d, and the same ``n_init``, seed,
    folds and training start. For the primary this is ``tree.PIT18``. *Choice:* §8 says
    "a lock that would otherwise PASS is rebuilt ... with the same K"; a sensitivity row's
    would-be PASS is rebuilt at its own K, smoothing and d, since §12.13 reruns "each
    level's full evaluation" at the variant. Downgrade only.
    """
    return tree.pit_variant(variant)


# ---------------------------------------------------------------------------------------
# the per-fold layout of a stamped path — §12.2, §12.4
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False, repr=False)
class _Layout:
    """Where each fold's rows sit in a paths Series, and the legs on its sessions.

    Row ``start[i]:stop[i]`` of the paths is fold ``folds[i]``'s training-then-test
    path; its first ``n_train[i]`` rows are the training segment. ``x[i]`` holds the ten
    signal legs (§12.3) on those sessions, in that order. Built once per path, so the
    real partition and every placebo draw run through the same arrays.
    """

    folds: tuple[int, ...]
    start: tuple[int, ...]
    stop: tuple[int, ...]
    n_train: tuple[int, ...]
    x: tuple[np.ndarray, ...]
    names: tuple[str, ...]

    def __iter__(self):
        return iter(zip(self.folds, self.start, self.stop, self.n_train, self.x, strict=True))


def _layout(legs: pd.DataFrame, paths: pd.Series) -> _Layout:
    if list(paths.index.names) != ["fold", "segment", "session"]:
        raise ValueError("paths must be indexed by (fold, segment, session)")
    fold = paths.index.get_level_values("fold").to_numpy()
    segment = paths.index.get_level_values("segment").to_numpy()
    session = paths.index.get_level_values("session")
    folds, starts, stops, n_train, xs = [], [], [], [], []
    for f in dict.fromkeys(fold.tolist()):
        rows = np.flatnonzero(fold == f)
        start, stop = int(rows[0]), int(rows[-1]) + 1
        if stop - start != len(rows):
            raise ValueError(f"fold {f}: its rows must be contiguous")
        seg = segment[start:stop]
        n_tr = int((seg == "train").sum())
        if not ((seg[:n_tr] == "train").all() and (seg[n_tr:] == "test").all()):
            raise ValueError(f"fold {f}: its training rows must precede its test rows")
        dates = pd.DatetimeIndex(session[start:stop])
        if not dates.is_monotonic_increasing or dates.has_duplicates:
            raise ValueError(f"fold {f}: its path must run forward in time")
        folds.append(int(f))
        starts.append(start)
        stops.append(stop)
        n_train.append(n_tr)
        xs.append(legs.loc[dates].to_numpy(float))
    return _Layout(tuple(folds), tuple(starts), tuple(stops), tuple(n_train), tuple(xs),
                   tuple(str(c) for c in legs.columns))


def _lag(values: np.ndarray) -> np.ndarray:
    """One fold's path lagged one session inside the fold (§12.2): position 0 is NaN."""
    out = np.empty(len(values), dtype=float)
    out[:1] = np.nan
    out[1:] = values[:-1]
    return out


def _states(cells: Cells) -> tuple[int, ...]:
    """The states seen anywhere on the path (``context.qualifying_cells``'s rows)."""
    return tuple(int(s) for s in np.unique(cells.table.index.get_level_values("state")))


def _fold_layout(layout: _Layout, fold: int) -> tuple[int, int, int, np.ndarray]:
    if fold not in layout.folds:
        raise KeyError(f"fold {fold} is not on the path")
    i = layout.folds.index(fold)
    return layout.start[i], layout.stop[i], layout.n_train[i], layout.x[i]


# ---------------------------------------------------------------------------------------
# the profiles and the map — §12.5, §12.7
# ---------------------------------------------------------------------------------------


def _profile(
    x: np.ndarray, held: np.ndarray, states: Sequence[int], qualifying: frozenset[int]
) -> np.ndarray:
    """§12.5 steps 1-3 (and §12.7's test profile) on one segment, as an array.

    ``x`` is the segment's legs (sessions x signals), ``held`` its lagged states (NaN
    where none is held). With ``Q`` the sessions whose lagged state qualifies:
    ``D_ik = √252 (μ_ik − μ̄_i) / σ̄_i``, ``μ_ik`` the mean of ``x_i`` over the cell of
    state *k*, ``μ̄_i`` and ``σ̄_i`` (ddof 1) over ``Q``. Rows of non-qualifying states are
    NaN. Plain numpy means (pairwise summation), no BLAS, so the result does not depend on
    a thread count. A non-finite leg, a zero or undefined ``σ̄`` gives a non-finite row.
    """
    out = np.full((len(states), x.shape[1]), np.nan)
    rows = [i for i, k in enumerate(states) if k in qualifying]
    if not rows:
        return out
    masks = [held == states[i] for i in rows]
    pooled = x[np.logical_or.reduce(masks)]
    with np.errstate(all="ignore"):
        if pooled.shape[0] < 2:
            return out
        centre = pooled.mean(axis=0)
        scale = pooled.std(axis=0, ddof=1)
        for i, mask in zip(rows, masks, strict=True):
            cell = x[mask]
            mean = cell.mean(axis=0) if cell.shape[0] else np.full(x.shape[1], np.nan)
            out[i] = SQRT_PERIODS * (mean - centre) / scale
    return out


def _tilt_map(profile: np.ndarray, qualifying_rows: np.ndarray) -> np.ndarray | None:
    """§12.5 steps 4, 5 and 7 on a training profile.

    Row-centre each qualifying state across the signals, then divide the whole table by
    the root-mean-square of the centred values over every qualifying (i, k). ``None``
    (the fold holds the control) when no state qualifies or the RMS is exactly zero, as
    the lock writes it: no tolerance is added (ten legs equal to the last bit would leave
    rounding residues, which cannot happen on ten distinct signals). A non-finite RMS is
    not zero: the map comes back non-finite and the reading is UNDECIDABLE.
    """
    if not qualifying_rows.any():
        return None
    table = profile[qualifying_rows]
    centred = table - table.mean(axis=1, keepdims=True)
    rms = float(np.sqrt(np.mean(centred**2)))
    if rms == 0.0:
        return None
    m = np.full_like(profile, np.nan)
    with np.errstate(all="ignore"):
        m[qualifying_rows] = centred / rms
    return m


def _as_frame(values: np.ndarray, states: Sequence[int], names: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(values, index=pd.Index(list(states), name="state"), columns=list(names))


def _segment_profile(
    legs: pd.DataFrame, paths: pd.Series, fold: int, cells: Cells, segment: str
) -> pd.DataFrame:
    layout = _layout(legs, paths)
    start, stop, n_tr, x = _fold_layout(layout, fold)
    held = _lag(paths.to_numpy(float)[start:stop])
    states = _states(cells)
    if segment == "train":
        values = _profile(x[:n_tr], held[:n_tr], states, cells.train.get(fold, frozenset()))
    else:
        values = _profile(x[n_tr:], held[n_tr:], states, cells.test.get(fold, frozenset()))
    return _as_frame(values, states, layout.names)


def training_profile(
    legs: pd.DataFrame, paths: pd.Series, fold: int, cells: Cells
) -> pd.DataFrame:
    """``D_ik`` of §12.5 steps 1-3 for fold ``fold``: states x signals.

    On fold *f*'s training segment, with the lagged states of its own path (§12.2; the
    first training session holds none and is skipped). ``Q_f`` is the union of the
    **qualifying** training cells (``cells.train[f]``, decided on the stamped path,
    §12.4); ``D_ik = √252 (μ_ik − μ̄_i) / σ̄_i``, with ``μ̄_i``, ``σ̄_i`` (ddof 1) over
    ``Q_f``. NaN rows for non-qualifying states. ``Σ_k n_k D_ik = 0`` on the training
    window. This is also A-2's training profile (§12.7).
    """
    return _segment_profile(legs, paths, fold, cells, "train")


def test_profile(legs: pd.DataFrame, paths: pd.Series, fold: int, cells: Cells) -> pd.DataFrame:
    """A-2's test profile ``D^te_ik`` (§12.7): the same computation on fold *f*'s test
    sessions, lagged states along fold *f*'s path (first test session included, holding
    the label stamped on the last training session), over ``Q^te_f``, the union of the
    qualifying **test** cells (``cells.test[f]``). A fold with one qualifying test cell
    gives that cell a zero profile."""
    return _segment_profile(legs, paths, fold, cells, "test")


test_profile.__test__ = False  # a lock quantity, not a pytest test


def estimate_m(
    legs: pd.DataFrame, paths: pd.Series, fold: int, cells: Cells
) -> pd.DataFrame | None:
    """The level-A map ``m`` of fold ``fold`` (§12.5 steps 1-5, 7): states x signals.

    ``D`` of :func:`training_profile`; step 4 row-centres each qualifying state across the
    ten signals (``D̃_ik = D_ik − mean_j D_jk``); step 5 divides by the root-mean-square
    of ``D̃`` over every qualifying (i, k), so the table has unit RMS. Rows of
    non-qualifying states are NaN (they abstain at equal weight, step 6). Returns
    ``None`` when the fold holds the control (step 7: no qualifying state, or an RMS of
    exactly zero). ``cells`` is ``tree.cells(paths)``.
    """
    d = training_profile(legs, paths, fold, cells)
    qualifying = np.array([k in cells.train.get(fold, frozenset()) for k in d.index])
    m = _tilt_map(d.to_numpy(float), qualifying)
    return None if m is None else _as_frame(m, d.index, d.columns)


# ---------------------------------------------------------------------------------------
# the selector — §4, §12.3, §12.5 step 6, §12.6 "Arms"
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False, repr=False)
class Selector:
    """The level-A selector on one partition: per-fold maps, tilt tables, traded path.

    ``maps[f]`` is ``m`` (§12.5) or ``None`` when fold *f* holds the control; ``tables[f]``
    the tilt ``protocol.tilt_table(m, d)`` (§4, §12.5 step 6) or ``None``; ``traded``
    the traded path of §12.3 (``tree.traded_mix``) with its fallback flags (§12.6).
    ``finite`` is False when a qualifying row of some ``m`` is not finite, which makes
    every lock built on it UNDECIDABLE (§13.4). The ``repr`` shows counts only.
    """

    maps: dict[int, pd.DataFrame | None]
    tables: dict[int, pd.DataFrame | None]
    traded: TradedMix
    finite: bool

    @property
    def control_folds(self) -> int:
        return sum(m is None for m in self.maps.values())

    @property
    def fallback_sessions(self) -> int:
        return int(self.traded.fallback.sum())

    def __repr__(self) -> str:
        return (f"Selector({len(self.maps)} folds, {self.control_folds} at the control, "
                f"{self.fallback_sessions} fallback test sessions, finite={self.finite})")


def _selector(
    data: TreeData,
    paths: pd.Series,
    layout: _Layout,
    train_cells: Mapping[int, frozenset[int]],
    states: Sequence[int],
    d: float,
) -> Selector:
    if tuple(f.number for f in data.folds) != layout.folds:
        raise ValueError("the path's folds are not the tree's")
    labels = paths.to_numpy(float)
    maps: dict[int, pd.DataFrame | None] = {}
    tables: dict[int, pd.DataFrame | None] = {}
    finite = True
    for fold, start, stop, n_tr, x in layout:
        held = _lag(labels[start:stop])[:n_tr]
        qualifying = train_cells.get(fold, frozenset())
        m = _tilt_map(_profile(x[:n_tr], held, states, qualifying),
                      np.array([k in qualifying for k in states], dtype=bool))
        if m is None:
            maps[fold] = tables[fold] = None
            continue
        rows = np.array([k in qualifying for k in states], dtype=bool)
        finite = finite and bool(np.isfinite(m[rows]).all())
        frame = _as_frame(m, states, layout.names)
        maps[fold] = frame
        tables[fold] = protocol.tilt_table(frame, d=d)
    traded = tree.traded_mix(data.sessions, data.folds, tables, paths,
                             train_cells=train_cells, names=layout.names)
    return Selector(maps, tables, traded, finite)


def selector_mix(
    data: TreeData, paths: pd.Series, variant: Variant = PRIMARY, cells: Cells | None = None
) -> Selector:
    """The level-A selector on a stamped path (§12.6 "Arms").

    For each fold, ``m`` of §12.5 on its training segment, the tilt
    ``protocol.tilt_table(m, d=variant.d)`` (§4: ``(1/n)(1 + d·m_ik)``, floored at zero,
    renormalised), and the traded path of §12.3 through ``tree.traded_mix``: equal weight
    before the first test session, fold *f*'s table via ``map_states(..., lag=1)`` on
    fold *f*'s own path, restricted to its qualifying training states. A **fallback
    session** (lagged state missing, non-qualifying, or the fold at the control) holds
    the equal-weight row and is flagged. ``paths`` may be the partition's, a placebo
    draw's (``PlaceboDraws.draw``) or a witness's; ``cells`` defaults to
    ``tree.cells(paths)``.
    """
    c = tree.cells(paths) if cells is None else cells
    return _selector(data, paths, _layout(data.legs, paths), c.train, _states(c), variant.d)


# ---------------------------------------------------------------------------------------
# A-2's statistic — §12.7
# ---------------------------------------------------------------------------------------


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """§12.7: ``scipy.stats.spearmanr`` (average ranks for ties); 0 if either side has
    zero rank variance (all values equal); NaN if either side is not finite."""
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return NAN
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return 0.0
    return float(stats.spearmanr(a, b).statistic)


def _transfer(
    layout: _Layout, labels: np.ndarray, cells: Cells, states: Sequence[int]
) -> tuple[float, dict[tuple[int, int], float]]:
    """R of §12.7 on one labelling of the path (real, placebo draw or witness)."""
    rho: dict[tuple[int, int], float] = {}
    for fold, start, stop, n_tr, x in layout:
        pairs = [k for f, k in cells.pairs if f == fold]
        if not pairs:
            continue
        held = _lag(labels[start:stop])
        train = _profile(x[:n_tr], held[:n_tr], states, cells.train.get(fold, frozenset()))
        test = _profile(x[n_tr:], held[n_tr:], states, cells.test.get(fold, frozenset()))
        for k in pairs:
            i = states.index(k)
            rho[(fold, k)] = _spearman(train[i], test[i])
    values = np.array([rho[p] for p in cells.pairs if p in rho], dtype=float)
    return (float(values.mean()) if values.size else NAN), rho


@dataclass(frozen=True)
class RankTransfer:
    """A-2's statistic (§12.7): ``R`` = plain mean of ``rho[(f, k)]`` over the qualifying
    (fold, state) pairs. Per-fold means are diagnostics only (§12.14)."""

    R: float
    rho: dict[tuple[int, int], float]

    @property
    def per_fold(self) -> dict[int, float]:
        folds = sorted({f for f, _ in self.rho})
        return {f: float(np.mean([v for (g, _), v in self.rho.items() if g == f]))
                for f in folds}


def rank_transfer(legs: pd.DataFrame, paths: pd.Series, cells: Cells | None = None) -> RankTransfer:
    """A-2's rank-transfer statistic ``R`` on a stamped path (§12.7) — a READING quantity.

    For each (fold, state) pair qualifying on both segments (``cells.pairs``), the
    Spearman correlation across the ten signals of the training profile ``D_·k``
    (§12.5 step 3) and the test profile ``D^te_·k``; ``R`` is their plain mean. The
    instrument never calls this on the real partition (§13.1 step 2).
    """
    c = tree.cells(paths) if cells is None else cells
    R, rho = _transfer(_layout(legs, paths), paths.to_numpy(float), c, _states(c))
    return RankTransfer(R, rho)


# ---------------------------------------------------------------------------------------
# placebo jobs, for tree.draw_map — §12.6 step 4, §12.7, §12.11
# ---------------------------------------------------------------------------------------


def _null_job(shared: tuple[_Layout, PlaceboDraws, Cells, tuple[int, ...]], j: int) -> float:
    """R^(j) (§12.7): both profiles recomputed with draw j's labels, training and test.
    Every draw qualifies exactly the real cells (§12.4, ``tree.placebo_draws``)."""
    layout, draws, cells, states = shared
    return _transfer(layout, draws.draw(j).to_numpy(float), cells, states)[0]


def _arm_job(shared: tuple[Any, ...], j: int) -> tuple[float, float]:
    """(Δ^(j), β^(j)) of §12.6 step 4: ``m`` re-estimated on draw j's training labels,
    the selector rebuilt at the same d on draw j's path, its 5 bp book paired against the
    same control. A non-finite map gives (NaN, NaN): the lock is then UNDECIDABLE, and no
    draw is dropped (§13.4)."""
    data, layout, draws, train_cells, states, d, control_sharpe = shared
    selector = _selector(data, draws.draw(j), layout, train_cells, states, d)
    if not selector.finite:
        return NAN, NAN
    book = tree.build_book(data, selector.traded.mix, costs=(DECIDING_BPS,))
    return tree.sharpe(book.net5) - control_sharpe, book.beta()


# ---------------------------------------------------------------------------------------
# the NFCI diagnostic — §12.6 step 6, §3
# ---------------------------------------------------------------------------------------


def nfci_diagnostic(
    data: TreeData, declared: pd.Series, pit: Partition, K: int
) -> dict[str, Any]:
    """§12.6 step 6: the 18-feature partition against the declared one, labels only.

    - **Agreement:** per fold, ``context.align_labels(declared test labels, PIT test
      labels, K)`` on the fold's test sessions (stamped, industry sessions), then the
      share of all test sessions (5,031 on the real store) on which the relabelled PIT
      state equals the declared one; per-fold shares are given too.
    - **OOS transitions per year:** transitions inside each fold's test segment of the
      PIT paths, summed, over the summed test years — the convention of the declared
      clock (12.55/yr, ``scripts/measure_twosigma_context.py``).
    - **Pooled η²:** ``context.eta_squared(wf.aligned, log rv)`` on the feature calendar,
      the convention of the declared 0.142 (§12.2: the aligned series serves pooled
      descriptive statistics only).

    Counts only, no return.
    """
    agree_total, n_total, n_transitions = 0, 0, 0
    per_fold: dict[str, float] = {}
    for fold in data.folds:
        a = declared.xs((fold.number, "test"), level=("fold", "segment"))
        b = pit.paths.xs((fold.number, "test"), level=("fold", "segment"))
        if a.isna().any() or b.isna().any() or not a.index.equals(b.index):
            raise ValueError("the NFCI diagnostic needs fully labelled, identical test sessions")
        mapping = align_labels(a.astype(np.int64), b.astype(np.int64), K)
        agree = int((mapping[b.to_numpy(np.int64)] == a.to_numpy(np.int64)).sum())
        per_fold[str(fold.number)] = agree / len(a)
        agree_total += agree
        n_total += len(a)
        n_transitions += transitions(b.to_numpy())
    years = sum(f.test_years for f in data.folds)
    panel = tree.variant_panel(data, PIT18)
    return {
        "features": len(PIT18.features),
        "test_sessions": n_total,
        "agree_sessions": agree_total,
        "agreement": agree_total / n_total,
        "agreement_per_fold": per_fold,
        "transitions_per_year": n_transitions / years,
        "eta2_log_rv": float(eta_squared(pit.wf.aligned, panel[LOG_RV])),
    }


# ---------------------------------------------------------------------------------------
# the instrument — §12.6 steps 1-6, §12.7 "Instrument", §13.1 steps 1-3
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False, repr=False)
class _Run:
    """Everything the instrument built; the reading continues from it. Never printed."""

    variant: Variant
    partition: Partition
    cells: Cells
    states: tuple[int, ...]
    layout: _Layout
    draws: PlaceboDraws
    selector: Selector
    book: BookResult
    control: BookResult
    pair: PairThreshold
    kill: Kill
    null: np.ndarray | None
    pit_partition: Partition | None
    boot_draws: int
    printout: dict[str, Any]

    def __repr__(self) -> str:
        return f"_Run({self.variant.name}, {self.draws!r})"


def _check_partition(partition: Partition, variant: Variant) -> None:
    for model in partition.wf.models:
        if model.kmeans.n_clusters != variant.K or tuple(model.columns) != variant.features:
            raise ValueError(f"the partition passed is not {variant.name}'s")


def _run(
    data: TreeData,
    variant: Variant,
    draws: int | PlaceboDraws,
    *,
    boot_draws: int,
    partition: Partition | None,
    workers: int | None,
    a2: bool | None = None,
    nfci: bool | None = None,
) -> _Run:
    with_a2 = ("A" in variant.levels) if a2 is None else a2
    with_nfci = (variant.role == "primary") if nfci is None else nfci
    part = tree.partition_paths(data, variant) if partition is None else partition
    _check_partition(part, variant)
    paths = part.paths
    cells = tree.cells(paths)
    states = _states(cells)
    layout = _layout(data.legs, paths)
    if isinstance(draws, PlaceboDraws):
        placebo = tree.check_draws(draws, paths)
    else:
        placebo = tree.placebo_draws(paths, n=int(draws), seed=PLACEBO_SEED)

    selector = _selector(data, paths, layout, cells.train, states, variant.d)
    book = tree.build_book(data, selector.traded.mix)
    control = tree.control_book(data)
    pool = os.cpu_count() if workers is None else workers
    pair = tree.pair_threshold(book.net5, control.net5, draws=boot_draws, seed=BOOT_SEED,
                               workers=pool)
    killed = tree.kill(book.turnover - control.turnover, pair.threshold, book.sigma)
    null = None
    if with_a2:
        null = np.asarray(tree.draw_map(_null_job, placebo.n, (layout, placebo, cells, states),
                                        workers=workers), dtype=float)
    pit = tree.partition_paths(data, pit_variant(variant)) if with_nfci else None

    undecidable = _pre_reading(variant, cells, placebo, selector, book, control, pair,
                               killed, null)
    years = sum(f.test_years for f in data.folds)
    printout: dict[str, Any] = {
        "lock": tree.LOCK_COMMIT,
        "variant": variant.name,
        "K": variant.K,
        "d": variant.d,
        "smoothing": int(variant.smoothing or 0),
        "features": len(variant.features),
        "placebo": {"method": PLACEBO_METHOD, "draws": placebo.n, "seed": placebo.seed,
                    "exact": placebo.exact},
        "bootstrap": {"draws": boot_draws, "seed": BOOT_SEED, "blocks": list(POWER_BLOCKS)},
        "counts": {
            "test_sessions": len(data.test_sessions),
            "cells_possible": len(data.folds) * variant.K,
            "train_cells": cells.n_train,
            "abstaining_rows": len(data.folds) * variant.K - cells.n_train,
            "test_cells": cells.n_test,
            "a2_cells": cells.n_pairs,
            "a2_cells_per_fold": cells.per_fold,
            "cell_floor": variant.cell_floor,
            "control_folds": selector.control_folds,
            "fallback_sessions": selector.fallback_sessions,
            "fallback_share": selector.traded.fallback_share,
            "map_finite": selector.finite,
        },
        "A-1": {
            **{("T_A1" if k == "threshold" else k): v for k, v in pair.as_dict().items()},
            "standalone_threshold": protocol.standalone_sharpe_threshold(years, alpha=BONFERRONI),
            "turnover": {"selector": book.turnover, "control": control.turnover,
                         "delta": killed.delta_turnover},
            "kill": {"k_kill": killed.k_kill, "fires": killed.fires},
            "sigma": {"selector": book.sigma, "control": control.sigma},
            "cap_share": {"selector": book.cap_share, "control": control.cap_share},
            "missing_sessions": {"selector": book.missing, "control": control.missing},
        },
    }
    if null is not None:
        summary = tree.null_summary(null)
        printout["A-2"] = {"cells": cells.n_pairs, "q50": summary.q50, "q95": summary.q95,
                           "q99": summary.q99, "q99-q50": summary.resolution}
    if pit is not None:
        printout["nfci"] = nfci_diagnostic(data, paths, pit, variant.K)
    printout["undecidable"] = undecidable
    return _Run(variant, part, cells, states, layout, placebo, selector, book, control, pair,
                killed, null, pit, boot_draws, tree.printable(printout))


def _pre_reading(
    variant: Variant,
    cells: Cells,
    placebo: PlaceboDraws,
    selector: Selector,
    book: BookResult,
    control: BookResult,
    pair: PairThreshold,
    killed: Kill,
    null: np.ndarray | None,
) -> dict[str, list[str]]:
    """Why A-1 or A-2 is UNDECIDABLE before its reading, from instrument quantities only
    (§13.4: such a lock is not read and spends no trial row).

    Both locks: a placebo check that is not exact (§12.11); fewer A-2 cells than the floor
    (§13.4, a level condition, applied to A-1 as well, since only A-1 can read FAIL at
    level A). A-1 (§12.6 line 1): a map ``m`` that is not finite, a paired leg missing on
    a test session, the fallback on more than half the test sessions, a non-finite
    instrument value (T_A1, SE*, σ, turnover) or an undetermined kill. A-2 (§12.7): any
    non-finite ``R^(j)``. Counts and flags only; nothing here reads a return statistic.
    """
    common = []
    if not placebo.exact:
        common.append("the placebo check is not exact (§12.11)")
    if cells.n_pairs < variant.cell_floor:
        common.append(f"{cells.n_pairs} A-2 cells, below the floor {variant.cell_floor:g} "
                      "(§13.4)")
    a1 = list(common)
    if not selector.finite:
        a1.append("the map m is not finite (§13.4)")
    if book.missing or control.missing:
        a1.append("a paired leg is missing on a test session (§13.4)")
    share = selector.traded.fallback_share
    if not (np.isfinite(share) and share <= tree.MAX_FALLBACK_SHARE):
        a1.append(f"the selector is at the fallback on {share:.1%} of the test sessions "
                  "(§12.6 line 1)")
    values = {"T_A1": pair.threshold, "SE*": pair.se_star, "sigma": book.sigma,
              "the control's sigma": control.sigma, "turnover": book.turnover,
              "the control's turnover": control.turnover}
    bad = [name for name, v in values.items() if not np.isfinite(v)]
    if bad:
        a1.append(f"non-finite instrument reading: {', '.join(bad)} (§13.4)")
    if killed.fires is None:
        a1.append("the kill line is undetermined (§13.4)")
    out = {"A-1": a1}
    if null is not None:
        a2 = list(common)
        broken = int((~np.isfinite(null)).sum())
        if broken:
            a2.append(f"{broken} of {len(null)} placebo draws of R are not finite (§13.4)")
        out["A-2"] = a2
    return out


def instrument(
    data: TreeData,
    variant: Variant = PRIMARY,
    draws: int | PlaceboDraws = N_DRAWS,
    *,
    boot_draws: int = BOOT_DRAWS,
    partition: Partition | None = None,
    workers: int | None = None,
) -> dict[str, Any]:
    """The level-A instrument of one variant (§12.6 steps 1-6, §12.7, §13.1 steps 1-2).

    Builds the partition (``tree.partition_paths``; or ``partition``, which must be the
    variant's), its cells (§12.4), the uniform placebo (§12.11: ``draws`` draws, seed 0,
    ``method="uniform"``), ``m`` per fold (§12.5), the selector and the control on the
    traded path (§12.3), and returns, as JSON types, only what §13.1 step 2 allows:

    - ``counts``: cells (training, test, A-2 pairs with the floor), abstaining (fold,
      state) rows, folds at the control, fallback sessions and share (§12.6 step 5);
    - ``A-1``: the blinded bootstrap SE and MDEs at α 0.05 and 0.05/6 for blocks 21 / 63 /
      126 on the demeaned 5 bp legs (step 1), ``T_A1 = max(0.338, max_b MDE(0.05/6, b))``
      with SE* and its block (step 2), Lo's standalone threshold beside it (§12.12), the
      held turnover of both arms and ΔT, ``K_kill`` and whether the kill fires (step 3),
      realised sd, cap share and missing sessions of both arms (step 5);
    - ``A-2`` (variants that evaluate level A, not the d rows): the qualifying cells and
      the null's q50, q95, q99 and q99 − q50 (§12.7);
    - ``nfci`` (the primary only): :func:`nfci_diagnostic` (§12.6 step 6);
    - ``undecidable``: for A-1 (and A-2), why the lock is UNDECIDABLE before its reading,
      if it is (§13.4); such a lock is not read and spends no trial row.

    A non-finite or undetermined value is returned as ``tree.NON_FINITE``
    (``'non-finite'``), never as a number. The result is what the threshold file's
    section (:func:`section_name`) records, and what :func:`read` recomputes and
    verifies bitwise. ``boot_draws`` and an integer ``draws`` other than 2,000 and
    1,000 exist for synthetic tests; the file records both counts.
    """
    with threadpool_limits(limits=1):
        return _run(data, variant, draws, boot_draws=boot_draws, partition=partition,
                    workers=workers).printout


# ---------------------------------------------------------------------------------------
# verdicts — §12.6 lines 1-10, §12.7
# ---------------------------------------------------------------------------------------


def a1_lines(
    *,
    delta5: float,
    delta10: float,
    threshold: float,
    p: float,
    kill_fires: bool | None,
    beta_pct: float,
    diff_pct: float,
    delta_w: float,
    fallback_share: float,
    leg_missing: bool,
    undecidable: bool = False,
    holm_rejected: bool | None = None,
) -> str:
    """A-1's verdict lines 1-8 of §12.6, the first that applies, else PASS (before line 9).

    ``tree.paired_verdict``: 1 UNDECIDABLE on any non-finite input (a percentile is NaN
    when any placebo Δ^(j) or β^(j) is), an unknown kill, a missing paired leg, a
    fallback share above 0.5, or ``undecidable``, which carries every other reason of
    line 1 and §13.4 (a non-finite map or witness, a non-exact placebo, A-2 cells below
    the floor, a non-finite 20 bp column); 2 FAIL (cost), the kill fired; 3 FAIL, Δ ≤ 0
    at 5 bp; 4 UNDECIDED, Δ ≤ 0 at 10 bp; 5 UNDERPOWERED, Δ < T_A1 or Holm not
    rejecting; 6 DOWNGRADED (beta), beta percentile ≥ 0.95; 7 DOWNGRADED (not
    conditional), difference percentile < 0.95; 8 DOMINATED (volatility), Δ_W ≥ Δ.
    ``holm_rejected`` defaults to the provisional ``p ≤ 0.05/6`` (§13.3).
    """
    return tree.paired_verdict(
        delta5=delta5, delta10=delta10, threshold=threshold, p=p, kill_fires=kill_fires,
        beta_pct=beta_pct, diff_pct=diff_pct, delta_w=delta_w, fallback_share=fallback_share,
        leg_missing=leg_missing, holm_rejected=holm_rejected, other_finite=not undecidable,
    )


def a1_verdict(*, rebuild: str | None = None, **lines: Any) -> str:
    """A-1's verdict, §12.6 lines 1-10: :func:`a1_lines`, then line 9 for a would-be PASS
    only — ``rebuild`` is lines 1-8 applied to the 18-feature rebuild (§8 control 5),
    required then; anything but PASS reads DOWNGRADED (PIT). Line 10 is PASS."""
    verdict = a1_lines(**lines)
    return tree.apply_pit(verdict, rebuild) if verdict == PASS else verdict


def a2_lines(
    *,
    R: float,
    p: float,
    exact: bool,
    R_W: float,
    cells: int,
    floor: float,
    undecidable: bool = False,
    holm_rejected: bool | None = None,
) -> str:
    """A-2's verdict of §12.7 before its PIT clause.

    UNDECIDABLE: ``R`` or ``p`` not finite (``p`` is NaN when any ``R^(j)`` is), the
    placebo check not exact, fewer than ``floor`` qualifying cells (half of the 5K cells),
    or ``undecidable``; also when ``R_W`` is not finite, since neither PASS nor DOMINATED
    can then be read (§13.4: a non-finite reading value). The lock holds if ``p ≤ 0.01``
    and Holm rejects (default: the provisional ``p ≤ 0.05/6``); it holds and ``R_W ≥ R``:
    DOMINATED; it holds and ``R_W < R``: PASS (then :func:`a2_verdict`'s PIT clause).
    Otherwise NOT SHOWN: no power was measured for A-2, which never reads FAIL.
    """
    finite = bool(np.isfinite([R, R_W]).all())
    return tree.placebo_verdict(
        statistic=R, p=p, exact=exact, dominated=bool(R_W >= R) if finite else None,
        holm_rejected=holm_rejected, gate_failed=False, powered=False,
        other_undecidable=undecidable or cells < floor,
    )


def a2_verdict(*, rebuild: str | None = None, **lines: Any) -> str:
    """A-2's verdict, §12.7: :func:`a2_lines`; a would-be PASS stays PASS only if the lock
    rebuilt on the 18-feature partition holds with its own witness (``rebuild``, the same
    lines applied to it, PASS), else DOWNGRADED (PIT) (§8 control 5)."""
    verdict = a2_lines(**lines)
    return tree.apply_pit(verdict, rebuild) if verdict == PASS else verdict


# ---------------------------------------------------------------------------------------
# the reading — §12.6 "Reading", §12.7, §8 controls 3-5, §12.13, §13.1 step 4
# ---------------------------------------------------------------------------------------


def _cost_key(bps: float) -> str:
    return f"{bps:g}"


@dataclass(frozen=True)
class _Witness:
    """W1 through the same code as the partition (§8 control 4, §12.6, §12.7)."""

    delta_w: float
    R_W: float
    train_cells: int
    pairs: int
    finite: bool


def _witness(data: TreeData, variant: Variant, control: BookResult, with_a2: bool) -> _Witness:
    """Witness W1 of §12.8 at the variant's K (quantile bins of log rv, cut-offs on each
    fold's training sessions, stamped, lagged one session): qualification on its stamped
    path, the §12.5 estimator per fold, the same selector at the same d and the same traded
    path give ``Δ_W`` against the same control; the same A-2 code gives ``R_W``.
    *Choice:* at the smoothing sensitivity the witness is not smoothed — §12.13 runs the
    witnesses "as for the primary", and W1 is a volatility partition, not a context one."""
    paths = tree.witness_paths(data, "W1", variant.K)
    cells = tree.cells(paths)
    states = _states(cells)
    layout = _layout(data.legs, paths)
    selector = _selector(data, paths, layout, cells.train, states, variant.d)
    book = tree.build_book(data, selector.traded.mix, costs=(DECIDING_BPS,))
    delta_w = (tree.sharpe(book.net5) - tree.sharpe(control.net5)) if selector.finite else NAN
    r_w = _transfer(layout, paths.to_numpy(float), cells, states)[0] if with_a2 else NAN
    return _Witness(delta_w, r_w, cells.n_train, cells.n_pairs, selector.finite)


def _a1_reading(
    data: TreeData, run: _Run, witness: _Witness, *, workers: int | None
) -> dict[str, Any]:
    """A-1's reading on one instrument run (§12.6 "Reading"): the statistics, the verdict
    inputs, and lines 1-8 at the provisional Bonferroni bar (no PIT line)."""
    book, control, variant = run.book, run.control, run.variant
    sharpe = {"selector": {_cost_key(b): tree.sharpe(book.net(b)) for b in COST_COLUMNS},
              "control": {_cost_key(b): tree.sharpe(control.net(b)) for b in COST_COLUMNS}}
    delta = {k: sharpe["selector"][k] - sharpe["control"][k] for k in sharpe["selector"]}
    reading = tree.pair_reading(delta[_cost_key(DECIDING_BPS)], run.pair.se_star, book.net5,
                                control.net5)
    arms = tree.draw_map(_arm_job, run.draws.n,
                         (data, run.layout, run.draws, run.cells.train, run.states, variant.d,
                          sharpe["control"][_cost_key(DECIDING_BPS)]),
                         workers=workers)
    null_delta = np.array([a for a, _ in arms], dtype=float)
    null_beta = np.array([b for _, b in arms], dtype=float)
    beta = book.beta()
    beta_pct = protocol.placebo_percentile(beta, null_beta)
    diff_pct = protocol.placebo_percentile(delta[_cost_key(DECIDING_BPS)], null_delta)
    per_fold = {}
    for fold in data.folds:
        test = fold.test(data.sessions)
        per_fold[str(fold.number)] = (tree.sharpe(book.net5.loc[test])
                                      - tree.sharpe(control.net5.loc[test]))

    leg_missing = bool(book.missing or control.missing)
    share = run.selector.traded.fallback_share
    # line 1's reasons that tree.paired_verdict does not see among its own inputs
    other = {
        "the map m is not finite": not run.selector.finite,
        "the placebo check is not exact": not run.draws.exact,
        "the witness map is not finite": not witness.finite,
        f"{run.cells.n_pairs} A-2 cells, below the floor {variant.cell_floor:g} (§13.4)":
            run.cells.n_pairs < variant.cell_floor,
        "the 20 bp difference is not finite": not np.isfinite(delta[_cost_key(COST_COLUMNS[-1])]),
    }
    # line 1's reasons carried by paired_verdict's inputs, named for the note (§13.4)
    seen = {
        "a placebo arm's delta or beta is not finite":
            not (np.isfinite(null_delta).all() and np.isfinite(null_beta).all()),
        "a reading value (delta, p, beta, T_A1, delta_w) is not finite": not np.isfinite(
            [delta[_cost_key(DECIDING_BPS)], delta[_cost_key(COST_COLUMNS[1])], reading.p,
             beta, run.pair.threshold, witness.delta_w]).all(),
        "the kill is undetermined": run.kill.fires is None,
        "a paired leg is missing on a test session": leg_missing,
        f"fallback on {share:.1%} of the test sessions":
            not (np.isfinite(share) and share <= tree.MAX_FALLBACK_SHARE),
    }
    inputs = {
        "delta5": delta[_cost_key(DECIDING_BPS)], "delta10": delta[_cost_key(COST_COLUMNS[1])],
        "threshold": run.pair.threshold, "p": reading.p, "kill_fires": run.kill.fires,
        "beta_pct": beta_pct, "diff_pct": diff_pct, "delta_w": witness.delta_w,
        "fallback_share": share, "leg_missing": leg_missing,
        "undecidable": any(other.values()),
    }
    verdict = a1_lines(**inputs)
    reasons = [r for r, hit in (*other.items(), *seen.items()) if hit]
    return {
        "read": True,
        "sharpe": sharpe,
        "delta": delta,
        "T_A1": run.pair.threshold,
        "se_star": run.pair.se_star,
        "block_star": run.pair.block_star,
        "t_hac": reading.t_hac,
        "p_boot": reading.p_boot,
        "p_hac": reading.p_hac,
        "p_raw": reading.p_raw,
        "p": reading.p,
        "beta": {"selector": beta, "control": control.beta()},
        "beta_pct": beta_pct,
        "diff_pct": diff_pct,
        "placebo_arms": run.draws.n,
        "delta_w": witness.delta_w,
        "per_fold": per_fold,
        "turnover_delta": run.kill.delta_turnover,
        "k_kill": run.kill.k_kill,
        "kill_fires": run.kill.fires,
        "fallback_share": share,
        "leg_missing": leg_missing,
        "undecidable_reasons": reasons if verdict == UNDECIDABLE else [],
        "verdict_inputs": inputs,
        "verdict_lines": verdict,
    }


def _a2_reading(run: _Run, witness: _Witness) -> dict[str, Any]:
    """A-2's reading on one instrument run (§12.7): R, p, the witness, the verdict inputs
    and the lines at the provisional Bonferroni bar, before the PIT line."""
    if run.null is None:
        raise ValueError("this run did not compute A-2's null")
    R, rho = _transfer(run.layout, run.partition.paths.to_numpy(float), run.cells, run.states)
    transfer = RankTransfer(R, rho)
    p = protocol.placebo_p_value(R, run.null)
    reasons = []
    if not np.isfinite(R):
        reasons.append("R is not finite")
    if not np.isfinite(run.null).all():
        reasons.append("a placebo R^(j) is not finite")
    if not run.draws.exact:
        reasons.append("the placebo check is not exact")
    if run.cells.n_pairs < run.variant.cell_floor:
        reasons.append(f"{run.cells.n_pairs} A-2 cells, below the floor "
                       f"{run.variant.cell_floor:g}")
    if not np.isfinite(witness.R_W):
        reasons.append("R_W is not finite")
    inputs = {"R": R, "p": p, "exact": run.draws.exact, "R_W": witness.R_W,
              "cells": run.cells.n_pairs, "floor": run.variant.cell_floor, "undecidable": False}
    lines = a2_lines(**inputs)
    summary = tree.null_summary(run.null)
    return {
        "read": True,
        "R": R,
        "p": p,
        "placebo_pct": protocol.placebo_percentile(R, run.null),
        "q50": summary.q50,
        "q95": summary.q95,
        "q99": summary.q99,
        "R_W": witness.R_W,
        "cells": run.cells.n_pairs,
        "per_fold": {str(f): v for f, v in transfer.per_fold.items()},
        "undecidable_reasons": reasons if lines == UNDECIDABLE else [],
        "verdict_inputs": inputs,
        "verdict_lines": lines,
    }


def _unread(reasons: Sequence[str]) -> dict[str, Any]:
    """A lock UNDECIDABLE before its reading (§13.4): no statistic of it is computed, and
    it spends no trial row."""
    return {"read": False, "p": NAN, "undecidable_reasons": list(reasons),
            "verdict_inputs": None, "verdict_lines": UNDECIDABLE}


_LINES = {"A-1": a1_lines, "A-2": a2_lines}


def _could_pass(name: str, lock: Mapping[str, Any], primary: bool) -> bool:
    """Would the lock's lines read PASS at the provisional bar, or — for a primary, whose
    final Holm step comes after C — under a Holm rejection? (§8 control 5 is then due.)"""
    if not lock["read"]:
        return False
    outcomes = (None, True) if primary else (None,)
    return any(_LINES[name](**lock["verdict_inputs"], holm_rejected=h) == PASS
               for h in outcomes)


def _same(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    encode = tree.encode_thresholds
    return json.dumps(encode(a), sort_keys=True) == json.dumps(encode(b), sort_keys=True)


def _read(data: TreeData, run: _Run, *, workers: int | None) -> dict[str, Any]:
    """The level-A reading on a verified instrument run (§12.6 "Reading", §12.7).

    A lock the instrument found UNDECIDABLE (``printout["undecidable"]``) is not read: none
    of its statistics is computed and it spends no trial row (§13.4)."""
    variant = run.variant
    with_a2 = run.null is not None
    primary = variant.role == "primary"
    pre = run.printout["undecidable"]
    read_a1 = not pre["A-1"]
    read_a2 = with_a2 and not pre["A-2"]
    witness = _witness(data, variant, run.control, with_a2) if (read_a1 or read_a2) else None
    locks = {"A-1": (_a1_reading(data, run, witness, workers=workers) if read_a1
                     else _unread(pre["A-1"]))}
    if with_a2:
        locks["A-2"] = _a2_reading(run, witness) if read_a2 else _unread(pre["A-2"])

    # §8 control 5: only a lock that could PASS is rebuilt, on its own draws and
    # thresholds, with the same witness (W1 does not depend on the features); the
    # rebuild's own p is compared with 0.05/6 and never enters Holm
    would_pass = [name for name, lock in locks.items() if _could_pass(name, lock, primary)]
    rebuilt: dict[str, Any] = {}
    pit_printout = None
    if would_pass:
        rebuild = _run(data, pit_variant(variant), run.draws.n, boot_draws=run.boot_draws,
                       partition=run.pit_partition, workers=workers,
                       a2="A-2" in would_pass, nfci=False)
        pit_printout = rebuild.printout
        pre_pit = rebuild.printout["undecidable"]
        if "A-1" in would_pass:
            rebuilt["A-1"] = (_a1_reading(data, rebuild, witness, workers=workers)
                              if not pre_pit["A-1"] else _unread(pre_pit["A-1"]))
        if "A-2" in would_pass:
            rebuilt["A-2"] = (_a2_reading(rebuild, witness) if not pre_pit["A-2"]
                              else _unread(pre_pit["A-2"]))
    for name, lock in locks.items():
        lock["pit"] = rebuilt.get(name)

    out: dict[str, Any] = {
        "level": SECTION,
        "section": section_name(variant),
        "variant": variant.name,
        "role": variant.role,
        "sessions": len(data.test_sessions),
        "test_sessions": len(data.test_sessions),
        "lock_constants": run.draws.n == N_DRAWS and run.boot_draws == BOOT_DRAWS,
        "instrument": run.printout,
        **locks,
        "pit_instrument": pit_printout,
    }
    verdicts = lock_verdicts(out)
    for name, lock in locks.items():
        lock["verdict"] = verdicts[name]
    out["verdicts"] = verdicts
    out["level_verdict"] = verdicts.get(SECTION)
    out["holm_p"] = {name: lock["p"] for name, lock in locks.items()}
    out["trial_rows"] = trial_rows(out, variant)
    return out


def read(
    data: TreeData,
    variant: Variant = PRIMARY,
    draws: int | PlaceboDraws = N_DRAWS,
    thresholds: Mapping[str, Any] | None = None,
    *,
    boot_draws: int = BOOT_DRAWS,
    partition: Partition | None = None,
    workers: int | None = None,
    path: Path = THRESHOLDS_PATH,
    root: Path = ROOT,
    inputs: Sequence[str] = INPUT_FILES,
) -> dict[str, Any]:
    """The level-A reading of one variant — refuses to run unless §13.1 step 4 holds.

    1. Recomputes the instrument (:func:`instrument`'s code, same seeds) and calls
       ``tree.verify_for_reading`` on it for :func:`section_name`'s section: the file must
       be tracked and unmodified at HEAD, inputs and versions must match, and every
       threshold must be bitwise equal. ``thresholds``, if given, must equal the committed
       section. Otherwise ``tree.ReadingRefused``, before any statistic.
    2. A lock the instrument found UNDECIDABLE before its reading is not read (§13.4).
    3. A-1 (§12.6 "Reading"): ``Δ_A1 = SR(selector) − SR(control)`` on the test sessions
       at 5 (decides), 10 and 20 bp; ``p_A1 = max(p_boot, p_HAC)`` with the sign rule and
       ``p := 1`` for Δ ≤ 0 (``tree.pair_reading``); the 1,000 placebo arms (``m``
       re-estimated per draw, selector rebuilt, paired against the same control) giving
       ``Δ^(j)``, ``β^(j)`` and the two percentiles; ``Δ_W`` on witness W1; per-fold
       components (§12.14); verdict lines 1-8 of :func:`a1_verdict`.
    4. A-2 (§12.7), for variants that evaluate level A: ``R``, ``p_A2 =
       placebo_p_value(R, R^(·))``, its percentile, ``R_W`` on W1, per-fold means.
    5. §8 control 5, only for a lock that could PASS (at the provisional bar, or for a
       primary under a Holm rejection): the lock rebuilt on the 18-feature partition
       (:func:`pit_variant`) with its own draws (seed 0), its own thresholds by the same
       procedures, the same witness, and its p compared with 0.05/6.
    6. The provisional verdicts at the Bonferroni bar (``verdicts``, §13.3, §13.5), the
       p-values entering Holm (``holm_p``) and the §13.5 row metrics (``trial_rows``).
       Nothing is logged here. The final verdicts after the Holm step, once C is read:
       :func:`lock_verdicts` with the rejected set.

    d rows (``variant.levels == ("A-1",)``) read A-1 only, with the primary's partition
    and so the same ``m`` (§12.13).
    """
    with threadpool_limits(limits=1):
        run = _run(data, variant, draws, boot_draws=boot_draws, partition=partition,
                   workers=workers)
        stored = tree.verify_for_reading(run.printout, section=section_name(variant), path=path,
                                         root=root, inputs=inputs)
        if thresholds is not None and not _same(thresholds, stored):
            raise tree.ReadingRefused("the thresholds passed differ from the committed section")
        return _read(data, run, workers=workers)


# ---------------------------------------------------------------------------------------
# verdicts after Holm and trial rows — §13.2, §13.3, §13.5
# ---------------------------------------------------------------------------------------


def lock_verdicts(
    reading: Mapping[str, Any], rejected: Collection[str] | None = None
) -> dict[str, str]:
    """The lock verdicts and, where A-2 was evaluated, the level verdict of a reading.

    ``rejected`` is the set of primaries the final Holm step rejects (§13.3), known once C
    is read; ``None`` gives the provisional Bonferroni reading ``p ≤ 0.05/6`` that the
    results file states and the trial rows log (§13.5). A sensitivity never enters Holm
    and refuses ``rejected``. A lock not read (UNDECIDABLE before its reading) stays
    UNDECIDABLE. A would-be PASS goes through ``tree.apply_pit`` with the reading's PIT
    rebuild (§8 control 5), which :func:`read` computed for every lock that could pass.
    The level verdict is §13.2's (``tree.level_verdict``); a d row has none.
    """
    if rejected is not None and reading["role"] != "primary":
        raise ValueError("Holm runs over the six primaries only (§13.3); a sensitivity reads "
                         "its locks at 0.05/6")
    out: dict[str, str] = {}
    for name, lines in _LINES.items():
        lock = reading.get(name)
        if lock is None:
            continue
        if not lock["read"]:
            out[name] = UNDECIDABLE
            continue
        holm = None if rejected is None else name in rejected
        raw = lines(**lock["verdict_inputs"], holm_rejected=holm)
        pit = lock.get("pit")
        out[name] = tree.apply_pit(raw, None if pit is None else pit["verdict_lines"])
    if "A-2" in out:
        out[SECTION] = tree.level_verdict([out["A-1"], out["A-2"]])
    return out


def _note(lock: Mapping[str, Any], verdict: str) -> str:
    if verdict == UNDECIDABLE and lock.get("undecidable_reasons"):
        return "UNDECIDABLE: " + "; ".join(lock["undecidable_reasons"])
    return ""


def trial_rows(
    reading: Mapping[str, Any],
    variant: Variant,
    verdicts: Mapping[str, str] | None = None,
    *,
    primary_level: str | None = None,
) -> dict[str, dict[str, Any]]:
    """The §13.5 metrics of the level-A rows of one variant, keyed by ``test``, ready for
    ``tree.log_trial(test, variant, **row)``. Nothing is logged here.

    - Primary: ``A-1`` (sharpe = the selector's 5 bp net Sharpe, delta = Δ_A1, threshold
      = T_A1, p = p_A1, placebo_pct = the difference percentile) and ``A-2`` (sharpe NaN,
      delta = R, threshold = the null's q99, which R must exceed, p = p_A2, placebo_pct =
      R's percentile in the null); verdict = the lock verdict (provisional, §13.5).
    - K and smoothing rows: one row ``A`` with A-1's sharpe, delta, threshold, p and
      placebo_pct, ``p2`` = A-2's p, verdict = the level verdict (§13.2). If
      ``primary_level`` is PASS and delta ≤ 0, the note says PASS (not robust) (§12.13).
    - d rows: one row ``A-1``, A-1's metrics only, verdict = A-1's lock verdict.

    ``verdicts`` default to the reading's provisional ones. A lock UNDECIDABLE before its
    reading spends no row (§13.4): a primary lock's row is dropped, a d row too, and a
    K or smoothing row unless the other lock was read. The PIT rebuild spends none. An
    UNDECIDABLE row's note carries the reading that caused it (§13.4).
    """
    if variant.role == "control":
        return {}
    verdicts = reading["verdicts"] if verdicts is None else verdicts
    n = int(reading["sessions"])
    a1 = reading["A-1"]
    five = _cost_key(DECIDING_BPS)
    if a1["read"]:
        a1_row = {"sharpe": a1["sharpe"]["selector"][five], "delta": a1["delta"][five],
                  "threshold": a1["T_A1"], "p": a1["p"], "placebo_pct": a1["diff_pct"],
                  "sessions": n}
    else:
        a1_row = {"sharpe": NAN, "delta": NAN, "threshold": NAN, "p": NAN,
                  "placebo_pct": NAN, "sessions": n}
    if variant.role == "primary":
        rows = {}
        if a1["read"]:
            rows["A-1"] = {**a1_row, "verdict": verdicts["A-1"],
                           "note": _note(a1, verdicts["A-1"])}
        a2 = reading["A-2"]
        if a2["read"]:
            rows["A-2"] = {"sharpe": NAN, "delta": a2["R"], "threshold": a2["q99"],
                           "p": a2["p"], "placebo_pct": a2["placebo_pct"], "sessions": n,
                           "verdict": verdicts["A-2"], "note": _note(a2, verdicts["A-2"])}
        return rows
    if "A" not in variant.levels:
        if not a1["read"]:
            return {}
        return {"A-1": {**a1_row, "p2": NAN, "verdict": verdicts["A-1"],
                        "note": _note(a1, verdicts["A-1"])}}
    a2 = reading["A-2"]
    if not (a1["read"] or a2["read"]):
        return {}
    level = verdicts[SECTION]
    notes = [n_ for n_ in (_note(a1, verdicts["A-1"]), _note(a2, verdicts["A-2"])) if n_]
    delta = a1_row["delta"]
    if primary_level == PASS and np.isfinite(delta) and delta <= 0:
        notes.append("PASS (not robust): delta <= 0 at this sensitivity (§12.13)")
    return {"A": {**a1_row, "p2": a2["p"], "verdict": level, "note": " | ".join(notes)}}

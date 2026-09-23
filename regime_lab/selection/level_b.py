"""Level B of the Two Sigma tree, the second-moment channel — `docs/PRESPEC_TWOSIGMA.md`.

The specification is `docs/PRESPEC_TWOSIGMA.md`, **LOCKED at commit 30f7d69**. This module
implements its §6 "Level B", §12.8 (B-1: the covariance forecast and its two volatility
witnesses), §12.9 (B-2: risk parity on the state-conditional matrix against risk parity on
the pooled one), §8 control 5 (the NFCI point-in-time rebuild) at level B, and the level-B
lines of §13.1-§13.5. It builds on the shared core :mod:`regime_lab.selection.tree` and
changes nothing in it. A deviation from the lock is an amendment in
`docs/PROTOCOL_FREEZE.md`, never an edit here.

**Two modes, as §13.1 separates them.**

- :func:`instrument` is §13.1 step 2 at level B. It builds the second-moment matrices,
  the two risk-parity arms and the B-1 null, and returns only what §12.8 "Instrument"
  and §12.9 "Instrument" allow: counts of cells and fallbacks, the arms' turnover with
  ΔT, the kill line, realised sd and cap share, the demeaned-leg MDEs and ``T_B2``, the
  null's 50th/95th/99th percentiles with ``q99 − q50``, and the return-space share of
  blend variance through correlations from fold 5's pooled matrix. It computes no G, no
  Sharpe, no beta, no witness and no placebo arm, and never exposes a matrix or any
  per-state quantity. Its output is the level-B section of the threshold file.
- :func:`read` is §13.1 step 4. It recomputes the instrument, refuses to go further
  unless :func:`regime_lab.selection.tree.verify_for_reading` passes (threshold file
  tracked by git and unmodified at HEAD, input SHA-256 and package versions equal,
  recomputed thresholds bitwise equal), then computes G, ``G_diag``, ``p_B1``, the
  witnesses, ``Δ_B2``, ``p_B2``, the placebo arms, the witness arm, the PIT rebuild for a
  lock that would otherwise pass, the lock verdicts and the trial-row metrics. It has no
  switch that skips the verification.

**Blindness (§11, §13.1 step 2).** Nothing here prints. Dataclasses whose fields carry
return-derived content (:class:`SecondMoments`, :class:`ForecastLosses`,
:class:`RiskParityArms`) show counts only in their ``repr``.

**Non-finite readings (§13.4).** A missing leg, a pooled matrix that is not positive
definite, a broken placebo draw or a failed risk-parity solve gives NaN downstream, and
every verdict function reads UNDECIDABLE on it. No verdict is ever read off a NaN.
"""

from __future__ import annotations

import dataclasses
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

from regime_lab.analysis.bootstrap import stationary_indices
from regime_lab.config import ROOT
from regime_lab.selection import protocol, tree
from regime_lab.selection.library import LIBRARY

NAN = float("nan")
LEVEL = "B"
LOCKS: tuple[str, ...] = tree.LEVEL_LOCKS[LEVEL]
#: §12.8: the two volatility witnesses; W1 alone enters B-2 (§12.9).
WITNESSES: tuple[str, ...] = ("W1", "W2")
#: §12.8: a matrix whose smallest eigenvalue is at most 1e-12 × its trace is not
#: positive definite.
PD_RELATIVE_FLOOR = 1e-12
#: How a non-finite value is written in the instrument's printout, which must stay
#: encodable by ``tree.encode_thresholds`` (it refuses non-finite floats): the core's.
NON_FINITE = tree.NON_FINITE


def _pool(workers: int | None) -> int:
    """``tree.draw_map``'s default (the CPU count) for helpers whose own default is 1."""
    return (os.cpu_count() or 1) if workers is None else workers


# ---------------------------------------------------------------------------------------
# the second-moment matrices — §12.8
# ---------------------------------------------------------------------------------------


def is_positive_definite(matrix: np.ndarray, *, floor: float = PD_RELATIVE_FLOOR) -> bool:
    """§12.8: positive definite iff finite and its smallest eigenvalue > ``floor`` × trace.

    The lock's rule is the negation: "smallest eigenvalue ≤ 1e-12 × trace" is not
    positive definite. A non-finite matrix, or one with a non-positive trace, is not.
    """
    m = np.asarray(matrix, dtype=float)
    if m.ndim != 2 or m.shape[0] != m.shape[1] or m.size == 0 or not np.isfinite(m).all():
        return False
    trace = float(np.trace(m))
    if not trace > 0:
        return False
    return bool(np.linalg.eigvalsh(m).min() > floor * trace)


def _second_moment(x: np.ndarray) -> np.ndarray:
    """``(1/n) Σ_t x_t x_tᵀ`` over the rows of ``x``, about zero (§12.8).

    Symmetrised to rounding, since the risk-parity solve checks symmetry. NaN when no
    row is given or a row holds a NaN.
    """
    if len(x) == 0:
        return np.full((x.shape[1], x.shape[1]), NAN)
    s = x.T @ x / len(x)
    return (s + s.T) / 2.0


@dataclass(frozen=True, eq=False, repr=False)
class SecondMoments:
    """The §12.8 matrices of one fold, estimated on its training segment.

    ``pooled`` is ``Σ_f`` over the fold's training sessions that have a lagged state
    (``n_pooled`` of them); ``states[k]`` is ``Σ_fk`` over the training cell (*f*, *k*)
    — sessions whose lagged state is *k* — for every qualifying *k* whose matrix is
    positive definite. ``fallback_states`` are the qualifying states whose matrix is not,
    which fall back to ``Σ_f`` and are counted. A non-qualifying state has no entry and
    uses ``Σ_f`` (§12.4). The ``repr`` shows counts only: a matrix is never printed.
    """

    fold: int
    pooled: np.ndarray
    pooled_pd: bool
    n_pooled: int
    states: Mapping[int, np.ndarray]
    fallback_states: frozenset[int]

    def matrix(self, state: float) -> np.ndarray:
        """``Σ_fk`` for a state that has one, ``Σ_f`` otherwise (missing state included)."""
        if np.isfinite(state) and int(state) in self.states:
            return self.states[int(state)]
        return self.pooled

    def __repr__(self) -> str:
        return (f"SecondMoments(fold={self.fold}, pooled_pd={self.pooled_pd}, "
                f"n_pooled={self.n_pooled}, states={len(self.states)}, "
                f"fallbacks={len(self.fallback_states)})")


@dataclass(frozen=True, eq=False, repr=False)
class _FoldRows:
    """Where fold *f*'s rows sit in a paths Series (train then test) and its legs there."""

    number: int
    start: int
    stop: int
    n_train: int
    x_train: np.ndarray
    x_test: np.ndarray
    test_sessions: pd.DatetimeIndex


def _fold_rows(legs: pd.DataFrame, paths: pd.Series) -> tuple[_FoldRows, ...]:
    """Each fold's contiguous rows of ``paths`` and the legs on those sessions (§12.2).

    The paths format is ``context.session_paths``'s: index (fold, segment, session), each
    fold's rows contiguous, its training rows then its test rows, sessions increasing.
    Missing legs come out NaN.
    """
    if list(paths.index.names) != ["fold", "segment", "session"]:
        raise ValueError("paths must be indexed by (fold, segment, session)")
    fold = paths.index.get_level_values("fold").to_numpy()
    segment = paths.index.get_level_values("segment").to_numpy()
    session = pd.DatetimeIndex(paths.index.get_level_values("session"))
    out = []
    for number in dict.fromkeys(fold.tolist()):
        where = np.flatnonzero(fold == number)
        start, stop = int(where[0]), int(where[-1]) + 1
        if stop - start != len(where):
            raise ValueError(f"fold {number}: its rows must be contiguous")
        seg = segment[start:stop]
        n_train = int((seg == "train").sum())
        if not ((seg[:n_train] == "train").all() and (seg[n_train:] == "test").all()):
            raise ValueError(f"fold {number}: training rows must precede test rows")
        dates = session[start:stop]
        if not dates.is_monotonic_increasing or dates.has_duplicates:
            raise ValueError(f"fold {number}: its path must run forward in time")
        x = legs.reindex(dates).to_numpy(float)
        out.append(_FoldRows(int(number), start, stop, n_train, x[:n_train], x[n_train:],
                             pd.DatetimeIndex(dates[n_train:])))
    return tuple(out)


def _lagged(values: np.ndarray) -> np.ndarray:
    """A fold's path lagged one session inside the fold (§12.2); float, NaN first."""
    out = np.empty(len(values), dtype=float)
    if len(values):
        out[0] = NAN
        out[1:] = values[:-1]
    return out


def _draw_labels(codes: np.ndarray, j: int) -> np.ndarray:
    """Draw *j* of ``PlaceboDraws.codes`` as float labels, NaN where unlabelled."""
    column = codes[:, j]
    return np.where(column < 0, NAN, column.astype(float))


def _train_cells(
    paths: pd.Series, cells: tree.Cells | Mapping[int, frozenset[int]] | None
) -> Mapping[int, frozenset[int]]:
    if cells is None:
        return tree.cells(paths).train
    if isinstance(cells, tree.Cells):
        return cells.train
    return cells


def _fold_moments(
    rows: _FoldRows, lag_train: np.ndarray, qualifying: frozenset[int]
) -> SecondMoments:
    has_state = np.isfinite(lag_train)
    pooled = _second_moment(rows.x_train[has_state])
    states: dict[int, np.ndarray] = {}
    fallback: set[int] = set()
    for k in sorted(qualifying):
        matrix = _second_moment(rows.x_train[lag_train == k])
        if is_positive_definite(matrix):
            states[int(k)] = matrix
        else:
            fallback.add(int(k))
    return SecondMoments(rows.number, pooled, is_positive_definite(pooled),
                         int(has_state.sum()), states, frozenset(fallback))


def _moments(
    rows: Sequence[_FoldRows], labels: np.ndarray, train_cells: Mapping[int, frozenset[int]]
) -> tuple[SecondMoments, ...]:
    out = []
    for r in rows:
        lagged = _lagged(labels[r.start:r.stop])
        out.append(_fold_moments(r, lagged[:r.n_train], train_cells.get(r.number, frozenset())))
    return tuple(out)


def second_moments(
    legs: pd.DataFrame,
    paths: pd.Series,
    fold: int,
    cells: tree.Cells | Mapping[int, frozenset[int]] | None = None,
) -> SecondMoments:
    """``Σ_f`` and ``Σ_fk`` of fold ``fold`` on a stamped path (§12.8, §12.4, §12.2).

    ``legs`` are the §12.3 signal legs ``x`` (sessions x the ten signals); ``paths`` a
    stamped path in ``session_paths`` format (the partition, a placebo draw or a witness);
    ``cells`` its §12.4 cells (``tree.cells(paths)``, or its ``train`` mapping), computed
    when omitted. States are lagged one session along the fold's own path: the first
    training session has no lagged state and enters neither matrix. Sample estimates about
    zero, no shrinkage; a qualifying cell whose matrix is not positive definite falls back
    to ``Σ_f`` and is counted (:attr:`SecondMoments.fallback_states`).
    """
    rows = {r.number: r for r in _fold_rows(legs, paths)}
    if fold not in rows:
        raise KeyError(f"fold {fold} is not in the paths")
    r = rows[fold]
    lagged = _lagged(paths.to_numpy(float)[r.start:r.stop])
    return _fold_moments(r, lagged[:r.n_train], _train_cells(paths, cells).get(fold, frozenset()))


def correlation_share(matrix: np.ndarray) -> float:
    """Share of blend variance through the correlations, §6 Level B's formula.

    ``n(n−1)ρ̄ / (n + n(n−1)ρ̄)``, ρ̄ the signed mean pairwise correlation of ``matrix``
    (correlations of the second moment about zero, ``Σ_ij / √(Σ_ii Σ_jj)``). With the
    lock's position-space stand-in ρ̄ = +0.035 over ten signals it is 3.15 / 13.15 =
    23.9%. §12.8 "Instrument" prints it on fold 5's pooled training matrix, in return
    space. NaN on a non-finite matrix or a non-positive variance.
    """
    m = np.asarray(matrix, dtype=float)
    diag = np.diag(m)
    if not np.isfinite(m).all() or not (diag > 0).all():
        return NAN
    corr = m / np.sqrt(np.outer(diag, diag))
    n = len(diag)
    off = float(corr.sum() - np.trace(corr))
    total = n + off
    return off / total if total > 0 else NAN


# ---------------------------------------------------------------------------------------
# the QLIKE loss and B-1's statistic — §12.8
# ---------------------------------------------------------------------------------------


def qlike(sigma: np.ndarray, x: np.ndarray) -> np.ndarray:
    """The Gaussian QLIKE of §12.8: ``ℓ_t(Σ) = log det Σ + x_tᵀ Σ⁻¹ x_t`` for each row.

    ``sigma`` is one p x p matrix, applied to every row of ``x`` (n x p), or a stack of
    n matrices (n x p x p), one per row. Uses ``slogdet`` and ``solve``, never an
    explicit inverse. A row reads NaN where its matrix is not finite or its determinant
    is not positive, or where the row holds a NaN.
    """
    s = np.asarray(sigma, dtype=float)
    v = np.asarray(x, dtype=float)
    if v.ndim == 1:
        v = v[None, :]
    n, p = v.shape
    out = np.full(n, NAN)
    if s.ndim == 2:
        if s.shape != (p, p) or not np.isfinite(s).all():
            return out
        sign, logdet = np.linalg.slogdet(s)
        if not sign > 0:
            return out
        solved = np.linalg.solve(s, v.T)
        return logdet + np.einsum("ij,ji->i", v, solved)
    if s.shape != (n, p, p):
        raise ValueError("a stack of matrices needs one p x p matrix per row of x")
    finite = np.flatnonzero(np.isfinite(s).all(axis=(1, 2)))
    if finite.size:
        sign, logdet = np.linalg.slogdet(s[finite])
        good, logdet = finite[sign > 0], logdet[sign > 0]
        if good.size:
            solved = np.linalg.solve(s[good], v[good][..., None])[..., 0]
            out[good] = logdet + np.einsum("ij,ij->i", v[good], solved)
    return out


def diagonal_forecast(state_matrix: np.ndarray, pooled: np.ndarray) -> np.ndarray:
    """§12.8's diagnostic forecast: per-state variances with the pooled correlation.

    ``D_k C_f D_k``, with ``D_k = diag(√diag Σ_fk)`` and ``C_f`` the correlation matrix of
    ``Σ_f`` (about zero). Its loss gives ``G_diag``, never decisive.
    """
    sd_pooled = np.sqrt(np.diag(pooled))
    corr = pooled / np.outer(sd_pooled, sd_pooled)
    sd_state = np.sqrt(np.diag(state_matrix))
    return corr * np.outer(sd_state, sd_state)


@dataclass(frozen=True, eq=False, repr=False)
class ForecastLosses:
    """QLIKE losses of every test session, pooled and state-conditional (§12.8).

    ``sessions`` are the test sessions of every fold, in date order; ``pooled`` is
    ``ℓ_t(Σ_f)`` and ``state`` ``ℓ_t(Σ_f,k(t))``, *k(t)* the lagged state along fold
    *f*'s path, equal to ``pooled`` at a ``fallback`` session (lagged state missing, its
    training cell not qualifying, or its matrix not positive definite). The ``repr``
    shows counts only.
    """

    sessions: pd.DatetimeIndex
    pooled: np.ndarray
    state: np.ndarray
    fallback: np.ndarray
    moments: tuple[SecondMoments, ...]

    @property
    def gain(self) -> np.ndarray:
        """``ℓ_t(Σ_f) − ℓ_t(Σ_f,k(t))`` per test session; positive when the state wins."""
        return self.pooled - self.state

    @property
    def fallback_share(self) -> float:
        return float(self.fallback.mean()) if len(self.fallback) else NAN

    def __repr__(self) -> str:
        return (f"ForecastLosses({len(self.sessions):,} test sessions, "
                f"{int(self.fallback.sum()):,} at the pooled matrix)")


def _fallback_mask(lag_test: np.ndarray, moments: SecondMoments) -> np.ndarray:
    return ~np.isin(lag_test, np.array(sorted(moments.states), dtype=float))


def _fold_losses(
    rows: _FoldRows, lag_test: np.ndarray, moments: SecondMoments, diagonal: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pooled = qlike(moments.pooled, rows.x_test)
    state = pooled.copy()
    for k, matrix in moments.states.items():
        in_cell = lag_test == k
        if in_cell.any():
            forecast = diagonal_forecast(matrix, moments.pooled) if diagonal else matrix
            state[in_cell] = qlike(forecast, rows.x_test[in_cell])
    return pooled, state, _fallback_mask(lag_test, moments)


def _losses(
    rows: Sequence[_FoldRows],
    labels: np.ndarray,
    train_cells: Mapping[int, frozenset[int]],
    *,
    diagonal: bool = False,
) -> ForecastLosses:
    """The one code path that turns labels into losses, for the partition, every placebo
    draw and both witnesses alike (§12.8, §12.11)."""
    pooled, state, fallback, moments = [], [], [], []
    for r in rows:
        lagged = _lagged(labels[r.start:r.stop])
        m = _fold_moments(r, lagged[:r.n_train], train_cells.get(r.number, frozenset()))
        p, s, f = _fold_losses(r, lagged[r.n_train:], m, diagonal)
        pooled.append(p)
        state.append(s)
        fallback.append(f)
        moments.append(m)
    sessions = pd.DatetimeIndex(np.concatenate([r.test_sessions.to_numpy() for r in rows]))
    return ForecastLosses(sessions, np.concatenate(pooled), np.concatenate(state),
                          np.concatenate(fallback), tuple(moments))


def forecast_losses(
    legs: pd.DataFrame,
    paths: pd.Series,
    cells: tree.Cells | Mapping[int, frozenset[int]] | None = None,
    *,
    diagonal: bool = False,
) -> ForecastLosses:
    """Pooled and state-conditional QLIKE on every test session of a stamped path (§12.8).

    Matrices are re-estimated from ``paths``' own training labels, fold by fold; cells
    qualify on ``paths``' own stamped path. ``diagonal=True`` replaces each ``Σ_fk`` by
    :func:`diagonal_forecast` (``G_diag``). **A reading**: on the real partition this is
    the input of G, which only :func:`read` computes.
    """
    rows = _fold_rows(legs, paths)
    return _losses(rows, paths.to_numpy(float), _train_cells(paths, cells), diagonal=diagonal)


def _mean_or_nan(values: np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    if v.size == 0 or not np.isfinite(v).all():
        return NAN
    return float(v.mean())


def g_statistic(losses: ForecastLosses) -> float:
    """§12.8: ``G = mean over the test sessions of ℓ_t(Σ_f) − ℓ_t(Σ_f,k(t))``.

    A fallback session contributes zero. NaN if any session's loss is not finite.
    """
    return _mean_or_nan(losses.gain)


def _null_job(
    shared: tuple[tuple[_FoldRows, ...], np.ndarray, Mapping[int, frozenset[int]]], j: int
) -> float:
    rows, codes, train_cells = shared
    return g_statistic(_losses(rows, _draw_labels(codes, j), train_cells))


def b1_null(
    legs: pd.DataFrame,
    paths: pd.Series,
    draws: tree.PlaceboDraws,
    cells: tree.Cells | Mapping[int, frozenset[int]] | None = None,
    *,
    workers: int | None = None,
) -> np.ndarray:
    """B-1's null, ``G^(j)`` on each placebo draw (§12.8 "Null", §12.11).

    Every matrix is re-estimated on the draw's training labels and the draw's test labels
    pick the forecast, through the same code as G. Draws qualify exactly the real cells
    (``tree.placebo_draws``), so the real ``cells.train`` serves every draw. Non-finite
    draws are kept as NaN: the lock is then UNDECIDABLE (§13.4). Ordered by *j* whatever
    ``workers`` is.
    """
    if not draws.drawn_on(paths):
        raise ValueError("the draws were not drawn on these paths")
    rows = _fold_rows(legs, paths)
    shared = (rows, draws.codes, dict(_train_cells(paths, cells)))
    return np.asarray(tree.draw_map(_null_job, draws.n, shared, workers=workers), dtype=float)


# ---------------------------------------------------------------------------------------
# the witnesses' test — §12.8
# ---------------------------------------------------------------------------------------


def _boot_job(shared: tuple[np.ndarray, tuple[int, ...], int, int], j: int) -> np.ndarray:
    values, blocks, draws, seed = shared
    rng = np.random.default_rng(seed)
    n = len(values)
    columns = [np.ascontiguousarray(values[:, c]) for c in range(values.shape[1])]
    means = np.empty((draws, len(columns)))
    for d in range(draws):
        idx = stationary_indices(n, blocks[j], rng)
        for c, column in enumerate(columns):
            means[d, c] = column[idx].mean()
    return np.array([means[:, c].std(ddof=1) for c in range(len(columns))])


def bootstrap_mean_se(
    values: np.ndarray,
    *,
    blocks: Sequence[int] = tree.POWER_BLOCKS,
    draws: int = tree.BOOT_DRAWS,
    seed: int = tree.BOOT_SEED,
    workers: int | None = 1,
) -> dict[int, np.ndarray]:
    """sd (ddof 1) of the stationary-bootstrap mean, per mean block (§12.8, §12.12).

    ``analysis.bootstrap.stationary_indices`` at each mean block, ``draws`` index paths
    from ``numpy.random.default_rng(seed)`` seeded afresh for each block (as
    ``analysis.bootstrap.paired_sharpe_difference`` does per call). ``values`` is one
    series or n x m columns; every column is resampled on the same index paths, which is
    the same as bootstrapping each column alone with the same seed. Returns
    ``{block: array of m SEs}``.
    """
    v = np.asarray(values, dtype=float)
    v = v[:, None] if v.ndim == 1 else v
    blocks = tuple(int(b) for b in blocks)
    results = tree.draw_map(_boot_job, len(blocks), (v, blocks, int(draws), int(seed)),
                            workers=min(_pool(workers), len(blocks)))
    return dict(zip(blocks, results, strict=True))


@dataclass(frozen=True)
class WitnessReading:
    """§12.8's one-sided test that the context forecasts better than a witness.

    ``d`` is ``d^W_t = ℓ_t(Σ_W,k_W(t)) − ℓ_t(Σ_f,k(t))``, positive when the context wins.
    ``p_hac`` = 1 − Φ(t_HAC-6); ``p_boot`` = 1 − Φ(mean / SE_boot), SE_boot the
    stationary-bootstrap SE of the mean at the block (21 / 63 / 126) that gives the
    largest SE, the first on a tie; ``p`` = the larger. All NaN on a non-finite ``d``.
    """

    mean: float
    t_hac: float
    se_boot: dict[int, float]
    block: int
    p_hac: float
    p_boot: float
    p: float

    @property
    def beaten(self) -> bool | None:
        """Is the witness beaten at one-sided 0.05? ``None`` on a non-finite reading."""
        return bool(self.p <= tree.WITNESS_P_BAR) if np.isfinite(self.p) else None


def _witness_reading(
    d: np.ndarray, se: Mapping[int, float], lags: int = tree.HAC_LAGS
) -> WitnessReading:
    blocks = tuple(se)
    nan = WitnessReading(NAN, NAN, {b: NAN for b in blocks}, 0, NAN, NAN, NAN)
    if d.size < 2 or not np.isfinite(d).all():
        return nan
    mean = float(d.mean())
    t = protocol.paired_hac_t(pd.Series(d), pd.Series(np.zeros(len(d))), lags=lags)
    block = max(blocks, key=lambda b: (se[b], -blocks.index(b)))
    se_star = float(se[block])
    if not (np.isfinite(t) and np.isfinite(se_star) and se_star > 0):
        return WitnessReading(mean, t, dict(se), block, NAN, NAN, NAN)
    p_hac = float(stats.norm.sf(t))
    p_boot = float(stats.norm.sf(mean / se_star))
    return WitnessReading(mean, float(t), dict(se), block, p_hac, p_boot, max(p_hac, p_boot))


def witness_tests(
    differentials: Mapping[str, np.ndarray],
    *,
    blocks: Sequence[int] = tree.POWER_BLOCKS,
    draws: int = tree.BOOT_DRAWS,
    seed: int = tree.BOOT_SEED,
    lags: int = tree.HAC_LAGS,
    workers: int | None = 1,
) -> dict[str, WitnessReading]:
    """:class:`WitnessReading` of each loss differential (§12.8 "Two witnesses").

    The differentials must share one length (the test sessions). They are bootstrapped
    together on the same index paths; a non-finite one reads NaN without touching the
    others.
    """
    names = list(differentials)
    arrays = [np.asarray(differentials[n], dtype=float) for n in names]
    if len({a.shape for a in arrays}) > 1:
        raise ValueError("the differentials must share one length")
    finite = [a.size >= 2 and np.isfinite(a).all() for a in arrays]
    se: dict[int, np.ndarray] = {int(b): np.full(len(names), NAN) for b in blocks}
    if any(finite):
        columns = np.column_stack([a for a, ok in zip(arrays, finite, strict=True) if ok])
        booted = bootstrap_mean_se(columns, blocks=blocks, draws=draws, seed=seed,
                                   workers=workers)
        where = np.flatnonzero(finite)
        for b, values in booted.items():
            se[b][where] = values
    return {name: _witness_reading(a, {b: float(se[b][i]) for b in se}, lags)
            for i, (name, a) in enumerate(zip(names, arrays, strict=True))}


def witness_test(d: np.ndarray, **kwargs: Any) -> WitnessReading:
    """:func:`witness_tests` on one differential."""
    return witness_tests({"W": d}, **kwargs)["W"]


# ---------------------------------------------------------------------------------------
# B-2's risk-parity arms — §12.9
# ---------------------------------------------------------------------------------------


def _risk_parity(matrix: np.ndarray) -> np.ndarray | None:
    try:
        return np.asarray(protocol.risk_parity_mix(matrix), dtype=float)
    except (ValueError, RuntimeError):
        return None


def risk_parity_tables(
    moments: Sequence[SecondMoments], names: Sequence[str] = LIBRARY
) -> tuple[dict[int, pd.DataFrame], dict[int, pd.Series]] | None:
    """``risk_parity_mix`` of every matrix (§12.9): a states x signals table per fold for
    its ``Σ_fk``, and ``risk_parity_mix(Σ_f)`` per fold, the pooled arm and the fallback.

    ``None`` when a pooled matrix is not positive definite (the level is UNDECIDABLE,
    §12.8, §13.4) or a solve fails: the arm is then not built and every reading on it is
    NaN. States without a matrix (non-qualifying, or not positive definite) have no row,
    so ``tree.traded_mix`` holds the fallback there.
    """
    names = list(names)
    tables: dict[int, pd.DataFrame] = {}
    pooled: dict[int, pd.Series] = {}
    for m in moments:
        base = _risk_parity(m.pooled) if m.pooled_pd else None
        if base is None:
            return None
        pooled[m.fold] = pd.Series(base, index=names)
        rows: dict[int, np.ndarray] = {}
        for k, matrix in m.states.items():
            weights = _risk_parity(matrix)
            if weights is None:
                return None
            rows[k] = weights
        tables[m.fold] = pd.DataFrame(
            np.array([rows[k] for k in rows]).reshape(len(rows), len(names)),
            index=pd.Index(list(rows), dtype=np.int64), columns=names,
        )
    return tables, pooled


@dataclass(frozen=True, eq=False, repr=False)
class RiskParityArms:
    """B-2's two arms on the traded path (§12.9, §12.3).

    ``pooled_mix`` holds ``risk_parity_mix(Σ_f)`` on fold *f*'s test sessions (equal
    weight before the first); ``state`` holds ``risk_parity_mix(Σ_fk)`` of the lagged
    state with the pooled mix as fallback, flagged per test session. ``pooled_weights`` is
    the pooled arm's ``blend``; ``state_weights`` the state arm's ``blend`` rescaled
    session by session to that gross (``protocol.match_gross``). Both books are
    volatility-targeted and netted on the full path, read on the test sessions. The
    ``repr`` shows sd, turnover, cap share and the fallback share only.
    """

    pooled_mix: pd.DataFrame
    state_mix: tree.TradedMix
    pooled_weights: pd.DataFrame
    state_weights: pd.DataFrame
    pooled: tree.BookResult
    state: tree.BookResult

    def __repr__(self) -> str:
        return (f"RiskParityArms(state={self.state!r}, pooled={self.pooled!r}, "
                f"fallback_share={self.state_mix.fallback_share:.3f})")


def _state_arm(
    data: tree.TreeData,
    paths: pd.Series,
    tables: Mapping[int, pd.DataFrame],
    fallback: Mapping[int, pd.Series],
    train_cells: Mapping[int, frozenset[int]],
    reference: pd.DataFrame,
    costs: Sequence[float],
) -> tuple[tree.TradedMix, pd.DataFrame, tree.BookResult]:
    traded = tree.traded_mix(data.sessions, data.folds, tables, paths, fallback,
                             train_cells=train_cells)
    weights = protocol.match_gross(protocol.blend(data.signals, traded.mix), reference)
    return traded, weights, tree.build_book(data, weights=weights, costs=costs)


def risk_parity_arms(
    data: tree.TreeData,
    paths: pd.Series,
    cells: tree.Cells | Mapping[int, frozenset[int]] | None = None,
    *,
    reference: RiskParityArms | None = None,
    costs: Sequence[float] = tree.COST_COLUMNS,
) -> RiskParityArms | None:
    """B-2's arms on a stamped path (§12.9): pooled and state-conditional risk parity.

    Matrices come from ``paths``' own training labels (:func:`second_moments`), then
    ``risk_parity_mix`` per matrix; the state arm is ``tree.traded_mix`` with the fold's
    pooled mix as fallback, ``blend`` to instrument weights, ``match_gross`` to the pooled
    arm's gross, and ``tree.build_book``. ``reference`` passes another path's arms whose
    pooled arm this one's state arm is matched to and read against: the witness arm of
    §12.9 ("against the same pooled arm") and the placebo arms. ``None`` when a pooled
    matrix is not positive definite or a solve fails.
    """
    train_cells = _train_cells(paths, cells)
    moments = _moments(_fold_rows(data.legs, paths), paths.to_numpy(float), train_cells)
    built = risk_parity_tables(moments)
    if built is None:
        return None
    tables, pooled_mixes = built
    if reference is None:
        pooled_mix = tree.per_fold_mix(data.sessions, data.folds, pooled_mixes)
        pooled_weights = protocol.blend(data.signals, pooled_mix)
        pooled_book = tree.build_book(data, weights=pooled_weights, costs=costs)
    else:
        pooled_mix = reference.pooled_mix
        pooled_weights = reference.pooled_weights
        pooled_book = reference.pooled
    traded, weights, book = _state_arm(data, paths, tables, pooled_mixes, train_cells,
                                       pooled_weights, costs)
    return RiskParityArms(pooled_mix, traded, pooled_weights, weights, pooled_book, book)


# ---------------------------------------------------------------------------------------
# verdicts — §12.8, §12.9, §13.2
# ---------------------------------------------------------------------------------------


def _finite(*values: float | None) -> bool:
    return all(v is not None and np.isfinite(v) for v in values)


def b1_verdict(
    *,
    statistic: float,
    p: float,
    exact: bool,
    p_w1: float,
    p_w2: float,
    pooled_pd: bool,
    cells_ok: bool,
    fallback_share: float,
    holm_rejected: bool | None = None,
) -> str:
    """§12.8 "Verdict", before the PIT rebuild (pass the result to ``tree.apply_pit``).

    - UNDECIDABLE: G or p not finite (p is NaN when any ``G^(j)`` is), the placebo check
      not exact, a pooled training matrix not positive definite, and — §13.4, at level
      B — fewer qualifying training cells than the floor, more than half the test
      sessions at the pooled matrix, or a witness reading that is not finite;
    - NOT SHOWN: the lock does not hold (``p ≤ 0.01`` and Holm rejecting, by default the
      provisional ``p ≤ 0.05/6``); B-1 never reads FAIL, no power was measured;
    - DOMINATED: the lock holds and not both ``p_W1 ≤ 0.05`` and ``p_W2 ≤ 0.05``;
    - PASS otherwise.
    """
    dominated = (None if not _finite(p_w1, p_w2)
                 else not (p_w1 <= tree.WITNESS_P_BAR and p_w2 <= tree.WITNESS_P_BAR))
    level_ok = (pooled_pd and cells_ok and _finite(fallback_share)
                and fallback_share <= tree.MAX_FALLBACK_SHARE)
    return tree.placebo_verdict(statistic=statistic, p=p, exact=exact, dominated=dominated,
                                holm_rejected=holm_rejected, other_undecidable=not level_ok)


def b2_verdict(
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
    exact: bool,
    pooled_pd: bool,
    cells_ok: bool,
    holm_rejected: bool | None = None,
) -> str:
    """§12.9 "Verdict lines": §12.6 lines 1-8 with Δ_B2, T_B2, p_B2, B-2's kill, Δ_W,B2.

    Line 1 (UNDECIDABLE) also takes, at level B (§13.4, §12.11), a placebo check that is
    not exact, a pooled training matrix that is not positive definite and fewer
    qualifying training cells than the floor. Pass the result to ``tree.apply_pit``.
    """
    return tree.paired_verdict(
        delta5=delta5, delta10=delta10, threshold=threshold, p=p, kill_fires=kill_fires,
        beta_pct=beta_pct, diff_pct=diff_pct, delta_w=delta_w, fallback_share=fallback_share,
        leg_missing=leg_missing, holm_rejected=holm_rejected,
        other_finite=bool(exact and pooled_pd and cells_ok),
    )


# ---------------------------------------------------------------------------------------
# the instrument — §12.8 "Instrument", §12.9 "Instrument", §13.1 step 2
# ---------------------------------------------------------------------------------------


def section_name(variant: tree.Variant) -> str:
    """The threshold file's section for a variant's level-B instrument (§13.1 step 3,
    ``tree.section_name``): ``B`` for the primary, ``B:<variant name>`` for a sensitivity.
    The PIT rebuild has none: its thresholds are computed at the reading, never committed
    (§8 control 5)."""
    return tree.section_name(LEVEL, variant)


threshold_section = section_name


def pit_variant(variant: tree.Variant) -> tree.Variant:
    """The §8 control-5 rebuild of a variant (``tree.pit_variant``): the 18 features
    without the NFCI, the same K, smoothing, n_init, seed, folds and training start.
    ``tree.PIT18`` for the primary."""
    return tree.pit_variant(variant)


def _check_variant(variant: tree.Variant, allow_control: bool) -> None:
    if LEVEL not in variant.levels:
        raise ValueError(f"{variant.name}: not evaluated at level B (§7)")
    if variant.role == "control" and not allow_control:
        raise ValueError(f"{variant.name}: the PIT rebuild runs inside the reading only "
                         "(§8 control 5), see pit_rebuild")


def _resolve_paths(
    data: tree.TreeData,
    variant: tree.Variant,
    partition: tree.Partition | None,
    paths: pd.Series | None,
) -> pd.Series:
    """The variant's stamped paths: ``partition.paths``, ``paths``, or the refit (§12.2)."""
    if partition is not None and paths is not None:
        raise ValueError("pass a partition or its paths, not both")
    if partition is not None:
        return partition.paths
    return tree.partition_paths(data, variant).paths if paths is None else paths


def _resolve_draws(draws: tree.PlaceboDraws | int | None, paths: pd.Series) -> tree.PlaceboDraws:
    """§12.11's placebo on ``paths``: a count (``None``: 1,000) is drawn at seed 0; draws
    passed in must be this path's (``tree.check_draws``: same index and stamped labels),
    uniform, at seed 0."""
    if draws is None or isinstance(draws, int | np.integer):
        n = tree.N_DRAWS if draws is None else int(draws)
        return tree.placebo_draws(paths, n, tree.PLACEBO_SEED)
    if not isinstance(draws, tree.PlaceboDraws):
        raise TypeError("draws must be a PlaceboDraws, a count or None")
    return tree.check_draws(draws, paths)


_clean = tree.printable


@dataclass(frozen=True, eq=False, repr=False)
class _Instrument:
    """What the instrument built, kept for the reading; ``printout`` is all it shows."""

    variant: tree.Variant
    paths: pd.Series
    cells: tree.Cells
    draws: tree.PlaceboDraws
    rows: tuple[_FoldRows, ...]
    moments: tuple[SecondMoments, ...]
    arms: RiskParityArms | None
    null: np.ndarray
    pair: tree.PairThreshold
    kill: tree.Kill
    pooled_pd: bool
    cells_ok: bool
    fallback_share: float
    printout: dict[str, Any]


def _build_instrument(
    data: tree.TreeData,
    variant: tree.Variant,
    draws: tree.PlaceboDraws | int | None,
    *,
    paths: pd.Series | None,
    workers: int | None,
    boot_draws: int = tree.BOOT_DRAWS,
    allow_control: bool = False,
) -> _Instrument:
    _check_variant(variant, allow_control)
    if list(data.legs.columns) != list(LIBRARY) or list(data.signals) != list(LIBRARY):
        raise ValueError("the legs and the signals must be the ten LIBRARY signals, in order")
    if paths is None:
        paths = tree.partition_paths(data, variant).paths
    draws = _resolve_draws(draws, paths)
    cells = tree.cells(paths)
    rows = _fold_rows(data.legs, paths)
    test = pd.DatetimeIndex(np.concatenate([r.test_sessions.to_numpy() for r in rows]))
    if not test.equals(data.test_sessions):
        raise ValueError("the paths' test sessions are not the tree's")
    labels = paths.to_numpy(float)
    moments = _moments(rows, labels, cells.train)
    pooled_pd = all(m.pooled_pd for m in moments)
    cells_ok = cells.n_train >= variant.cell_floor
    fallback = np.concatenate([_fallback_mask(_lagged(labels[r.start:r.stop])[r.n_train:], m)
                               for r, m in zip(rows, moments, strict=True)])
    fallback_share = float(fallback.mean())
    legs_complete = all(np.isfinite(r.x_test).all() for r in rows)

    null = b1_null(data.legs, paths, draws, cells, workers=workers)
    summary = tree.null_summary(null)

    arms = risk_parity_arms(data, paths, cells) if pooled_pd else None
    if arms is not None:
        pair = tree.pair_threshold(arms.state.net5, arms.pooled.net5, draws=boot_draws,
                                   seed=tree.BOOT_SEED, workers=_pool(workers))
        delta_turnover = arms.state.turnover - arms.pooled.turnover
        kill = tree.kill(delta_turnover, pair.threshold, arms.state.sigma)
    else:
        nan = {b: NAN for b in tree.POWER_BLOCKS}
        pair = tree.PairThreshold(nan, dict(nan), dict(nan), NAN, NAN, 0, NAN)
        kill = tree.Kill(NAN, NAN, None)
    if arms is not None and not np.array_equal(arms.state_mix.fallback.to_numpy(), fallback):
        raise AssertionError("B-1 and B-2 disagree on the fallback sessions")  # pragma: no cover

    level_reasons = []
    if not draws.exact:
        level_reasons.append("placebo check not exact (§12.11)")
    if not pooled_pd:
        bad = [m.fold for m in moments if not m.pooled_pd]
        level_reasons.append(f"pooled matrix not positive definite in fold(s) {bad} (§12.8)")
    if not cells_ok:
        level_reasons.append(f"{cells.n_train} qualifying training cells, below the floor "
                             f"{variant.cell_floor:g} (§13.4)")
    if fallback_share > tree.MAX_FALLBACK_SHARE:
        level_reasons.append("more than half the test sessions at the pooled matrix (§13.4)")
    if not legs_complete:
        level_reasons.append("a signal leg is missing on a test session (§13.4)")
    b1_reasons = list(level_reasons)
    non_finite_draws = int((~np.isfinite(null)).sum())
    if non_finite_draws:
        b1_reasons.append(f"{non_finite_draws} placebo draw(s) of G not finite (§13.4)")
    b2_reasons = list(level_reasons)
    if arms is None:
        b2_reasons.append("the risk-parity arms could not be built (§12.9)")
    else:
        if arms.state.missing or arms.pooled.missing:
            b2_reasons.append("a paired leg is missing on a test session (§13.4)")
        if not np.isfinite(pair.threshold):
            b2_reasons.append("T_B2 not finite (§13.4)")
        if kill.fires is None:
            b2_reasons.append("the kill line is not finite (§13.4)")

    def arm_value(attribute: str) -> dict[str, Any]:
        if arms is None:
            return {"state": NAN, "pooled": NAN}
        return {"state": getattr(arms.state, attribute), "pooled": getattr(arms.pooled, attribute)}

    n_folds = len(rows)
    printout = {
        "level": LEVEL,
        "variant": variant.name,
        "K": variant.K,
        "smoothing": variant.smoothing if variant.smoothing is not None else 0,
        "features": len(variant.features),
        "test_sessions": len(test),
        "cells": {
            "qualifying_training": cells.n_train,
            "training": variant.K * n_folds,
            "per_fold": "/".join(str(len(cells.train[f])) for f in cells.train),
            "floor": float(variant.cell_floor),
            "abstaining_test_sessions": cells.abstaining,
        },
        "matrices": {
            "pooled_positive_definite": [m.pooled_pd for m in moments],
            "cells_not_positive_definite": sum(len(m.fallback_states) for m in moments),
        },
        "fallback": {"test_sessions": int(fallback.sum()), "share": fallback_share},
        "placebo": {"method": draws.method, "draws": draws.n, "seed": draws.seed,
                    "exact": draws.exact},
        "bootstrap": {"method": "stationary", "draws": int(boot_draws), "seed": tree.BOOT_SEED,
                      "blocks": list(tree.POWER_BLOCKS)},
        "B-1": {
            "null": {"q50": summary.q50, "q95": summary.q95, "q99": summary.q99,
                     "q99-q50": summary.resolution},
            "non_finite_draws": non_finite_draws,
            "correlation_share_fold5": correlation_share(
                max(moments, key=lambda m: m.fold).pooled),
        },
        "B-2": {
            "mde": pair.as_dict(),
            "T_B2": pair.threshold,
            "turnover": {**arm_value("turnover"), "delta": kill.delta_turnover},
            "kill": {"k_kill": kill.k_kill, "fires": kill.fires},
            "sigma": arm_value("sigma"),
            "cap_share": arm_value("cap_share"),
            "missing_test_sessions": arm_value("missing"),
        },
        "undecidable": {"B-1": b1_reasons, "B-2": b2_reasons},
    }
    return _Instrument(variant, paths, cells, draws, rows, moments, arms, null, pair, kill,
                       pooled_pd, cells_ok, fallback_share, _clean(printout))


def instrument(
    data: tree.TreeData,
    variant: tree.Variant = tree.PRIMARY,
    draws: tree.PlaceboDraws | int | None = None,
    *,
    partition: tree.Partition | None = None,
    paths: pd.Series | None = None,
    boot_draws: int = tree.BOOT_DRAWS,
    workers: int | None = None,
) -> dict[str, Any]:
    """The level-B instrument (§12.8, §12.9 "Instrument before the reading", §13.1 step 2).

    The variant's stamped paths come from ``partition`` (or ``paths``), else from
    ``tree.partition_paths(data, variant)``; ``draws`` is a ``tree.PlaceboDraws`` on those
    paths (uniform, seed 0) or a count drawn at seed 0 (``None``: 1,000, §12.11). Pass them
    to share one partition and one placebo across levels. ``boot_draws`` is the lock's
    2,000 (§12.6 step 1); another value exists for tests and is written into the printout,
    so a threshold file made with it cannot pass a reading at the lock's value. Returns
    the JSON printout that is the level-B section of the threshold file
    (:func:`section_name`), and nothing else:

    - ``cells``: qualifying training cells (of K x folds), per fold, the floor, and the
      test sessions whose lagged state has no qualifying training cell;
    - ``matrices``: whether each fold's pooled matrix is positive definite, and how many
      qualifying cells fell back for not being so; ``fallback``: test sessions at the
      pooled matrix (B-2's fallback count, >50% UNDECIDABLE);
    - ``placebo``: method, draws, seed and exactness; ``bootstrap``: its draws, seed and
      blocks;
    - ``B-1``: the null's q50, q95, q99 and ``q99 − q50`` (all ``non-finite`` if any draw
      is), the count of non-finite draws, and :func:`correlation_share` of fold 5's pooled
      training matrix;
    - ``B-2``: ``tree.pair_threshold`` on the arms' 5 bp net legs (MDEs at 0.05 and
      0.05/6 per block, ``T_B2``, SE*, its block), turnover of both arms and ΔT (state −
      pooled, restricted first), ``K_kill`` and whether the kill fires, the arms' realised
      sd and cap share, and test sessions with a missing net return;
    - ``undecidable``: why B-1 or B-2 is UNDECIDABLE before its reading, if it is.

    It computes the state matrices and the arms, which the demeaned legs need, and never
    returns them; it computes no G, Sharpe, beta, witness or placebo arm. Non-finite
    values are written ``non-finite``. Deterministic: bitwise the same on every run and
    whatever ``workers`` is.
    """
    _check_variant(variant, allow_control=False)
    paths = _resolve_paths(data, variant, partition, paths)
    with threadpool_limits(limits=1):
        return _build_instrument(data, variant, draws, paths=paths, workers=workers,
                                 boot_draws=boot_draws).printout


# ---------------------------------------------------------------------------------------
# the reading — §12.8, §12.9, §8 control 5, §13.1 step 4, §13.5
# ---------------------------------------------------------------------------------------


def _arm_job(
    shared: tuple[tree.TreeData, tuple[_FoldRows, ...], tree.PlaceboDraws,
                  Mapping[int, frozenset[int]], pd.DataFrame, float],
    j: int,
) -> tuple[float, float]:
    data, rows, draws, train_cells, pooled_weights, pooled_sharpe = shared
    moments = _moments(rows, _draw_labels(draws.codes, j), train_cells)
    built = risk_parity_tables(moments)
    if built is None:
        return NAN, NAN
    tables, pooled_mixes = built
    _, _, book = _state_arm(data, draws.draw(j), tables, pooled_mixes, train_cells,
                            pooled_weights, (tree.DECIDING_BPS,))
    if book.missing:
        return NAN, NAN
    return tree.sharpe(book.net5) - pooled_sharpe, book.beta()


def placebo_arms(
    data: tree.TreeData,
    arms: RiskParityArms,
    paths: pd.Series,
    draws: tree.PlaceboDraws,
    cells: tree.Cells | Mapping[int, frozenset[int]] | None = None,
    *,
    workers: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """B-2's placebo arms (§12.9 "Beta and difference downgrades", §12.6 step 4).

    On each draw, the state matrices are re-estimated on the draw's training labels, risk
    parity is solved on each, and the state arm is rebuilt on the draw's path, matched to
    the same pooled arm's gross, targeted and netted at 5 bp. Returns ``Δ^(j)`` =
    SR(state arm of draw j) − SR(pooled arm) and ``β^(j)``, NaN for a broken draw.
    **A reading**: these are Sharpe differences on returns (§12.0 row 24).
    """
    rows = _fold_rows(data.legs, paths)
    shared = (data, rows, draws, dict(_train_cells(paths, cells)), arms.pooled_weights,
              tree.sharpe(arms.pooled.net5))
    out = tree.draw_map(_arm_job, draws.n, shared, workers=workers)
    return (np.array([d for d, _ in out], dtype=float), np.array([b for _, b in out], dtype=float))


@dataclass(frozen=True, eq=False, repr=False)
class _WitnessLosses:
    losses: ForecastLosses
    arms: RiskParityArms | None


def _witness(data: tree.TreeData, which: str, K: int,
             reference: RiskParityArms | None) -> _WitnessLosses:
    paths = tree.witness_paths(data, which, K)
    cells = tree.cells(paths)
    losses = forecast_losses(data.legs, paths, cells)
    arms = (risk_parity_arms(data, paths, cells, reference=reference)
            if which == "W1" and reference is not None else None)
    return _WitnessLosses(losses, arms)


def _delta(state: tree.BookResult, pooled: tree.BookResult, bps: float) -> float:
    if state.missing or pooled.missing:
        return NAN
    return tree.sharpe(state.net(bps)) - tree.sharpe(pooled.net(bps))


def _evaluate(
    data: tree.TreeData,
    inst: _Instrument,
    *,
    workers: int | None,
    boot_draws: int = tree.BOOT_DRAWS,
    witnesses: Mapping[str, _WitnessLosses] | None = None,
) -> dict[str, Any]:
    """Every level-B reading on an instrument already built, before the PIT rebuild.

    A lock the instrument already found UNDECIDABLE (``printout["undecidable"]``) is not
    read: none of its real statistics is computed, it reads UNDECIDABLE and spends no
    trial row (§13.4). ``boot_draws`` exists for tests; :func:`read` keeps the lock's 2,000.
    """
    variant = inst.variant
    read_b1 = not inst.printout["undecidable"]["B-1"]
    read_b2 = not inst.printout["undecidable"]["B-2"]
    if witnesses is None and (read_b1 or read_b2):
        witnesses = {w: _witness(data, w, variant.K, inst.arms) for w in WITNESSES}

    g = g_diag = p_b1 = pct_b1 = NAN
    tests = {w: _witness_reading(np.array([]), {b: NAN for b in tree.POWER_BLOCKS})
             for w in WITNESSES}
    if read_b1 and witnesses is not None:
        labels = inst.paths.to_numpy(float)
        losses = _losses(inst.rows, labels, inst.cells.train)
        g = g_statistic(losses)
        g_diag = g_statistic(_losses(inst.rows, labels, inst.cells.train, diagonal=True))
        p_b1 = protocol.placebo_p_value(g, inst.null)
        pct_b1 = protocol.placebo_percentile(g, inst.null)
        differentials = {}
        for w in WITNESSES:
            if not witnesses[w].losses.sessions.equals(losses.sessions):
                raise ValueError(f"{w}: its test sessions are not the partition's")
            differentials[w] = witnesses[w].losses.state - losses.state
        tests = witness_tests(differentials, draws=boot_draws, workers=workers)
    b1_inputs = dict(statistic=g, p=p_b1, exact=inst.draws.exact, p_w1=tests["W1"].p,
                     p_w2=tests["W2"].p, pooled_pd=inst.pooled_pd, cells_ok=inst.cells_ok,
                     fallback_share=inst.fallback_share)

    arms = inst.arms
    deltas = {b: NAN for b in tree.COST_COLUMNS}
    reading = tree.PairReading(NAN, NAN, NAN, NAN, NAN, NAN, NAN)
    null_delta = np.full(inst.draws.n, NAN)
    null_beta = np.full(inst.draws.n, NAN)
    beta_state = sharpe_state = sharpe_pooled = delta_w = NAN
    leg_missing = True
    if read_b2 and arms is not None and witnesses is not None:
        leg_missing = bool(arms.state.missing or arms.pooled.missing)
        deltas = {b: _delta(arms.state, arms.pooled, b) for b in tree.COST_COLUMNS}
        sharpe_state = tree.sharpe(arms.state.net5)
        sharpe_pooled = tree.sharpe(arms.pooled.net5)
        reading = tree.pair_reading(deltas[tree.DECIDING_BPS], inst.pair.se_star,
                                    arms.state.net5, arms.pooled.net5)
        beta_state = arms.state.beta()
        null_delta, null_beta = placebo_arms(data, arms, inst.paths, inst.draws, inst.cells,
                                             workers=workers)
        w1_arms = witnesses["W1"].arms
        if w1_arms is not None:
            delta_w = _delta(w1_arms.state, arms.pooled, tree.DECIDING_BPS)
    beta_pct = protocol.placebo_percentile(beta_state, null_beta)
    diff_pct = protocol.placebo_percentile(deltas[tree.DECIDING_BPS], null_delta)
    b2_inputs = dict(
        delta5=deltas[tree.DECIDING_BPS], delta10=deltas[protocol.COST_BPS["conservative"]],
        threshold=inst.pair.threshold, p=reading.p, kill_fires=inst.kill.fires,
        beta_pct=beta_pct, diff_pct=diff_pct, delta_w=delta_w,
        fallback_share=inst.fallback_share, leg_missing=leg_missing, exact=inst.draws.exact,
        pooled_pd=inst.pooled_pd, cells_ok=inst.cells_ok)

    return {
        "B-1": {
            "read": read_b1,
            "G": g,
            "G_diag": g_diag,
            "p": p_b1,
            "placebo_pct": pct_b1,
            "threshold": tree.null_summary(inst.null).q99,
            "witnesses": {w: dataclasses.asdict(tests[w]) for w in WITNESSES},
            "verdict_inputs": b1_inputs,
            "verdict_before_pit": b1_verdict(**b1_inputs),
        },
        "B-2": {
            "read": read_b2,
            "delta": {float(b): v for b, v in deltas.items()},
            "sharpe_state": sharpe_state,
            "sharpe_pooled": sharpe_pooled,
            "reading": dataclasses.asdict(reading),
            "threshold": inst.pair.threshold,
            "kill": dataclasses.asdict(inst.kill),
            "beta_state": beta_state,
            "beta_pct": beta_pct,
            "diff_pct": diff_pct,
            "placebo_delta": null_delta,
            "placebo_beta": null_beta,
            "delta_w": delta_w,
            "verdict_inputs": b2_inputs,
            "verdict_before_pit": b2_verdict(**b2_inputs),
        },
        "_witnesses": witnesses,
    }


def _would_pass(evaluation: Mapping[str, Any]) -> dict[str, bool]:
    """Would a lock pass at the provisional Bonferroni reading or at any Holm outcome?"""
    out = {}
    for lock, verdict in (("B-1", b1_verdict), ("B-2", b2_verdict)):
        inputs = evaluation[lock]["verdict_inputs"]
        out[lock] = tree.PASS in (verdict(**inputs), verdict(**inputs, holm_rejected=True))
    return out


@dataclass(frozen=True, eq=False, repr=False)
class PitRebuild:
    """§8 control 5 at level B: both locks rebuilt on the 18-feature partition.

    ``verdicts`` are §12.8's and §12.9's lines applied to the rebuild, before any PIT line
    (the rebuild's own p compared with 0.05/6, never entering Holm); ``instrument`` its
    thresholds, computed at the reading and never part of the threshold file.
    """

    variant: tree.Variant
    verdicts: dict[str, str]
    instrument: dict[str, Any]
    statistics: dict[str, Any]


def pit_rebuild(
    data: tree.TreeData,
    variant: tree.Variant = tree.PRIMARY,
    *,
    n_draws: int = tree.N_DRAWS,
    workers: int | None = None,
    boot_draws: int = tree.BOOT_DRAWS,
    _witnesses: Mapping[str, _WitnessLosses] | None = None,
) -> PitRebuild:
    """The NFCI point-in-time rebuild of level B (§8 control 5, §12.8, §12.9 line 9).

    The partition refitted on :func:`pit_variant` (the 18 features without ``fin_nfci``
    and ``fin_nfci_chg13w``; same K, ``n_init``, seed, folds, training start), its own
    ``n_draws`` uniform draws at seed 0 (1,000 in the lock), its thresholds by the same
    procedures (T′_B2, the null's percentiles), its witnesses as for the primary, its
    placebo arms; then the lock lines, with the rebuild's own p compared with 0.05/6
    (it never enters Holm). Downgrade only, no trial row. **A reading**: :func:`read`
    calls it, after the verification, for a lock that would otherwise pass.
    """
    rebuild = pit_variant(variant)
    paths = tree.partition_paths(data, rebuild).paths
    with threadpool_limits(limits=1):
        inst = _build_instrument(data, rebuild, int(n_draws), paths=paths, workers=workers,
                                 boot_draws=boot_draws, allow_control=True)
        evaluation = _evaluate(data, inst, workers=workers, boot_draws=boot_draws,
                               witnesses=_rebased(_witnesses, data, inst))
    statistics = {lock: {k: v for k, v in evaluation[lock].items() if k != "verdict_inputs"}
                  for lock in LOCKS}
    return PitRebuild(rebuild, {lock: evaluation[lock]["verdict_before_pit"] for lock in LOCKS},
                      inst.printout, statistics)


def _rebased(
    witnesses: Mapping[str, _WitnessLosses] | None, data: tree.TreeData, inst: _Instrument
) -> Mapping[str, _WitnessLosses] | None:
    """Witness losses do not depend on the partition and are reused; W1's arm is matched
    to the pooled arm of the partition being read, so it is rebuilt against it."""
    if witnesses is None or inst.arms is None:
        return None
    w1 = witnesses["W1"]
    paths = tree.witness_paths(data, "W1", inst.variant.K)
    arms = risk_parity_arms(data, paths, reference=inst.arms)
    return {**witnesses, "W1": _WitnessLosses(w1.losses, arms)}


def _encoded(value: Any) -> Any:
    return json.loads(json.dumps(tree.encode_thresholds(value)))


def lock_verdicts(
    reading: Mapping[str, Any], rejected: Collection[str] | None = None
) -> dict[str, str]:
    """The lock and level verdicts from a reading (§12.8, §12.9, §8 control 5, §13.2).

    ``rejected`` is the set of primaries Holm rejects (§13.3), after C is read; ``None``
    gives the provisional Bonferroni reading ``p ≤ 0.05/6`` that each level's results file
    states. A sensitivity never enters Holm and refuses ``rejected``. A would-be PASS goes
    through ``tree.apply_pit`` with the reading's PIT rebuild, which :func:`read` computed
    whenever a lock could pass under some Holm outcome.
    """
    if rejected is not None and reading.get("role", "primary") != "primary":
        raise ValueError("Holm runs over the six primaries only (§13.3); a sensitivity reads "
                         "its locks at 0.05/6")
    out = {}
    for lock, verdict in (("B-1", b1_verdict), ("B-2", b2_verdict)):
        holm = None if rejected is None else lock in rejected
        raw = verdict(**reading[lock]["verdict_inputs"], holm_rejected=holm)
        pit = reading.get("pit")
        out[lock] = tree.apply_pit(raw, None if pit is None else pit["verdicts"][lock])
    out[LEVEL] = tree.level_verdict([out[lock] for lock in LOCKS])
    return out


def _finite_value(value: Any) -> bool:
    return value is not None and not isinstance(value, bool) and bool(np.isfinite(value))


def _causes(lock: str, reading: Mapping[str, Any]) -> list[str]:
    """What made a lock UNDECIDABLE (§13.4: written with the reading that caused it): the
    instrument's reasons for a lock not read, else the non-finite or failing inputs of its
    verdict lines."""
    pre = reading.get("instrument", {}).get("undecidable", {}).get(lock, [])
    entry = reading[lock]
    if not entry.get("read", True):
        return list(pre)
    inputs = entry.get("verdict_inputs") or {}
    names = {
        "B-1": (("statistic", "G is not finite"),
                ("p", "p_B1 is not finite (a placebo draw of G is not)"),
                ("p_w1", "the W1 witness test is not finite"),
                ("p_w2", "the W2 witness test is not finite")),
        "B-2": (("delta5", "the 5 bp difference is not finite"),
                ("delta10", "the 10 bp difference is not finite"),
                ("threshold", "T_B2 is not finite"),
                ("p", "p_B2 is not finite"),
                ("beta_pct", "the beta percentile is not finite (a placebo arm's or the "
                             "state arm's beta is not)"),
                ("diff_pct", "the difference percentile is not finite (a placebo arm's "
                             "delta is not)"),
                ("delta_w", "the W1 witness arm's difference is not finite")),
    }[lock]
    out = [text for key, text in names if key in inputs and not _finite_value(inputs[key])]
    if lock == "B-2" and inputs.get("kill_fires", False) is None:
        out.append("the kill line is undetermined")
    if inputs.get("leg_missing"):
        out.append("a paired leg is missing on a test session")
    if inputs.get("exact") is False:
        out.append("the placebo check is not exact")
    if inputs.get("pooled_pd") is False:
        out.append("a pooled training matrix is not positive definite")
    if inputs.get("cells_ok") is False:
        out.append("qualifying training cells below the floor")
    share = inputs.get("fallback_share")
    if share is not None and (not _finite_value(share) or share > tree.MAX_FALLBACK_SHARE):
        out.append("more than half the test sessions at the pooled matrix")
    return out or list(pre)


def _note(lock: str, reading: Mapping[str, Any], verdict: str, base: str = "") -> str:
    parts = [base] if base else []
    if verdict == tree.UNDECIDABLE:
        causes = _causes(lock, reading)
        parts.append("UNDECIDABLE: " + ("; ".join(causes) or "a non-finite reading"))
    return " | ".join(parts)


def trial_rows(
    reading: Mapping[str, Any],
    variant: tree.Variant,
    verdicts: Mapping[str, str] | None = None,
    *,
    primary_level: str | None = None,
) -> dict[str, dict[str, Any]]:
    """The §13.5 metrics of the level-B rows, keyed by ``test``, ready for
    ``tree.log_trial(test, variant, **row)``. Nothing is logged here.

    Primary: rows ``B-1`` (sharpe NaN, delta G, threshold the null's q99, p ``p_B1``,
    placebo_pct G's percentile in the null) and ``B-2`` (sharpe the state arm's, delta
    ``Δ_B2`` at 5 bp, threshold ``T_B2``, p ``p_B2`` with 1 when Δ ≤ 0, placebo_pct the
    difference percentile among the placebo arms), each with its lock verdict. K and
    smoothing sensitivities: one row ``B`` with B-2's sharpe, delta, threshold, p and
    placebo_pct, ``p2`` = ``p_B1``, and the level verdict; if ``primary_level`` is PASS
    and delta ≤ 0, the note says PASS (not robust) (§12.13). ``verdicts`` default to the
    reading's provisional ones. A lock that was UNDECIDABLE before its reading
    (``reading[lock]["read"]`` false) spends no row (§13.4); the sensitivity row is spent
    unless neither lock was read; the PIT rebuild spends none. An UNDECIDABLE row's note
    carries the reading that caused it (§13.4).
    """
    if variant.role == "control":
        return {}
    verdicts = reading["verdicts"] if verdicts is None else verdicts
    b1, b2 = reading["B-1"], reading["B-2"]
    was_read = {"B-1": bool(b1.get("read", True)), "B-2": bool(b2.get("read", True))}
    sessions = int(reading["test_sessions"])
    b2_row = {"sharpe": b2["sharpe_state"], "delta": b2["delta"][tree.DECIDING_BPS],
              "threshold": b2["threshold"], "p": b2["reading"]["p"],
              "placebo_pct": b2["diff_pct"], "sessions": sessions}
    if variant.role == "primary":
        rows = {
            "B-1": {"sharpe": NAN, "delta": b1["G"], "threshold": b1["threshold"], "p": b1["p"],
                    "placebo_pct": b1["placebo_pct"], "sessions": sessions,
                    "verdict": verdicts["B-1"],
                    "note": _note("B-1", reading, verdicts["B-1"],
                                  "delta = G, threshold = the null's q99, p = p_B1 (§12.8)")},
            "B-2": {**b2_row, "verdict": verdicts["B-2"],
                    "note": _note("B-2", reading, verdicts["B-2"])},
        }
        return {lock: row for lock, row in rows.items() if was_read[lock]}
    if not any(was_read.values()):
        return {}
    notes = [n for n in (_note("B-1", reading, verdicts["B-1"]),
                         _note("B-2", reading, verdicts["B-2"])) if n]
    delta = b2_row["delta"]
    if primary_level == tree.PASS and _finite_value(delta) and delta <= 0:
        notes.append("PASS (not robust): delta <= 0 at this sensitivity (§12.13)")
    return {LEVEL: {**b2_row, "p2": b1["p"], "verdict": verdicts[LEVEL],
                    "note": " | ".join(notes)}}


def read(
    data: tree.TreeData,
    variant: tree.Variant = tree.PRIMARY,
    draws: tree.PlaceboDraws | int | None = None,
    thresholds: Mapping[str, Any] | None = None,
    *,
    partition: tree.Partition | None = None,
    paths: pd.Series | None = None,
    section: str | None = None,
    boot_draws: int = tree.BOOT_DRAWS,
    workers: int | None = None,
    pit_draws: int = tree.N_DRAWS,
    root: Path = ROOT,
    path: Path = tree.THRESHOLDS_PATH,
    inputs: Sequence[str] = tree.INPUT_FILES,
) -> dict[str, Any]:
    """The level-B reading (§13.1 step 4, §12.8, §12.9, §8 control 5, §13.5).

    1. Recomputes the instrument (same arguments as :func:`instrument`) and calls
       ``tree.verify_for_reading`` on it (section :func:`section_name` unless ``section``
       is given): it raises ``tree.ReadingRefused`` unless the threshold file is tracked
       by git and unmodified at HEAD, the inputs' SHA-256 and the package versions match,
       and the recomputed instrument equals the committed one bitwise. ``thresholds``,
       if given, must equal what is committed, or the reading is refused too (as at
       levels A and C; ``None`` relies on the recomputation alone). There is no way to
       skip this step. ``boot_draws`` (the lock's 2,000) also sets the witnesses'
       bootstrap; ``pit_draws`` is the PIT rebuild's own placebo (the lock's 1,000).
    2. Computes G, ``G_diag`` (diagnostic), ``p_B1`` and G's placebo percentile; the two
       witnesses and their tests; ``Δ_B2`` at 5, 10 and 20 bp, ``p_B2``, the state arm's
       beta, the placebo arms and the two percentiles, and ``Δ_W,B2`` on W1.
    3. Runs the PIT rebuild (``pit_draws`` draws, seed 0) when a lock would pass at the
       provisional reading or under some Holm outcome, then the provisional verdicts.

    Returns every statistic, the provisional lock and level verdicts (``verdicts``), the
    Holm p-values (``holm_p``) and the trial-row metrics (``trial_rows``) — it logs
    nothing. Final verdicts after Holm: :func:`lock_verdicts` with the rejected set.
    """
    _check_variant(variant, allow_control=False)
    paths = _resolve_paths(data, variant, partition, paths)
    with threadpool_limits(limits=1):
        inst = _build_instrument(data, variant, draws, paths=paths, workers=workers,
                                 boot_draws=boot_draws)
    stored = tree.verify_for_reading(inst.printout, section=section or section_name(variant),
                                     path=path, root=root, inputs=inputs)
    if thresholds is not None and _encoded(thresholds) != _encoded(stored):
        raise tree.ReadingRefused("the thresholds passed are not the committed ones")

    with threadpool_limits(limits=1):
        evaluation = _evaluate(data, inst, workers=workers, boot_draws=boot_draws)
    witnesses = evaluation.pop("_witnesses")
    pit = None
    if any(_would_pass(evaluation).values()):
        rebuilt = pit_rebuild(data, variant, n_draws=pit_draws, workers=workers,
                              boot_draws=boot_draws, _witnesses=witnesses)
        pit = {"variant": rebuilt.variant.name, "verdicts": rebuilt.verdicts,
               "instrument": rebuilt.instrument, "statistics": rebuilt.statistics}
    reading: dict[str, Any] = {
        "level": LEVEL,
        "variant": variant.name,
        "role": variant.role,
        "packages": tree.installed_versions(),
        "instrument": inst.printout,
        "test_sessions": len(data.test_sessions),
        **evaluation,
        "pit": pit,
    }
    verdicts = lock_verdicts(reading)
    reading["verdicts"] = verdicts
    reading["holm_p"] = {"B-1": evaluation["B-1"]["p"], "B-2": evaluation["B-2"]["reading"]["p"]}
    reading["trial_rows"] = trial_rows(reading, variant, verdicts)
    return reading

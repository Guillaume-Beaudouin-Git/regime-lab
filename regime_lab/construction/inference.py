"""Bridgewater study — the inference of the lock: the guard, P1 on quarters, bars and power.

`docs/PRESPEC_BRIDGEWATER.md` §12.10-§12.16 name the functions below. Before this module
existed, the guard (``GuardedLevelA``) and the power at a shift (``power_shift``) lived
only in ``scripts/measure_bridgewater_power.py`` at ``39ef608``; that script keeps its
copies as the record of the build, and every instrument calls these.

- **The blindness guard** (:class:`GuardedLevelA`). It refuses to run the engine on any
  label path whose agreement with a forbidden (real) path, under the best relabelling of
  the cells, reaches :data:`GUARD_AGREEMENT`. D is invariant to a relabelling of the
  cells, so an exact-equality guard would let a relabelled real path through; so would a
  near-real variant (another lag, the +1-month stamps).
- **P1 on the quarterly sequence** (:func:`placebo_fold_paths`, :func:`null_statistics`):
  uniform over join-free orders within each calendar block (`evaluate.p1_draws`), each
  draw expanded to one path per fold with the real availability masks, which carry no
  label content. ``weights`` holds a leg at fixed weights (B2's map leg) and draws only
  the evaluation partition.
- **Bars and power of the verdict rule** (:func:`bar`, :func:`verdict_power`,
  :func:`shift_for_power`): a PASS needs ``Δ ≥ T`` and ``p ≤ α_S`` together; the power of
  that rule under a location shift of the null is what the lock states, not the power of
  the percentile line alone.
- **The witness on a common scale** (:func:`z_score`): a reduction standardised in its
  own content-free null, so that partitions with different clocks, occupancies and null
  widths can be compared.
- Level C's placebo and the planted simulations of the lock measurement, as module-level
  tasks for `selection.tree.draw_map` (:func:`c_placebo_task`, :func:`planted_task`).
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from regime_lab.construction import declared as D
from regime_lab.construction import evaluate as E
from regime_lab.construction import sleeves as S
from regime_lab.selection.folds import Fold

#: The guard refuses a path that agrees with a forbidden one on this share of sessions,
#: or more, under the best relabelling of its cells. Measured on 2,000 per-block draws:
#: at most 0.558 for the quadrant and 0.737 for B1, whose test blocks can barely move;
#: the within-vintage axis agrees 0.853 with the declared one, a lag shift about 0.98.
GUARD_AGREEMENT = 0.80
#: Paths that share fewer labelled sessions than this are not compared.
GUARD_MIN_OVERLAP = 63

Labels = pd.Series | Mapping[int, pd.Series]


class GuardRefused(RuntimeError):
    """A real (forbidden) label path reached the engine before the lock."""


# ------------------------------------------------------------------------------ guard


def _codes(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.where(np.isfinite(values), np.nan_to_num(values, nan=-1.0), -1.0).astype(np.int64)


def best_relabel_agreement(a: np.ndarray, b: np.ndarray, *, min_overlap: int = 1) -> float:
    """Share of the sessions labelled in both paths on which they agree, under the best
    one-to-one relabelling of ``a``'s cells onto ``b``'s. NaN below ``min_overlap``.

    ``a`` and ``b`` are integer codes, -1 where there is no label.
    """
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    both = (a >= 0) & (b >= 0)
    overlap = int(both.sum())
    if overlap < max(min_overlap, 1):
        return float("nan")
    n = int(max(a[both].max(), b[both].max())) + 1
    confusion = np.bincount(a[both] * n + b[both], minlength=n * n).reshape(n, n)
    best = max(int(confusion[np.arange(n), list(p)].sum())
               for p in itertools.permutations(range(n)))
    return best / overlap


def _paths_of(labels: Labels) -> list[pd.Series]:
    return list(labels.values()) if isinstance(labels, Mapping) else [labels]


class GuardedLevelA(E.LevelA):
    """`evaluate.LevelA` that refuses any path too close to a forbidden (real) path."""

    def __init__(self, *args: object, max_agreement: float = GUARD_AGREEMENT,
                 min_overlap: int = GUARD_MIN_OVERLAP, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.max_agreement = max_agreement
        self.min_overlap = min_overlap
        self.forbidden: list[np.ndarray] = []
        self.runs = 0
        self.closest_seen = 0.0

    def forbid(self, labels: Labels) -> None:
        """Register a real path, or every fold's path of a real mapping."""
        for path in _paths_of(labels):
            self.forbidden.append(_codes(path.reindex(self.sessions).to_numpy(float)))

    def closeness(self, labels: Labels) -> float:
        """The largest best-relabelling agreement of ``labels`` with a forbidden path."""
        worst = 0.0
        for path in _paths_of(labels):
            codes = _codes(path.reindex(self.sessions).to_numpy(float))
            for bad in self.forbidden:
                if np.array_equal(codes, bad):
                    return 1.0
                share = best_relabel_agreement(codes, bad, min_overlap=self.min_overlap)
                if np.isfinite(share):
                    worst = max(worst, share)
        return worst

    def run(self, labels: Labels, *, weights: np.ndarray | pd.DataFrame | None = None
            ) -> E.LevelAResult:
        share = self.closeness(labels)
        if share >= self.max_agreement:
            raise GuardRefused(f"blindness guard: a path agreeing {share:.3f} with a real "
                               "label path reached LevelA.run")
        self.closest_seen = max(self.closest_seen, share)
        self.runs += 1
        return super().run(labels, weights=weights)


# ----------------------------------------------------------------- P1 on quarters


class FoldPathMaker:
    """Expands a quarterly label vector into one masked session path per fold.

    The masks are `declared.fold_masks`: availability dates only, no label content.
    """

    def __init__(self, quarters: pd.PeriodIndex, available: pd.Series,
                 sessions: pd.DatetimeIndex, folds: Sequence[Fold]):
        self.sessions = pd.DatetimeIndex(sessions)
        self.quarters = pd.PeriodIndex(quarters, freq="Q")
        where = self.quarters.get_indexer(D.session_quarters(self.sessions))
        self.position = where
        self.masks = D.fold_masks(available.reindex(self.quarters), self.sessions, folds)
        self.numbers = [f.number for f in folds]

    def __call__(self, values: np.ndarray) -> dict[int, pd.Series]:
        values = np.asarray(values, dtype=float)
        if values.shape != (len(self.quarters),):
            raise ValueError("one value per quarter")
        path = np.where(self.position >= 0, values[np.clip(self.position, 0, None)], np.nan)
        return {n: pd.Series(np.where(self.masks[n], path, np.nan), index=self.sessions,
                             name="label") for n in self.numbers}


def quarter_draws(labels: pd.Series, n_draws: int, *, seed: int,
                  blocks: pd.Series | None = None) -> pd.DataFrame:
    """P1 draws of a quarterly sequence: quarters × draws, exact per block, or it raises."""
    grouped = None if blocks is None else blocks.reindex(labels.index)
    return E.p1_draws(labels.astype(float), n_draws, seed=seed, blocks=grouped).draws


def placebo_fold_paths(
    partition: D.Partition,
    sessions: pd.DatetimeIndex,
    folds: Sequence[Fold],
    n_draws: int,
    *,
    seed: int,
    blocks: pd.Series | None = None,
) -> Iterator[dict[int, pd.Series]]:
    """The P1 draws of ``partition``, each as the per-fold paths `LevelA.run` reads.

    Draw ``d`` depends on ``seed`` and ``d`` only. Content-free by construction.
    """
    draws = quarter_draws(partition.labels, n_draws, seed=seed, blocks=blocks)
    maker = FoldPathMaker(partition.labels.index, partition.available, sessions, folds)
    for column in draws.columns:
        yield maker(draws[column].to_numpy(float))


def null_statistics(
    engine: E.LevelA,
    partition: D.Partition,
    n_draws: int,
    *,
    statistics: Mapping[str, Callable[[E.LevelAResult], float]],
    seed: int,
    blocks: pd.Series | None = None,
    weights: np.ndarray | pd.DataFrame | None = None,
) -> dict[str, E.NullDistribution]:
    """Level A (or a fixed-weight leg) on P1 draws of a quarterly partition (§12.10).

    Each draw refits the balanced leg on its own training cells, unless ``weights``
    holds the other leg fixed (B2's map leg, §12.13), and reads D on its own test cells.
    """
    out: dict[str, list[float]] = {name: [] for name in statistics}
    for paths in placebo_fold_paths(partition, engine.sessions, engine.folds, n_draws,
                                    seed=seed, blocks=blocks):
        result = engine.run(paths, weights=weights)
        for name, fn in statistics.items():
            out[name].append(float(fn(result)))
    return {name: E.NullDistribution(np.asarray(v, dtype=float)) for name, v in out.items()}


def session_blocks(sessions: pd.DatetimeIndex, folds: Sequence[Fold]) -> pd.Series:
    """P1 blocks for a daily path: each session in its own calendar segment (lag 0)."""
    s = pd.DatetimeIndex(sessions)
    return E.calendar_blocks(s, s, folds, lag=0)


def draws_against_real(real: pd.Series, draws: pd.DataFrame) -> dict[str, float]:
    """Label-only hygiene of a set of draws: identical draws and agreement with the real."""
    truth = real.to_numpy(float)[:, None]
    agree = (draws.to_numpy(float) == truth).mean(axis=0)
    return {"identical": int((agree == 1.0).sum()), "agreement_mean": float(agree.mean()),
            "agreement_max": float(agree.max())}


def blockwise_identical(real: pd.Series, draws: pd.DataFrame, blocks: pd.Series) -> dict[str, int]:
    """Per block: the number of draws that reproduce the real labels inside that block."""
    ids = blocks.reindex(real.index).to_numpy()
    out = {}
    for name in pd.unique(ids):
        mine = ids == name
        same = (draws.to_numpy(float)[mine] == real.to_numpy(float)[mine, None]).all(axis=0)
        out[str(name)] = int(same.sum())
    return out


# ------------------------------------------------------------------ bars and power


def bar(null: E.NullDistribution, alpha: float, floor: float = E.DRAFT_MDE) -> float:
    """``T = max(floor, MDE_shift(α), MDE_sd(α))``; on an atom, ``max(floor, MDE_sd(α))``.

    NaN if any draw is not finite (§13.4: the test is then UNDECIDABLE).
    """
    if not null.finite:
        return float("nan")
    sd_part = null.mde_sd(alpha)
    if null.is_atom:
        return float(max(floor, sd_part))
    return float(max(floor, null.mde_shift(alpha), sd_part))


def p_values(null: np.ndarray, values: np.ndarray | float) -> np.ndarray:
    """``(1 + #{draws ≥ v}) / (1 + n)`` for every value (`protocol.placebo_p_value`)."""
    ordered = np.sort(np.asarray(null, dtype=float))
    v = np.atleast_1d(np.asarray(values, dtype=float))
    above = ordered.size - np.searchsorted(ordered, v, side="left")
    return (1.0 + above) / (1.0 + ordered.size)


def pct_rank(null: np.ndarray, values: np.ndarray | float) -> np.ndarray:
    """Percentile of each value in ``null``, ties counted half (`placebo_percentile`)."""
    ordered = np.sort(np.asarray(null, dtype=float))
    v = np.atleast_1d(np.asarray(values, dtype=float))
    below = np.searchsorted(ordered, v, side="left")
    equal = np.searchsorted(ordered, v, side="right") - below
    return (below + 0.5 * equal) / ordered.size


def percentile_power(null: np.ndarray, shift: float, alpha: float,
                     floor: float | None = None) -> float:
    """Share of draws that, shifted by ``shift``, reach the ``1 − α`` percentile (and
    ``floor``). The power script's ``power_shift`` at ``39ef608``: it tests the
    percentile line alone, which is NOT the verdict rule (see :func:`verdict_power`)."""
    x = np.asarray(null, dtype=float) + shift
    ok = pct_rank(null, x) >= 1.0 - alpha
    if floor is not None:
        ok &= x >= floor
    return float(ok.mean())


def verdict_power(null: np.ndarray, shift: float, threshold: float, alpha: float, *,
                  se: float | None = None) -> float:
    """Power of the PASS line under a location shift: ``Δ ≥ T`` and ``p ≤ α`` together.

    Each draw of the null, moved by ``shift``, is one Δ under the alternative. ``p`` is
    the P1 p-value of that Δ in the unshifted null; with ``se``, the two-sided normal
    p-value ``2(1 − Φ(|Δ|/se))`` must also be at most ``α`` (the bootstrap line).
    """
    x = np.asarray(null, dtype=float) + shift
    ok = (x >= threshold) & (p_values(null, x) <= alpha)
    if se is not None:
        ok &= 2.0 * stats.norm.sf(np.abs(x) / se) <= alpha
    return float(ok.mean())


def shift_for_power(null: np.ndarray, threshold: float, alpha: float, *, power: float = 0.80,
                    se: float | None = None, high: float = 10.0) -> float:
    """The smallest shift at which :func:`verdict_power` reaches ``power`` (bisection)."""
    if verdict_power(null, high, threshold, alpha, se=se) < power:
        return float("nan")
    lo, hi = 0.0, high
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if verdict_power(null, mid, threshold, alpha, se=se) >= power:
            hi = mid
        else:
            lo = mid
    return float(hi)


def z_score(value: float, null: np.ndarray) -> float:
    """A statistic standardised in its own content-free null: ``(x − mean) / sd``."""
    v = np.asarray(null, dtype=float)
    if not (np.isfinite(value) and np.isfinite(v).all() and v.size > 1):
        return float("nan")
    sd = float(np.std(v, ddof=1))
    return float((value - v.mean()) / sd) if sd > 0 else float("nan")


def effective_bar(null: np.ndarray, threshold: float, witness_z: Sequence[float]) -> float:
    """The PASS bar once the witnesses are known: Δ must clear ``T`` and exceed, on the
    common scale, the largest witness z (§12.15): ``max(T, mean + sd · max z_W)``."""
    v = np.asarray(null, dtype=float)
    z = [w for w in witness_z if np.isfinite(w)]
    if not z:
        return float(threshold)
    return float(max(threshold, v.mean() + np.std(v, ddof=1) * max(z)))


def entropy_rank(corr: np.ndarray) -> float:
    """Effective rank ``exp(−Σ p_i log p_i)``, ``p_i = λ_i / Σ λ`` (Roy & Vetterli)."""
    eig = np.clip(np.linalg.eigvalsh(np.asarray(corr, dtype=float)), 0.0, None)
    p = eig / eig.sum()
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


# ---------------------------------------------------------------- planted effects


def plant_variance(excess: pd.DataFrame, path: pd.Series,
                   factors: Mapping[int, Mapping[str, float]]) -> pd.DataFrame:
    """Multiply the variance of chosen instruments by a factor on the sessions of a cell.

    ``factors[k][name] = r`` scales the returns of ``name`` by ``√r`` on every session
    whose label in ``path`` is ``k``. Used only on content-free (placebo) partitions,
    to measure power against a planted, known effect.
    """
    codes = _codes(path.reindex(excess.index).to_numpy(float))
    scale = np.ones(excess.shape)
    columns = list(excess.columns)
    for cell, spec in factors.items():
        rows = codes == int(cell)
        for name, r in spec.items():
            if r <= 0:
                raise ValueError("variance factors must be positive")
            scale[rows, columns.index(name)] *= np.sqrt(r)
    return excess * scale


@dataclass(frozen=True)
class PlantedSetup:
    """What every planted task shares (pickled once per worker)."""

    excess: pd.DataFrame
    configuration: str
    stamped: pd.Series
    folds: tuple[Fold, ...]
    quarters: pd.PeriodIndex
    available: pd.Series
    draws: np.ndarray
    #: One tuple per task: (draw column, scenario name, variance factor, book kind).
    tasks: tuple[tuple[int, str, float, str], ...]
    scenarios: Mapping[str, Mapping[int, tuple[str, ...]]]


def planted_task(setup: PlantedSetup, j: int) -> dict[str, float]:
    """One planted world: plant, rebuild the object, run the declared pipeline.

    Returns the reduction, both legs' D, the volatility-matched reduction and the two
    witnesses' reductions (the last two on the targeted book only).
    """
    column, scenario, r, book = setup.tasks[j]
    labels = pd.Series(setup.draws[:, column], index=setup.quarters)
    session_path = D.contemporaneous_path(labels, pd.DatetimeIndex(setup.excess.index))
    names = setup.scenarios[scenario]
    factors = {cell: dict.fromkeys(members, r) for cell, members in names.items()}
    planted = plant_variance(setup.excess, session_path, factors)
    panel = D.sleeve_panel(setup.configuration, setup.stamped, excess=planted)
    if book == "targeted":
        engine = E.LevelA(panel.sleeve_returns, setup.folds, leg_builder=D.leg_builder(panel))
    else:
        engine = E.LevelA(panel.sleeve_returns, setup.folds, scaling="none")
    maker = FoldPathMaker(setup.quarters, setup.available, panel.sessions, setup.folds)
    result = engine.run(maker(setup.draws[:, column]))
    out = {"reduction": result.reduction, "d_blind": result.d_blind,
           "d_balanced": result.d_balanced}
    if book == "targeted":
        out["p2_vol"] = E.exposure_matched(result, panel.excess, on="vol",
                                           missing_as_zero=True).reduction
        w1 = E.volatility_witness_labels(panel.excess["^GSPC"], setup.folds,
                                         sessions=panel.sessions)
        out["w1"] = engine.run(w1).reduction
        last = D.quarter_last_sessions(setup.quarters, panel.sessions)
        per_fold, _ = D.quarterly_volatility_labels(D.traded_blind_driver(engine),
                                                    panel.sessions, setup.folds, last)
        out["w2"] = engine.run(D.fold_paths(per_fold, last, panel.sessions,
                                            setup.folds)).reduction
    return out


# ------------------------------------------------------------------------- level C


def sharpe(x: pd.Series) -> float:
    """``√252 · mean / sd`` (ddof 1) over the finite values; NaN if undefined."""
    v = x.dropna()
    sd = float(v.std(ddof=1)) if len(v) > 1 else float("nan")
    return float(np.sqrt(E.PERIODS) * v.mean() / sd) if sd > 0 else float("nan")


def transitions_between(draw: pd.Series, first: pd.Timestamp, last: pd.Timestamp
                        ) -> pd.DatetimeIndex:
    """The transition stamps of one stamped sequence that fall in ``[first, last]``."""
    moves = E.transition_stamps(draw)
    return moves[(moves >= first) & (moves <= last)]


def count_in_window(dates: pd.DatetimeIndex, sessions: pd.DatetimeIndex,
                    window: pd.DatetimeIndex) -> int:
    """Rebalances whose first held session (the next session) falls in ``window``."""
    held = E.stamps_to_sessions(dates, sessions, lag=1)
    return int(np.asarray(held.isin(window)).sum())


def conditioned_draws(
    stamps: pd.Series,
    n_accept: int,
    *,
    seed: int,
    blocks: pd.Series,
    sessions: pd.DatetimeIndex,
    window: pd.DatetimeIndex,
    target: int,
    max_draws: int = 20_000,
) -> tuple[pd.DataFrame, np.ndarray]:
    """The first ``n_accept`` P1 draws whose test-span transition count equals ``target``.

    Draws come in their fixed order (draw ``d`` depends on ``seed`` and ``d`` only), so
    the accepted set is deterministic whatever the batch sizes used to reach it. Returns
    the accepted draws and their numbers.
    """
    s0, s1 = pd.DatetimeIndex(sessions)[0], pd.DatetimeIndex(sessions)[-1]
    grouped = blocks.reindex(stamps.index)
    size = min(max_draws, max(2 * n_accept, 16))
    while True:
        draws = E.p1_draws(stamps, size, seed=seed, blocks=grouped).draws
        kept = []
        for d in draws.columns:
            moves = transitions_between(draws[d], s0, s1)
            if count_in_window(moves, sessions, window) == target:
                kept.append(d)
                if len(kept) == n_accept:
                    return draws[kept], np.asarray(kept)
        if size >= max_draws:
            raise RuntimeError(f"only {len(kept)} of {size} draws match {target} transitions")
        size = min(max_draws, 2 * size)


@dataclass(frozen=True)
class CSetup:
    """What every level-C placebo task shares."""

    excess: pd.DataFrame
    sleeves: Mapping[str, tuple[str, ...]]
    dates: tuple[pd.DatetimeIndex, ...]
    test: pd.DatetimeIndex
    calendar_sharpe: float
    cost_column: str = "headline"


def c_placebo_task(setup: CSetup, j: int) -> float:
    """``SR(blind book on draw j's transition dates) − SR(calendar book)`` on the test
    sessions, net of the cost column (§12.14). Content-free: the dates are a P1 draw's."""
    book, _, _ = S.blind_book(setup.excess, setup.sleeves, list(setup.dates[j]))
    net = S.net_returns(book, setup.cost_column).reindex(setup.test)
    return sharpe(net) - setup.calendar_sharpe

"""Replication of Shu, Yu and Mulvey (2024), to their protocol and not ours.

Every earlier hypothesis in this programme was built rather than replicated, and
each null was therefore ambiguous: a result that fails could be a wrong idea or a
different implementation, and there was no way to tell them apart. This module
does the other thing. It follows the published protocol exactly — their three
features, their three-thousand-day window, their six-month refit, their monthly
penalty selection, their one-day delay, their ten basis points — and stops.

Only once the published number is reproduced does it become meaningful to change
one thing. The one thing, here, is the benchmark: the paper measures against buy
and hold, and a companion study found that no regime overlay in it beat a
dynamic volatility target, which is free and delivers the same drawdown
reduction. That comparison is in the runner, not here.

Reference figures for the S&P 500 over 1990-2023, one day delay, ten basis
points: buy and hold at a Sharpe of 0.48 and a worst drawdown of −55.2%, the
jump model at 0.68 and −26.6%, with 44% turnover and 80% average exposure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Half-lives of the three features, in trading days.
HL_DOWNSIDE = 10
HL_SORTINO_FAST = 20
HL_SORTINO_SLOW = 60

#: Estimation window and refit cadence, from the paper.
TRAIN_DAYS = 3000
REFIT_MONTHS = 6

#: Penalty grid searched monthly on the preceding eight years.
LAMBDA_GRID = (0.0, 5.0, 15.0, 35.0, 50.0, 70.0, 100.0, 150.0)
VALIDATION_YEARS = 8

#: One-way transaction cost.
COST_ONE_WAY = 0.0010

#: Signal at the close of t is executed at the close of t+1 and earns the return
#: of t+2, so the position lags the state by two sessions.
EXECUTION_LAG = 2


def _ewm(series: pd.Series, halflife: int) -> pd.Series:
    return series.ewm(halflife=halflife, min_periods=halflife).mean()


def features(excess: pd.Series) -> pd.DataFrame:
    """The paper's three features: downside deviation and two Sortino ratios.

    Deliberately narrow. A companion study fed fifty features to the same model
    family; this one keeps the three the authors chose, because the point is to
    reproduce their number rather than to improve on it.
    """
    downside = np.sqrt(_ewm(excess.where(excess < 0, 0.0) ** 2, HL_DOWNSIDE))
    return pd.DataFrame(
        {
            "downside": downside,
            "sortino_20": _ewm(excess, HL_SORTINO_FAST) / _ewm(downside, HL_SORTINO_FAST),
            "sortino_60": _ewm(excess, HL_SORTINO_SLOW) / _ewm(downside, HL_SORTINO_SLOW),
        }
    ).replace([np.inf, -np.inf], np.nan)


def fit_centroids(
    x: np.ndarray, *, penalty: float, n_states: int = 2, n_init: int = 10, max_iter: int = 20,
    seed: int = 0,
) -> np.ndarray:
    """Coordinate descent between centroids and the penalised state path.

    Ten restarts, as the paper specifies, because the objective is not convex
    and a single initialisation lands in a local minimum often enough to matter.
    """
    rng = np.random.default_rng(seed)
    n, d = x.shape
    best_obj, best = np.inf, None

    for _ in range(n_init):
        centroids = x[rng.choice(n, n_states, replace=False)].copy()
        for _ in range(max_iter):
            states, obj = _viterbi(x, centroids, penalty)
            updated = np.array(
                [
                    x[states == k].mean(axis=0) if (states == k).any() else centroids[k]
                    for k in range(n_states)
                ]
            )
            if np.allclose(updated, centroids):
                centroids = updated
                break
            centroids = updated
        states, obj = _viterbi(x, centroids, penalty)
        if obj < best_obj:
            best_obj, best = obj, centroids.copy()
    return best


def _viterbi(x: np.ndarray, centroids: np.ndarray, penalty: float) -> tuple[np.ndarray, float]:
    """Exact minimiser of the penalised objective by dynamic programming."""
    n, k = len(x), len(centroids)
    loss = 0.5 * ((x[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    value = np.empty((n, k))
    back = np.zeros((n, k), dtype=np.int64)
    value[0] = loss[0]
    for t in range(1, n):
        for state in range(k):
            costs = value[t - 1] + penalty * (np.arange(k) != state)
            back[t, state] = int(np.argmin(costs))
            value[t, state] = loss[t, state] + costs[back[t, state]]
    path = np.empty(n, dtype=np.int64)
    path[-1] = int(np.argmin(value[-1]))
    for t in range(n - 2, -1, -1):
        path[t] = back[t + 1, path[t + 1]]
    return path, float(value[-1].min())


def online_states(x: np.ndarray, centroids: np.ndarray, penalty: float) -> np.ndarray:
    """State at each date using only that date and everything before it.

    The forward recursion of the same dynamic programme, read at each step
    rather than after backtracking. Re-running the full programme on a trailing
    window and keeping its last value gives the same answer at a cost that grows
    with the square of the sample, which is why it is done this way.
    """
    n, k = len(x), len(centroids)
    loss = 0.5 * ((x[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    value = loss[0].copy()
    out = np.empty(n, dtype=np.int64)
    out[0] = int(np.argmin(value))
    for t in range(1, n):
        stay = value
        switch = value.min() + penalty
        value = loss[t] + np.minimum(stay, switch)
        out[t] = int(np.argmin(value))
    return out


def order_states(states: np.ndarray, excess: np.ndarray, n_states: int = 2) -> np.ndarray:
    """Rank states by the cumulative excess return earned in them.

    Ascending, so the last index is the bull state. A companion study lost three
    days to the opposite convention: the reference implementation sorts
    descending and a wrapper assumed otherwise, which held the weakest state for
    an entire backtest without raising anything.
    """
    totals = np.array(
        [excess[states == k].sum() if (states == k).any() else -np.inf for k in range(n_states)]
    )
    rank = {int(state): position for position, state in enumerate(np.argsort(totals))}
    return np.array([rank[int(s)] for s in states], dtype=np.int64)

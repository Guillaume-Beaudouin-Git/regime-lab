"""Idea 6 of the advisor's note: the regime as one input of a risk forecast, on 46 markets.

The protocol is `docs/PRESPEC_RISQUE.md`. The only asset the programme has shown for
the sparse jump state is information about forward *variance* (`docs/RESULTS_FINAL.md`).
This module judges it where that asset should pay: as one regressor of a standard
volatility forecast, per market, estimated on the past alone, scored by a loss that is
robust to the noise of the realised-variance proxy (QLIKE, Patton 2011), and then as the
risk estimate that sizes a book.

**The forecaster.** A log-HAR in the spirit of Corsi (2009): the log of the realised
variance over the next ``horizon`` sessions, regressed on the log of the mean squared
return over the last 5, 22 and 66 sessions, plus optional regressors (the log VIX, a
stress dummy). The daily component of the textbook HAR is replaced by the quarterly one
because the panel sits on a common calendar where each market's own holidays are zero
returns (about 4% of sessions), whose log is minus infinity.

**Past only, refitted every session.** The coefficients used at the close of ``t`` come
from an OLS on the rows ``s <= t - horizon`` only, whose targets are fully realised by
``t``. The expanding fit is computed from cumulative cross-products, so a daily refit
costs one small linear solve per session. The variance forecast is
``exp(x'b + s2 / 2)``, the log-normal correction with the training residual variance.

**Abstention.** A dummy regressor whose two buckets do not both hold ``min_bucket``
training rows is not estimable; the augmented model then abstains and returns the base
model's forecast (the H-b convention of `scripts/run_m3_evaluation.py`). No session is
dropped for want of a conditional estimate.

Nothing here reads a state, a price file or a result.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from regime_lab.analysis.bootstrap import stationary_indices
from regime_lab.extensions.trend import MAX_LEVERAGE, VOL_TARGET, tsmom_signal
from regime_lab.extensions.vehicle import portfolio_scalar

PERIODS = 252
HORIZON = 21
HAR_WINDOWS = (5, 22, 66)
MIN_ROWS = 504
MIN_BUCKET = 63
EWMA_LAMBDA = 0.94
EWMA_WARMUP = 66
BLIND_TOLERANCE = 1e-12


# ---------------------------------------------------------------------------
# targets and regressors
# ---------------------------------------------------------------------------
def log_returns(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Log returns; missing wherever either price is missing or not positive.

    A continuous futures series can settle below zero (crude oil on 2020-04-20). No log
    return exists there, and none is invented: the session stays missing.
    """
    return np.log(prices.where(prices > 0)).diff()


def forward_variance(returns: pd.DataFrame | pd.Series, horizon: int = HORIZON):
    """Sum of squared returns over ``t+1 .. t+horizon``; missing if any of them is."""
    squared = returns**2
    return squared.rolling(horizon, min_periods=horizon).sum().shift(-horizon)


def har_components(returns: pd.Series, windows: Sequence[int] = HAR_WINDOWS) -> pd.DataFrame:
    """Log of the mean squared return over each trailing window ending at ``t``.

    Strict windows: a window with a missing return is missing. A window whose mean is
    exactly zero (a run of stale prices) has no log and is missing too.
    """
    squared = returns**2
    out = {}
    for w in windows:
        mean = squared.rolling(w, min_periods=w).mean()
        out[f"har_{w}"] = np.log(mean.where(mean > 0))
    return pd.DataFrame(out, index=returns.index)


def ewma_log_variance(
    returns: pd.Series, *, lam: float = EWMA_LAMBDA, warmup: int = EWMA_WARMUP,
    horizon: int = HORIZON,
) -> pd.Series:
    """Log of the RiskMetrics variance scaled to ``horizon`` sessions, known at ``t``."""
    ewma = (returns**2).ewm(alpha=1.0 - lam, adjust=False, min_periods=warmup,
                            ignore_na=True).mean()
    return np.log((ewma * horizon).where(ewma > 0)).where(returns.notna())


# ---------------------------------------------------------------------------
# the expanding forecaster
# ---------------------------------------------------------------------------
def _lagged_cumsum(a: np.ndarray, lag: int) -> np.ndarray:
    """Cumulative sum along axis 0, then shifted forward by ``lag`` rows, zero-filled."""
    c = np.cumsum(a, axis=0)
    out = np.zeros_like(c)
    if lag < len(c):
        out[lag:] = c[: len(c) - lag]
    return out


@dataclass(frozen=True)
class Moments:
    """Lagged cumulative cross-products of a design, ready for an expanding fit.

    ``xx[t]``, ``xy[t]``, ``yy[t]`` and ``n[t]`` sum over the training rows
    ``s <= t - horizon``. ``x`` is the design itself, used to forecast at ``t``.
    """

    x: np.ndarray
    xx: np.ndarray
    xy: np.ndarray
    yy: np.ndarray
    n: np.ndarray
    rows: np.ndarray
    y: np.ndarray
    horizon: int


def moments(y: np.ndarray, x: np.ndarray, rows: np.ndarray, *, horizon: int = HORIZON) -> Moments:
    """Build the cumulative moments of ``y`` on ``x`` over the training ``rows``.

    ``rows`` must already exclude any row where ``y`` or ``x`` is not finite; this is
    checked, so that every arm built on the same ``rows`` trains on the same sessions.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    rows = np.asarray(rows, dtype=bool)
    if x.ndim != 2 or len(x) != len(y) or len(rows) != len(y):
        raise ValueError("y, x and rows must share their first dimension")
    if not (np.isfinite(y[rows]).all() and np.isfinite(x[rows]).all()):
        raise ValueError("a training row holds a non-finite value")
    xz = np.where(rows[:, None], x, 0.0)
    yz = np.where(rows, y, 0.0)
    return Moments(
        x=x,
        xx=_lagged_cumsum(np.einsum("ti,tj->tij", xz, xz), horizon),
        xy=_lagged_cumsum(xz * yz[:, None], horizon),
        yy=_lagged_cumsum(yz * yz, horizon),
        n=_lagged_cumsum(rows.astype(float), horizon),
        rows=rows,
        y=y,
        horizon=horizon,
    )


def _solve(xx: np.ndarray, xy: np.ndarray, yy: np.ndarray, n: np.ndarray, x: np.ndarray,
           usable: np.ndarray) -> np.ndarray:
    """Variance forecast ``exp(x'b + s2/2)`` on the ``usable`` sessions, NaN elsewhere."""
    k = x.shape[1]
    out = np.full(len(x), np.nan)
    idx = np.flatnonzero(usable)
    if idx.size == 0:
        return out
    a, b = xx[idx], xy[idx]
    try:
        beta = np.linalg.solve(a, b[..., None])[..., 0]
    except np.linalg.LinAlgError:
        beta = np.einsum("tij,tj->ti", np.linalg.pinv(a), b)
    ssr = yy[idx] - 2.0 * (beta * b).sum(axis=1) + np.einsum("ti,tij,tj->t", beta, a, beta)
    s2 = np.maximum(ssr, 0.0) / (n[idx] - k)
    out[idx] = np.exp((x[idx] * beta).sum(axis=1) + 0.5 * s2)
    return out


def forecast(
    m: Moments, *, min_rows: int = MIN_ROWS, usable: np.ndarray | None = None
) -> np.ndarray:
    """Expanding-OLS variance forecast at every session where it is defined.

    Defined where the training set holds at least ``min_rows`` rows and today's design
    row is finite (and, if given, ``usable`` is true).
    """
    ok = (m.n >= min_rows) & np.isfinite(m.x).all(axis=1)
    if usable is not None:
        ok &= np.asarray(usable, dtype=bool)
    return _solve(m.xx, m.xy, m.yy, m.n, m.x, ok)


def forecast_with_dummy(
    m: Moments, dummy: np.ndarray, *, min_rows: int = MIN_ROWS, min_bucket: int = MIN_BUCKET,
    usable: np.ndarray | None = None, base: np.ndarray | None = None,
) -> np.ndarray:
    """The base design of ``m`` plus one 0/1 ``dummy`` column, fitted on the same rows.

    Only the cross-products that involve the dummy are accumulated here, so a rotation
    placebo costs one pass per draw. Where either bucket of the dummy holds fewer than
    ``min_bucket`` training rows, the model abstains and returns ``base`` (the base
    model's forecast, computed if not given).
    """
    d = np.asarray(dummy, dtype=float)
    if len(d) != len(m.y):
        raise ValueError("the dummy must be on the design's calendar")
    if not np.isfinite(d[m.rows]).all():
        raise ValueError("the dummy is missing on a training row")
    if not np.isin(d[np.isfinite(d)], (0.0, 1.0)).all():
        raise ValueError("the dummy must be 0 or 1")
    h = m.horizon
    dz = np.where(m.rows, d, 0.0)
    xz = np.where(m.rows[:, None], m.x, 0.0)
    yz = np.where(m.rows, m.y, 0.0)
    xd = _lagged_cumsum(xz * dz[:, None], h)
    dd = _lagged_cumsum(dz, h)
    dy = _lagged_cumsum(dz * yz, h)

    k = m.x.shape[1]
    xx = np.zeros((len(d), k + 1, k + 1))
    xx[:, :k, :k] = m.xx
    xx[:, :k, k] = xd
    xx[:, k, :k] = xd
    xx[:, k, k] = dd
    xy = np.column_stack([m.xy, dy])
    x = np.column_stack([m.x, d])

    defined = (m.n >= min_rows) & np.isfinite(x).all(axis=1)
    if usable is not None:
        defined &= np.asarray(usable, dtype=bool)
    estimable = defined & (dd >= min_bucket) & (m.n - dd >= min_bucket)
    out = _solve(xx, xy, m.yy, m.n, x, estimable)
    if base is None:
        base = forecast(m, min_rows=min_rows, usable=usable)
    fallback = defined & ~estimable
    out[fallback] = base[fallback]
    return out


# ---------------------------------------------------------------------------
# losses and tests
# ---------------------------------------------------------------------------
def qlike(target, forecast_):
    """QLIKE loss ``s/h - log(s/h) - 1``: zero at a perfect forecast, robust to proxy noise.

    Missing where either input is missing or not positive.
    """
    if isinstance(target, pd.Series | pd.DataFrame):
        ratio = (target / forecast_).where((target > 0) & (forecast_ > 0))
    else:
        s = np.asarray(target, dtype=float)
        h = np.asarray(forecast_, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where((s > 0) & (h > 0), s / h, np.nan)
    return ratio - np.log(ratio) - 1.0


def hac_mean_t(x: pd.Series | np.ndarray, *, lags: int) -> float:
    """Newey-West t of the mean of ``x`` (missing values dropped); NaN if too short."""
    v = np.asarray(pd.Series(x).dropna(), dtype=float)
    if len(v) < 30 or not v.std(ddof=1) > 0:
        return float("nan")
    fit = sm.OLS(v, np.ones(len(v))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(fit.tvalues[0])


def mde_z(alpha: float, power: float = 0.80) -> float:
    """``z(1 - alpha/2) + z(power)``, the programme's critical multiple of a standard error."""
    return float(stats.norm.ppf(1.0 - alpha / 2.0) + stats.norm.ppf(power))


def blinded_mean_se(
    series: np.ndarray, *, mean_block: int, draws: int = 2_000, seed: int = 0
) -> np.ndarray:
    """Stationary-block-bootstrap standard error of the mean of each column, demeaned first.

    ``series`` is sessions x columns, every column on the same sessions. Each column is
    demeaned before any resampling, so nothing but second moments leaves this function;
    one index path per draw is shared by every column, so paired columns stay paired.
    Refuses a non-finite input: a NaN is never silently dropped from a power calculation.
    """
    a = np.asarray(series, dtype=float)
    if a.ndim == 1:
        a = a[:, None]
    if not np.isfinite(a).all():
        raise ValueError("a column holds a non-finite value")
    a = a - a.mean(axis=0)
    if not (np.abs(a.mean(axis=0)) < BLIND_TOLERANCE).all():
        raise RuntimeError("blinding failed: a mean reached the power calculation")
    rng = np.random.default_rng(seed)
    n = len(a)
    means = np.empty((draws, a.shape[1]))
    for i in range(draws):
        means[i] = a[stationary_indices(n, mean_block, rng)].mean(axis=0)
    return means.std(axis=0, ddof=1)


def holm_reject(pvalues: Sequence[float], alpha: float = 0.05) -> np.ndarray:
    """Holm's step-down rejections at family level ``alpha``; a NaN p-value is never rejected."""
    p = np.asarray(pvalues, dtype=float)
    finite = np.isfinite(p)
    order = np.argsort(np.where(finite, p, np.inf))
    m = int(finite.sum())
    reject = np.zeros(len(p), dtype=bool)
    for rank, i in enumerate(order[:m]):
        if p[i] <= alpha / (m - rank):
            reject[i] = True
        else:
            break
    return reject


def participation_ratio(corr: np.ndarray) -> float:
    """Effective dimension ``(sum l)^2 / sum l^2`` of a correlation matrix's eigenvalues."""
    eig = np.linalg.eigvalsh(np.asarray(corr, dtype=float))
    return float(eig.sum() ** 2 / (eig**2).sum())


def fold_bounds(n: int, folds: int = 5) -> list[tuple[int, int]]:
    """``folds`` contiguous slices ``[start, stop)`` of ``range(n)``, as equal as possible."""
    if folds < 1 or n < folds:
        raise ValueError("need at least one session per fold")
    edges = np.linspace(0, n, folds + 1).round().astype(int)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(folds)]


def verdict(delta: float, mde: float, t: float, pct: float, controls: Sequence[float],
            folds_positive: int, *, folds_needed: int = 3) -> str:
    """The decision of `scripts/run_crisis_coupling.py`, with a walk-forward condition added.

    ``delta`` is signed so that a positive value means the state helps. USEFUL needs
    all of: delta >= MDE, a HAC t of the same sign, delta at or above the 95th
    percentile of the rotation placebo, delta above every control arm, and a positive
    delta in at least ``folds_needed`` of the folds.
    """
    values = [delta, mde, t, pct, *controls]
    if not all(np.isfinite(values)):
        return "NOT FINITE — no verdict"
    if delta >= mde:
        if (np.sign(t) == np.sign(delta) and pct >= 0.95 and all(delta > c for c in controls)
                and folds_positive >= folds_needed):
            return "USEFUL — all conditions hold"
        return "NOT SHOWN — above the MDE, fails the t, the placebo, a control or the folds"
    if delta > 0:
        return "UNDERPOWERED — positive, below the MDE, never useful"
    if delta > -mde:
        return "NOT USEFUL — the state does not improve on the arm it is added to"
    if np.sign(t) == np.sign(delta) and pct <= 0.05:
        return "HARMFUL — adding the state measurably worsens the arm"
    return "NOT USEFUL — negative beyond the MDE, not confirmed by the t or the placebo"


# ---------------------------------------------------------------------------
# the books
# ---------------------------------------------------------------------------
def annualised_vol(variance_forecast: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    """A ``horizon``-session variance forecast as an annualised volatility."""
    return np.sqrt(variance_forecast * PERIODS / horizon)


def trend_raw_weights(
    prices: pd.DataFrame, sigma: pd.DataFrame, *, lookback: int = 252, skip: int = 21,
    target: float = VOL_TARGET, cap: float = MAX_LEVERAGE,
) -> pd.DataFrame:
    """`vehicle.unscaled_weights` with a forecast volatility in place of sigma_63.

    The sign of the 12-minus-1 trend times ``min(target / sigma, cap)``, where ``sigma``
    is the annualised volatility forecast formed at the close of ``t``; the whole
    product is lagged one session. An instrument without a forecast holds nothing.
    """
    signal = tsmom_signal(prices, lookback=lookback, skip=skip)
    scale = (target / sigma.reindex_like(prices).replace(0.0, np.nan)).clip(upper=cap)
    return (signal * scale).shift(1).fillna(0.0)


def risk_parity_raw_weights(sigma: pd.DataFrame) -> pd.DataFrame:
    """Long-only inverse-volatility weights summing to one, lagged one session."""
    inverse = 1.0 / sigma.replace(0.0, np.nan)
    return inverse.div(inverse.sum(axis=1), axis=0).shift(1).fillna(0.0)


def targeted_book(
    raw: pd.DataFrame, returns: pd.DataFrame, *, window: int = 63,
    target: float = VOL_TARGET, cap: float = MAX_LEVERAGE,
) -> tuple[pd.Series, pd.DataFrame]:
    """Hold ``raw`` at the programme's portfolio volatility target (M3).

    The multiplier is `vehicle.portfolio_scalar`: ``target / sigma_63`` of the unscaled
    book's realised return, lagged one session, capped. Returns the gross book return
    and the weights actually held. A missing instrument return counts as zero, as in
    `vehicle.vol_targeted_book`.
    """
    unscaled = (raw * returns).sum(axis=1)
    multiplier, _ = portfolio_scalar(unscaled, window=window, target=target, cap=cap)
    weights = raw.mul(multiplier, axis=0)
    return (weights * returns).sum(axis=1).rename("book"), weights

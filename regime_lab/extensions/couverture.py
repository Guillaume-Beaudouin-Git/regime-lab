"""Which hedge, not only when: the stock-bond correlation regime.

The sign of the recent correlation between equities and government bonds has changed
over the decades: positive through the 1990s and again from 2021, negative from about
2000 to 2020. A bond hedge only works in the second world. This module builds:

- the regime label: sign of the 63-session correlation between equity and bond
  returns, with a hysteresis so that the label flips only after the other sign has
  held for ``k`` sessions (`correlation_regime`);
- the defensive books it is judged on: equities plus a defensive pocket whose leg is
  chosen by a label, and the switch "equities in calm, defensive leg in stress"
  (`held_weights`, `book`);
- the variance forecast it is judged on: log forward realised variance of a
  portfolio, regressed on the label beyond volatility and the VIX, in sample with a
  HAC t, out of sample with expanding refits, and a blinded minimum detectable
  effect (`incremental_fit`, `out_of_sample_gain`, `regression_mde`).

Nothing here reads a state, a price file or a result. See `docs/PRESPEC_COUVERTURE.md`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from regime_lab.selection.protocol import mde_z

PERIODS = 252
WINDOW = 63
HYSTERESIS = 21
HORIZON = 21
POSITIVE = 1.0
NEGATIVE = 0.0


# ---------------------------------------------------------------------------
# the label
# ---------------------------------------------------------------------------
def rolling_correlation(a: pd.Series, b: pd.Series, *, window: int = WINDOW) -> pd.Series:
    """Correlation of ``a`` and ``b`` over the last ``window`` sessions where both exist.

    Sessions where either series is missing are dropped before the window is counted.
    Aligning first and rolling after, as the descriptive note did, blanks the 63
    correlations that follow every bond-market holiday; dropping first costs one
    observation per holiday instead.
    """
    both = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    return both["a"].rolling(window).corr(both["b"]).dropna().rename("correlation")


def hysteresis(flag: pd.Series, k: int) -> pd.Series:
    """0/1 label that flips only after ``k`` consecutive sessions on the other side.

    Causal: the label on a session depends on that session and the ones before. It
    starts on the first value of ``flag``.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    values = flag.astype(int).to_numpy()
    out = np.empty(len(values), dtype=float)
    if len(values) == 0:
        return pd.Series(out, index=flag.index)
    current, run = values[0], 0
    for i, v in enumerate(values):
        run = run + 1 if v != current else 0
        if run >= k:
            current, run = v, 0
        out[i] = current
    return pd.Series(out, index=flag.index)


def correlation_regime(
    equity: pd.Series, bond: pd.Series, *, window: int = WINDOW, k: int = HYSTERESIS
) -> pd.Series:
    """``POSITIVE`` (1) when the equity-bond correlation regime is positive, else 0.

    ``equity`` and ``bond`` are returns of the same sessions; for a bond the return
    is taken as the negative of the yield change, so a positive label means bonds
    fall when equities fall, the world in which they do not hedge.
    """
    corr = rolling_correlation(equity, bond, window=window)
    return hysteresis(corr > 0, k).rename("correlation_regime")


def asof(labels: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """The last label stamped at or before d, for every session d of ``index``."""
    known = labels.dropna().sort_index()
    return known.reindex(known.index.union(index)).ffill().reindex(index)


def vix_rule(vix: pd.Series, *, min_periods: int = 252) -> pd.Series:
    """One when the VIX is below its expanding median, zero above. Causal.

    Same convention as the programme's volatility rules: the higher label is calm.
    """
    level = vix.expanding(min_periods=min_periods).median()
    return (vix < level).astype(float).where(level.notna()).rename("vix_rule")


def transitions_per_year(label: pd.Series) -> float:
    """Changes of label per calendar year over the label's own span."""
    label = label.dropna()
    if len(label) < 2:
        return float("nan")
    changes = int((label != label.shift()).sum()) - 1
    years = (label.index[-1] - label.index[0]).days / 365.25
    return changes / years


# ---------------------------------------------------------------------------
# the defensive books
# ---------------------------------------------------------------------------
def held_weights(
    w_eq: pd.Series,
    w_bond: pd.Series,
    w_gold: pd.Series,
    bond_share: pd.Series | float,
    *,
    equity_on: pd.Series | float,
    pocket_on: pd.Series | float,
) -> pd.DataFrame:
    """Weights held on each session by a book of equities and a defensive pocket.

    ``EQ = equity_on * w_eq``, ``BOND = pocket_on * bond_share * w_bond`` and
    ``GOLD = pocket_on * (1 - bond_share) * w_gold``. Each ``w`` is a leg's own
    volatility-targeted weight, known the session before; ``bond_share`` is 1 for a
    bond pocket, 0 for gold, 0.5 for the fixed mix, or ``1 - label`` for a pocket
    chosen by a label (label 0 -> bonds). A permanent pocket has
    ``equity_on = pocket_on = 0.5``; the switch has ``equity_on = 1 - stress`` and
    ``pocket_on = stress``.
    """
    return pd.DataFrame({
        "EQ": equity_on * w_eq,
        "BOND": pocket_on * bond_share * w_bond,
        "GOLD": pocket_on * (1.0 - bond_share) * w_gold,
    })


def book(returns: pd.DataFrame, held: pd.DataFrame) -> tuple[pd.Series, float]:
    """Daily return of ``held`` on ``returns`` and its annual turnover.

    A missing weight or return is never read as zero: the session's book return is
    NaN, and a caller's finiteness guard then refuses the verdict.
    """
    r = (held * returns[held.columns]).sum(axis=1, skipna=False)
    turnover = float(held.diff().abs().sum(axis=1, skipna=False).mean() * PERIODS)
    return r.rename("book"), turnover


# ---------------------------------------------------------------------------
# the variance forecast
# ---------------------------------------------------------------------------
def forward_log_variance(r: pd.Series, *, horizon: int = HORIZON) -> pd.Series:
    """``log sum_{j=t+1..t+horizon} r_j^2`` on session t: the risk still to come."""
    fwd = (r**2).rolling(horizon).sum().shift(-horizon)
    return np.log(fwd.where(fwd > 0)).rename("forward_log_variance")


def trailing_log_variance(
    r: pd.Series, *, window: int, horizon: int = HORIZON
) -> pd.Series:
    """Log of the mean squared return over the last ``window`` sessions, per ``horizon``.

    Known at the close of t; on the same scale as `forward_log_variance`.
    """
    level = (r**2).rolling(window).mean() * horizon
    return np.log(level.where(level > 0)).rename(f"log_rv{window}")


def expanding_rank(x: pd.Series, *, min_periods: int = 252) -> pd.Series:
    """Percentile rank of x_t among x_1..x_t: the causal volatility rank."""
    return x.expanding(min_periods=min_periods).rank(pct=True).rename("rank")


def _with_constant(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    return np.column_stack([np.ones(len(x)), x])


def incremental_r2(y: np.ndarray, controls: np.ndarray, regressor: np.ndarray) -> float:
    """R2 gained by adding ``regressor`` to an OLS of ``y`` on a constant and ``controls``."""
    y = np.asarray(y, dtype=float)
    base = _with_constant(controls)
    full = np.column_stack([base, np.asarray(regressor, dtype=float)])
    tss = float(((y - y.mean()) ** 2).sum())
    out = []
    for x in (base, full):
        coef, *_ = np.linalg.lstsq(x, y, rcond=None)
        out.append(1.0 - float(((y - x @ coef) ** 2).sum()) / tss)
    return out[1] - out[0]


def incremental_fit(
    y: np.ndarray, controls: np.ndarray, regressor: np.ndarray, *, lags: int
) -> dict[str, float]:
    """Incremental R2, coefficient and HAC t of ``regressor`` beyond ``controls``."""
    y = np.asarray(y, dtype=float)
    base_x = _with_constant(controls)
    base = sm.OLS(y, base_x).fit()
    full = sm.OLS(y, np.column_stack([base_x, np.asarray(regressor, dtype=float)])).fit(
        cov_type="HAC", cov_kwds={"maxlags": lags})
    k = base_x.shape[1]
    return {"incremental": float(full.rsquared - base.rsquared), "coef": float(full.params[k]),
            "t": float(full.tvalues[k]), "r2_base": float(base.rsquared), "n": int(len(y))}


def regression_mde(
    y: np.ndarray,
    controls: np.ndarray,
    regressor: np.ndarray,
    *,
    lags: int,
    alpha: float,
    power: float = 0.80,
) -> dict[str, float]:
    """Blinded minimum detectable coefficient and incremental R2 of ``regressor``.

    Frisch-Waugh: with ``l`` the residual of the regressor on the controls and ``v``
    the residual of the full regression, the HAC standard error of the coefficient
    is ``sqrt(n S) / sum(l^2)``, ``S`` the long-run variance of ``u = l v``. Adding any
    multiple of the regressor to ``y`` leaves ``v`` unchanged, so **the returned
    numbers do not depend on the effect at all**; the coefficient itself is never
    returned. The detectable coefficient is the standard error times
    ``z(1 - alpha/2) + z(power)``; its incremental R2 is
    ``beta^2 sum(l^2) / sum((y - ybar)^2)``, with ``y`` taken net of the fitted
    effect so that this too is blind.
    """
    y = np.asarray(y, dtype=float)
    x = _with_constant(controls)
    reg = np.asarray(regressor, dtype=float)
    lres = reg - x @ np.linalg.lstsq(x, reg, rcond=None)[0]
    e = y - x @ np.linalg.lstsq(x, y, rcond=None)[0]
    ssl = float((lres**2).sum())
    v = e - (float(lres @ e) / ssl) * lres
    u = lres * v
    fit = sm.OLS(u, np.ones(len(u))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    se_beta = len(u) * float(fit.bse[0]) / ssl
    beta = mde_z(alpha, power) * se_beta
    y_null = y - (float(lres @ e) / ssl) * reg
    tss = float(((y_null - y_null.mean()) ** 2).sum())
    return {"se": se_beta, "beta": beta, "incremental_r2": beta**2 * ssl / tss}


def out_of_sample_forecasts(
    y: np.ndarray,
    controls: np.ndarray,
    regressor: np.ndarray,
    *,
    min_train: int,
    refit: int,
    horizon: int = HORIZON,
) -> tuple[np.ndarray, np.ndarray]:
    """Base and full forecasts of ``y`` with expanding refits; NaN before ``min_train``.

    At each refit origin ``s`` both models are fitted on rows ``0..s - horizon``, the
    only rows whose ``horizon``-session target is fully observed at ``s``, and they
    forecast rows ``s..s + refit - 1``.
    """
    y = np.asarray(y, dtype=float)
    base_x = _with_constant(controls)
    full_x = np.column_stack([base_x, np.asarray(regressor, dtype=float)])
    n = len(y)
    pred_b = np.full(n, np.nan)
    pred_f = np.full(n, np.nan)
    for start in range(min_train, n, refit):
        train = slice(0, start - horizon + 1)
        block = slice(start, min(start + refit, n))
        cb = np.linalg.lstsq(base_x[train], y[train], rcond=None)[0]
        cf = np.linalg.lstsq(full_x[train], y[train], rcond=None)[0]
        pred_b[block] = base_x[block] @ cb
        pred_f[block] = full_x[block] @ cf
    return pred_b, pred_f


def out_of_sample_gain(
    y: np.ndarray,
    controls: np.ndarray,
    regressor: np.ndarray,
    *,
    min_train: int,
    refit: int,
    horizon: int = HORIZON,
    lags: int = HORIZON,
) -> dict[str, float]:
    """Out-of-sample R2 gain of adding ``regressor``, with expanding refits.

    The forecasts are `out_of_sample_forecasts`. The gain is ``1 - SSE_full /
    SSE_base`` on the forecast rows. Clark and West's adjusted statistic, with a HAC
    t, tests it: the full model nests the base one, so the raw loss difference is
    biased against it.
    """
    y = np.asarray(y, dtype=float)
    pred_b, pred_f = out_of_sample_forecasts(y, controls, regressor, min_train=min_train,
                                             refit=refit, horizon=horizon)
    mask = np.isfinite(pred_b)
    if mask.sum() < 30:
        return {"gain": float("nan"), "cw_t": float("nan"), "n": int(mask.sum())}
    yb, pb, pf = y[mask], pred_b[mask], pred_f[mask]
    sse_b = float(((yb - pb) ** 2).sum())
    sse_f = float(((yb - pf) ** 2).sum())
    cw = (yb - pb) ** 2 - ((yb - pf) ** 2 - (pb - pf) ** 2)
    fit = sm.OLS(cw, np.ones(len(cw))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return {"gain": 1.0 - sse_f / sse_b, "cw_t": float(fit.tvalues[0]), "n": int(mask.sum())}


def participation_ratio(corr: np.ndarray) -> float:
    """Effective dimension of a correlation matrix, ``(sum eig)^2 / sum eig^2``."""
    eig = np.linalg.eigvalsh(np.asarray(corr, dtype=float))
    return float(eig.sum() ** 2 / (eig**2).sum())

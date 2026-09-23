"""Objects whose return depends on variance or on crises, and their coupling to a state.

The programme's classifier carries information about forward *variance* and none
about forward *mean* (`docs/RESULTS_FINAL.md`). Every object it was coupled to until
now had a payoff roughly linear in the underlying, on which variance information can
only resize the position. The objects built here have a payoff that is itself a
function of variance, or that is known to crash in one kind of market state:

- a short one-month variance swap on the S&P 500, synthetic, laddered and marked to
  market with the VIX (`variance_swap_ladder`);
- a short position in one-month constant-maturity VIX futures, rebuilt from the
  exchange's daily settlements (`vix_futures_constant_maturity`);
- the momentum factor and a buy-write index, which only need the sizing and coupling
  helpers below.

Nothing here reads a state or a result. See `docs/PRESPEC_CRISE.md`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PERIODS = 252
TENOR = 21
REFERENCE_VOL = 0.20
VOL_TARGET = 0.10
VOL_WINDOW = 63
MAX_LEVERAGE = 3.0
STRESS = 0.0
MULTIPLIER = {"stop": 0.0, "half": 0.5}


def monthly_variance(vix: pd.Series, *, tenor: int = TENOR) -> pd.Series:
    """Implied variance over the next ``tenor`` sessions, from the VIX in points."""
    return (vix / 100.0) ** 2 * tenor / PERIODS


def variance_swap_ladder(
    log_returns: pd.Series,
    vix: pd.Series,
    *,
    tenor: int = TENOR,
    reference_vol: float = REFERENCE_VOL,
) -> pd.Series:
    """Daily return of a short one-month variance swap position, laddered.

    One tranche is sold at every close ``i``, with strike ``K_i`` equal to the implied
    variance over the next ``tenor`` sessions (``monthly_variance``), and it receives
    the sum of the next ``tenor`` squared log returns. Each tranche is marked to
    market daily, with the implied variance of its remaining life approximated by the
    current VIX (a flat term structure):

        V_s = K_i - sum_{j=i+1..s} r_j^2 - (i + tenor - s) / tenor * K_s

    so ``V_i = 0`` and ``V_{i+tenor} = K_i - RV_i``: whatever the approximation, the
    daily changes of one tranche sum exactly to its final payoff. A ladder holds
    ``tenor`` tranches of equal variance notional; summing their daily changes gives a
    return that depends only on ``K_{s-1}``, ``K_s`` and ``r_s``:

        R_s = c * ( K_{s-1} / tenor - r_s^2 - (tenor - 1) / (2 tenor) * (K_s - K_{s-1}) )

    that is: the strike accrues, realised variance is paid, and a rise in implied
    variance costs the ladder's average remaining vega. ``c`` fixes the units: one
    unit of the position is the variance notional at which a strike of
    ``reference_vol`` is worth 100% of capital. A swap costs nothing to enter, so the
    return is an excess return; its scale is irrelevant once the position is
    volatility-targeted.
    """
    strike = monthly_variance(vix.reindex(log_returns.index), tenor=tenor)
    scale = 1.0 / monthly_variance(pd.Series([reference_vol * 100.0]), tenor=tenor).iloc[0]
    previous = strike.shift(1)
    vega = (tenor - 1) / (2.0 * tenor)
    out = scale * (previous / tenor - log_returns**2 - vega * (strike - previous))
    return out.rename("short_variance_ladder")


def tranche_values(
    log_returns: pd.Series, vix: pd.Series, opened: int, *, tenor: int = TENOR
) -> pd.Series:
    """Marked-to-market value of the single tranche sold at position ``opened``.

    A reference implementation for the tests: the ladder of `variance_swap_ladder` is
    the average of these tranches' daily changes, times the unit scale.
    """
    strike = monthly_variance(vix.reindex(log_returns.index), tenor=tenor).to_numpy()
    r2 = (log_returns**2).to_numpy()
    k = strike[opened]
    values = {}
    for s in range(opened, min(opened + tenor, len(r2) - 1) + 1):
        realised = r2[opened + 1:s + 1].sum()
        remaining = (opened + tenor - s) / tenor
        values[log_returns.index[s]] = k - realised - remaining * strike[s]
    return pd.Series(values)


def expiry_calendar(expiries: pd.DatetimeIndex, sessions: pd.DatetimeIndex) -> pd.DataFrame:
    """Front and second contract held over each session, with the roll weights.

    For a close ``t``: the front contract is the first to expire strictly after
    ``t``, the second the one after it. The roll period runs from the previous
    expiry (exclusive) to the front's (inclusive), counted in sessions; the front
    weight is the share of that period still to run after ``t``. It falls from 1 to
    ``1 / period`` and the position rolls into the second contract a little every
    session, the construction of the S&P 500 VIX Short-Term Futures Index.
    """
    expiries = pd.DatetimeIndex(sorted(expiries.unique()))
    rows = []
    positions = pd.Series(np.arange(len(sessions)), index=sessions)
    for t in sessions:
        after = expiries[expiries > t]
        before = expiries[expiries <= t]
        if len(after) < 2 or len(before) == 0:
            continue
        front, second, previous = after[0], after[1], before[-1]
        period = int(((sessions > previous) & (sessions <= front)).sum())
        remaining = int(((sessions > t) & (sessions <= front)).sum())
        if period == 0:
            continue
        rows.append({"session": t, "front": front, "second": second,
                     "w_front": remaining / period, "position": positions[t]})
    return pd.DataFrame(rows).set_index("session")


def vix_futures_constant_maturity(
    settles: pd.DataFrame, sessions: pd.DatetimeIndex
) -> pd.Series:
    """Daily return of a long one-month constant-maturity VIX futures position.

    ``settles`` is indexed by session with one column per contract, named by its
    final settlement date. The weights are fixed at the close of ``t`` and earn the
    settle-to-settle change of the same two contracts on the next session; on the
    front's expiry day its settle is the final settlement value. A session whose
    contracts lack a settle on either day returns NaN, never zero.
    """
    settles = settles.reindex(sessions)
    settles.columns = pd.DatetimeIndex(settles.columns)
    calendar = expiry_calendar(pd.DatetimeIndex(settles.columns), sessions)
    out = pd.Series(np.nan, index=sessions, name="vix_cm_long")
    lookup = {c: settles[c].to_numpy() for c in settles.columns}
    for _, row in calendar.iterrows():
        i = int(row["position"])
        if i + 1 >= len(sessions):
            continue
        f1, f2 = lookup[row["front"]], lookup[row["second"]]
        w1 = float(row["w_front"])
        before = w1 * f1[i] + (1.0 - w1) * f2[i]
        after = w1 * f1[i + 1] + (1.0 - w1) * f2[i + 1]
        out.iloc[i + 1] = after / before - 1.0
    return out


def vol_target_weight(
    unit: pd.Series,
    *,
    target: float = VOL_TARGET,
    window: int = VOL_WINDOW,
    cap: float = MAX_LEVERAGE,
) -> pd.Series:
    """Weight held over each session: ``min(target / sigma, cap)``.

    ``sigma`` is the annualised realised volatility of ``unit`` over the ``window``
    sessions that end the session before: signal at T-1, position at T.
    """
    sigma = unit.rolling(window).std() * np.sqrt(PERIODS)
    return (target / sigma.replace(0.0, np.nan)).clip(upper=cap).shift(1).rename("weight")


def lagged_state(labels: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """The last label stamped at or before d-1, for every session d of ``index``."""
    known = labels.dropna().sort_index()
    asof = known.reindex(known.index.union(index)).ffill().reindex(index)
    return asof.shift(1)


def couple(weight: pd.Series, state: pd.Series, mode: str) -> pd.Series:
    """Multiply the weight by the coupling's multiplier on stress sessions."""
    if mode not in MULTIPLIER:
        raise ValueError(mode)
    stress = state.reindex(weight.index).eq(STRESS)
    return weight.where(~stress, weight * MULTIPLIER[mode])


def volatility_tail_rule(
    returns: pd.Series, *, quantile: float = 0.80, window: int = 21, min_periods: int = 252
) -> pd.Series:
    """One when realised volatility is below its expanding ``quantile``, zero above.

    The one-line rule at a stress share close to a fitted classifier's, where the
    median rule of `evaluation.predictive.volatility_quantile_placebo` calls half the
    sessions stress. Same convention: the higher label is the calm one. Causal.
    """
    realised = returns.rolling(window).std() * np.sqrt(PERIODS)
    level = realised.expanding(min_periods=min_periods).quantile(quantile)
    return (realised < level).astype(float).where(level.notna()).rename("tail_rule")


def annual_turnover(weight: pd.Series) -> float:
    """Mean absolute daily change of the weight, times 252."""
    return float(weight.diff().abs().mean() * PERIODS)

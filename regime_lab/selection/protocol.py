"""The Two Sigma portfolio protocol: how a signal library becomes one book.

Every rule here comes from `pilotage/plans_de_recherche/twosigma/PRESPEC_TWOSIGMA.md`
§4 (the selector, the control, the inherited protocol), §5 (costs), §7 (multiple
testing) and §8 control 3 (beta). That document is the DRAFT, kept unedited as the
record; the text that settles what it left open is the lock, `docs/PRESPEC_TWOSIGMA.md`,
whose §12-§13 decide. This module implements the draft's text as written and names
each place where that text left a choice open, rather than settling it silently.

**What this module does not know.** Nothing here is estimated from returns. The
tilt's ``m_ik``, the per-state covariance matrices and the partition itself are
INPUTS. The module assembles books and compares two of them; it has no opinion on
which one wins, and no function in it reads a mean off real data during the
pre-registration phase. How ``m_ik`` is estimated on training folds is a decision
for the lock, not for this file.

**The conventions, fixed once.**

- Signal weight matrices are dates x instruments and are the positions HELD on
  each session: they arrive already lagged one session by the library. The
  state-driven mixes built here are lagged one further session, because a
  partition labelled from features known at the close of t can only steer the
  book on t+1.
- A session where any input is missing yields NaN, never a silent zero or a
  partial book. The programme lost a sample start to that once (the truncation
  trap); a NaN is visible, a zero is not.
- Volatility targeting reuses `extensions.vehicle.portfolio_scalar`: 63-session
  realised volatility of the unscaled book, lagged one session, capped at 3.0.
- Costs are charged on ``|Δw|`` of the weights actually held, after the
  volatility multiplier, through `extensions.vehicle.charge_by_instrument`.
- Excess returns are taken at the INSTRUMENT level, ``r_i - rf``, so a book pays
  the cash rate on its signed net exposure only. A cross-sectional long-short book
  is close to self-financing, and subtracting ``rf`` from the book return as a
  whole would charge it for capital it does not use. This is the same "signed"
  funding reading as `scripts/run_m3_evaluation.py`.

**The power convention.** ARBITRAGE §4.3 counts three conventions in circulation.
The one this module exposes for the Two Sigma tree is the paired stationary block
bootstrap of `analysis.power`, on legs DEMEANED before any statistic, so that the
``observed`` field is zero by construction and only a standard error comes out;
the critical value is ``z(1 - α/2) + z(power)`` with α divided by the number of
primary tests. `standalone_sharpe_threshold` gives Lo's standalone convention
beside it, so the two can be printed side by side instead of being confused.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from regime_lab.analysis.power import PowerResult, minimum_detectable_sharpe_difference
from regime_lab.extensions.trend import MAX_LEVERAGE, VOL_TARGET
from regime_lab.extensions.vehicle import charge_by_instrument, portfolio_scalar

PERIODS = 252
VOL_WINDOW = 63

#: §4: the tilt strength. The sensitivities of §7 are (0.25, 1.00) and can never
#: found a PASS.
TILT_D = 0.50
TILT_SENSITIVITIES = (0.25, 1.00)

#: §5, US equities, round trip, charged on |Δw|. Priced before any test.
COST_BPS = {"realistic": 5.0, "conservative": 10.0, "stress": 20.0}

#: §4, inherited: HAC lag and the three mean block lengths, all reported.
HAC_LAGS = 6
POWER_BLOCKS = (21, 63, 126)

#: §7: six primary tests under Holm–Bonferroni. The draft declared 21 evaluations in
#: all; the lock (§7, §12.13) declares 20: the 6 primaries, K in {3, 5, 6} x 3 levels,
#: d in {0.25, 1.00} at A-1 only, and the 21-session smoothing x 3 levels.
FAMILY_ALPHA = 0.05
POWER = 0.80
PRIMARY_TESTS = 6
DECLARED_EVALUATIONS = 20

#: Ken French panels store percent; everything downstream is in decimal.
PERCENT = 100.0

#: `observed` on demeaned legs is a rounding residue near 1e-17. Anything larger
#: means a mean reached the statistic, and the calculation is refused.
BLIND_TOLERANCE = 1e-9


# ---------------------------------------------------------------------------
# the blend
# ---------------------------------------------------------------------------
def equal_mix(names: Sequence[str], index: pd.Index) -> pd.DataFrame:
    """The control's signal weights: 1/n on every signal, every session."""
    n = len(names)
    if n == 0:
        raise ValueError("a mix needs at least one signal")
    return pd.DataFrame(1.0 / n, index=index, columns=list(names))


def blend(
    signals: Mapping[str, pd.DataFrame], mix: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Instrument weights of a blend: ``Σ_s mix[t, s] · W_s[t, :]``.

    ``signals`` maps each signal name to its dates x instruments weight matrix, as
    held (already lagged). ``mix`` is dates x signal names with rows summing to one;
    ``None`` is the equal-weight control of §4. The blend is not renormalised to a
    gross: netting across signals is part of what the blend is, and the volatility
    target decides the book's scale afterwards.

    Every matrix must share one index and one set of columns. A session where any
    signal or any mix weight is missing comes out NaN.
    """
    names = list(signals)
    if not names:
        raise ValueError("no signal to blend")
    first = signals[names[0]]
    for name in names[1:]:
        other = signals[name]
        if not other.index.equals(first.index) or not other.columns.equals(first.columns):
            raise ValueError(f"signal {name!r} is not aligned with {names[0]!r}")

    if mix is None:
        mix = equal_mix(names, first.index)
    missing = [n for n in names if n not in mix.columns]
    if missing:
        raise ValueError(f"the mix has no weight for {missing}")
    weights = mix.reindex(index=first.index, columns=names).to_numpy(float)
    rows = np.isfinite(weights).all(axis=1)
    if not np.allclose(weights[rows].sum(axis=1), 1.0, atol=1e-9):
        raise ValueError("mix rows must sum to one")

    stack = np.stack([signals[n].to_numpy(float) for n in names])
    out = np.einsum("ts,stn->tn", weights, stack)
    return pd.DataFrame(out, index=first.index, columns=first.columns)


# ---------------------------------------------------------------------------
# the selector (level A) and the risk allocation (level B)
# ---------------------------------------------------------------------------
def tilt_table(m: pd.DataFrame, *, d: float = TILT_D) -> pd.DataFrame:
    """§4's tilt, state by state: ``(1/n)(1 + d·m_ik)``, floored at zero, renormalised.

    ``m`` is states x signals and is an INPUT: how it is estimated on training folds
    is fixed at the lock, not here. The floor is applied before the renormalisation,
    which is the order of the pre-lock measurement scripts; the other order does not
    produce weights that sum to one.

    A state whose row is incomplete, or whose every weight is floored to zero,
    abstains at equal weight rather than holding nothing.
    """
    if d < 0:
        raise ValueError("the tilt strength d must be non-negative")
    n = m.shape[1]
    raw = ((1.0 + d * m) / n).clip(lower=0.0)
    total = raw.sum(axis=1)
    table = raw.div(total, axis=0)
    abstain = m.isna().any(axis=1) | ~(total > 0)
    table.loc[abstain, :] = 1.0 / n
    return table


def switch_table(m: pd.DataFrame) -> pd.DataFrame:
    """The hard switch §4 declines on power grounds: each state holds its top signal only.

    ``m`` is states x signals; the row's largest ``m_ik`` takes the whole weight, the
    first one on a tie. A state whose row is incomplete abstains at equal weight, as in
    `tilt_table`. Kept as the reference arm of the power calculation, never traded.
    """
    n = m.shape[1]
    table = pd.DataFrame(0.0, index=m.index, columns=m.columns)
    complete = m.notna().all(axis=1)
    top = m.loc[complete].to_numpy(float).argmax(axis=1)
    table.loc[complete] = np.eye(n)[top]
    table.loc[~complete, :] = 1.0 / n
    return table


def risk_parity_mix(
    cov: pd.DataFrame | np.ndarray,
    *,
    budgets: np.ndarray | None = None,
    tol: float = 1e-10,
    max_iter: int = 10_000,
) -> pd.Series | np.ndarray:
    """Equal-risk-contribution weights across signals, summing to one.

    Solves ``x_i (Σx)_i = b_i`` for ``x > 0`` by cyclical coordinate descent
    (Griveau-Billion, Richard and Roncalli, 2013), then scales to a unit sum. Every
    weight is positive: no signal is ever shorted, so each one keeps the sign
    convention it was built with. With a diagonal ``Σ`` this is inverse volatility,
    which is the "diagonal first" reading of level B.

    ``cov`` is the covariance of the signals' book returns. Returns a Series indexed
    by signal when given a DataFrame, an array otherwise.
    """
    labels = cov.columns if isinstance(cov, pd.DataFrame) else None
    s = np.asarray(cov, dtype=float)
    if s.ndim != 2 or s.shape[0] != s.shape[1]:
        raise ValueError("covariance must be square")
    if not np.isfinite(s).all():
        raise ValueError("covariance must be finite")
    if not np.allclose(s, s.T, atol=1e-12 * max(1.0, float(np.abs(s).max()))):
        raise ValueError("covariance must be symmetric")
    diag = np.diag(s)
    if (diag <= 0).any():
        raise ValueError("every signal needs a positive variance")
    if np.linalg.eigvalsh(s).min() < -1e-10 * diag.sum():
        raise ValueError("covariance must be positive semi-definite")

    n = s.shape[0]
    b = np.full(n, 1.0 / n) if budgets is None else np.asarray(budgets, dtype=float)
    if b.shape != (n,) or (b <= 0).any():
        raise ValueError("budgets must be positive, one per signal")
    b = b / b.sum()

    x = 1.0 / np.sqrt(diag)
    for _ in range(max_iter):
        for i in range(n):
            c = s[i] @ x - s[i, i] * x[i]
            x[i] = (-c + np.sqrt(c * c + 4.0 * s[i, i] * b[i])) / (2.0 * s[i, i])
        contrib = x * (s @ x)
        if np.abs(contrib / contrib.sum() - b).max() < tol:
            break
    else:
        raise RuntimeError("risk parity did not converge")

    weights = x / x.sum()
    return pd.Series(weights, index=labels) if labels is not None else weights


def map_states(
    states: pd.Series,
    table: pd.DataFrame,
    *,
    fallback: pd.Series | None = None,
    lag: int = 1,
) -> pd.DataFrame:
    """Turn a state path and a per-state mix table into dated signal weights.

    ``states`` is stamped at the close on which each label is known; the mix held on
    session t is the row of the state known at t - ``lag``. ``lag=1`` is the T-1
    rule and is the default; the pre-lock turnover measurement used ``lag=0``, and
    passing it here is how that figure is reproduced, never how a book is traded.

    A session whose state is missing, or absent from ``table``, holds ``fallback`` —
    equal weight when none is given. At level B the fallback should be the pooled
    covariance's mix, which is the arm that level is compared against. No session
    is dropped for want of a conditional estimate.

    The lag is positional, so ``states`` must already sit on the calendar the book
    trades on, sorted and without duplicates: a state path left on a calendar with
    extra dates (the context features carry exchange holidays) would lag by the wrong
    number of sessions. Map it onto the trading sessions first.
    """
    if not (states.index.is_monotonic_increasing and states.index.is_unique):
        raise ValueError("states must be on a sorted calendar without duplicates")
    if not np.allclose(table.sum(axis=1).to_numpy(float), 1.0, atol=1e-9):
        raise ValueError("table rows must sum to one")
    names = list(table.columns)
    if fallback is None:
        fallback = pd.Series(1.0 / len(names), index=names)
    fallback = fallback.reindex(names)
    if fallback.isna().any() or not np.isclose(fallback.sum(), 1.0, atol=1e-9):
        raise ValueError("the fallback must weight every signal and sum to one")

    shifted = states.shift(lag) if lag else states
    codes = table.index.get_indexer(shifted.to_numpy())
    values = np.vstack([table.to_numpy(float), fallback.to_numpy(float)[None, :]])
    out = values[np.where(codes >= 0, codes, len(table))]
    return pd.DataFrame(out, index=states.index, columns=names)


def tilt_mix(
    states: pd.Series, m: pd.DataFrame, *, d: float = TILT_D, lag: int = 1
) -> pd.DataFrame:
    """The level-A selector's signal weights on each session: `tilt_table` by state."""
    return map_states(states, tilt_table(m, d=d), lag=lag)


def match_gross(weights: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Rescale each session of ``weights`` to the gross ``Σ|w|`` of ``reference``.

    §6 level B asks for the risk-parity blend "renormalised to the same gross" as
    the arm it is compared with. A row with zero gross cannot be rescaled and stays
    at zero.
    """
    own = weights.abs().sum(axis=1, min_count=1)
    target = reference.reindex(index=weights.index).abs().sum(axis=1, min_count=1)
    scale = (target / own.replace(0.0, np.nan)).where(own != 0.0, 0.0)
    return weights.mul(scale, axis=0)


# ---------------------------------------------------------------------------
# volatility target
# ---------------------------------------------------------------------------
def scale_to_target(
    book_returns: pd.Series,
    *,
    target: float = VOL_TARGET,
    window: int = VOL_WINDOW,
    cap: float = MAX_LEVERAGE,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Daily volatility targeting of a return series: (scaled, multiplier, cap_binds).

    The multiplier is `extensions.vehicle.portfolio_scalar`: ``target / σ63``,
    lagged one session, capped. The warm-up is returned as NaN rather than as the
    zero that function fills, so that a statistic cannot be diluted by sessions on
    which the book did not yet exist.
    """
    multiplier, cap_binds = portfolio_scalar(book_returns, window=window, target=target, cap=cap)
    defined = book_returns.rolling(window).count().shift(1) >= window
    scaled = (book_returns * multiplier).where(defined)
    return scaled.rename(book_returns.name), multiplier.where(defined), cap_binds.where(defined)


@dataclass(frozen=True)
class TargetedBook:
    """A blend held at the volatility target, gross of cost."""

    returns: pd.Series
    weights: pd.DataFrame
    multiplier: pd.Series
    cap_binds: pd.Series


def target_volatility(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    *,
    target: float = VOL_TARGET,
    window: int = VOL_WINDOW,
    cap: float = MAX_LEVERAGE,
) -> TargetedBook:
    """Hold instrument ``weights`` at the daily volatility target.

    ``weights`` are held on each session (already lagged); ``returns`` are the
    instruments' returns on that session. A session where any weight or any
    return is missing gives a NaN book return. The weights returned are the ones
    actually held, after the multiplier: they are what costs are charged on.
    """
    product = weights * returns.reindex(index=weights.index, columns=weights.columns)
    unscaled = product.sum(axis=1).where(product.notna().all(axis=1)).rename("book")
    scaled, multiplier, cap_binds = scale_to_target(unscaled, target=target, window=window, cap=cap)
    return TargetedBook(
        returns=scaled,
        weights=weights.mul(multiplier, axis=0),
        multiplier=multiplier,
        cap_binds=cap_binds,
    )


# ---------------------------------------------------------------------------
# costs and turnover
# ---------------------------------------------------------------------------
def net_of_costs(book_returns: pd.Series, weights: pd.DataFrame, *, bps: float) -> pd.Series:
    """Book return minus ``Σ|Δw| · bps / 10,000``, one rate for every industry (§5).

    Inherits `charge_by_instrument`'s treatment of an undefined weight: the change
    into or out of a NaN weight is charged zero. The first session after the
    volatility target's warm-up therefore enters the book for free. That session
    lies eleven years before the first test fold, so no paired statistic contains it;
    a book that goes undefined mid-sample would, and its book return is NaN there.
    """
    rates = pd.Series(float(bps), index=weights.columns)
    return charge_by_instrument(book_returns, weights, rates)


def annual_turnover(weights: pd.DataFrame) -> float:
    """Mean ``Σ|Δw|`` per session, times 252.

    A change counts only when every weight is defined on both sessions; the first
    session, and any session next to a missing weight, is skipped rather than counted
    as zero or as a partial sum over the legs that happen to be present. This is the
    pre-lock measurement's convention, which divides by T - 1 where
    `extensions.vehicle.turnover` divides by T. NaN when no change is defined.
    """
    diff = weights.diff().abs()
    changes = diff.sum(axis=1).where(diff.notna().all(axis=1)).iloc[1:]
    return float(changes.mean() * PERIODS) if changes.notna().any() else float("nan")


def cost_in_sharpe(turnover: float, bps: float, *, vol_target: float = VOL_TARGET) -> float:
    """Annual cost of ``turnover`` at ``bps``, in Sharpe units of a book at ``vol_target``.

    Valid only when ``turnover`` is measured on the weights of the book AS HELD at
    that target. Turnover measured on unscaled weights must first be multiplied by
    the volatility multiplier.
    """
    return turnover * bps / 10_000.0 / vol_target


def breakeven_bps(sharpe: float, turnover: float, *, vol_target: float = VOL_TARGET) -> float:
    """Round-trip bps at which ``turnover`` costs exactly ``sharpe`` (§5: 92.7)."""
    if turnover <= 0:
        return float("inf")
    return sharpe * vol_target * 10_000.0 / turnover


def turnover_kill(sharpe: float, bps: float, *, vol_target: float = VOL_TARGET) -> float:
    """Annual turnover at which ``bps`` consumes ``sharpe`` (§5: 12×/yr at 20 bp)."""
    return sharpe * vol_target * 10_000.0 / bps


# ---------------------------------------------------------------------------
# excess returns and the market
# ---------------------------------------------------------------------------
def factor_series(
    panel: pd.DataFrame,
    series_id: str,
    index: pd.DatetimeIndex,
    *,
    stamp: str = "available_at",
    fill: bool = False,
) -> pd.Series:
    """One Ken French daily series from a long PIT panel, in decimal, on ``index``.

    ``panel`` has the store's columns ``series_id``, ``period``, ``available_at`` and
    ``value`` (percent). ``stamp`` picks which date the value is placed at; ``fill``
    carries the last value forward, which is right for a rate and wrong for a return.
    """
    if stamp not in ("available_at", "period"):
        raise ValueError("stamp must be 'available_at' or 'period'")
    rows = panel.loc[panel["series_id"] == series_id]
    if rows.empty:
        raise KeyError(f"{series_id!r} is not in the panel")
    series = (
        rows.assign(
            available_at=pd.to_datetime(rows["available_at"]), period=pd.to_datetime(rows["period"])
        )
        .sort_values([stamp, "period"])
        .drop_duplicates(stamp, keep="last")
        .set_index(stamp)["value"]
        .astype(float)
        / PERCENT
    )
    series.index = series.index.astype("datetime64[ns]")
    target = pd.DatetimeIndex(index).astype("datetime64[ns]")
    if fill:
        series = series.reindex(series.index.union(target)).ffill()
    return series.reindex(target).set_axis(index).rename(series_id)


def risk_free(panel: pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series:
    """``ff_rf``, daily, stamped at ``available_at`` and never at ``period``, carried forward.

    On the current store the two stamps coincide on every ``ff_rf`` row, so the
    choice moves nothing today. It is not self-evident: `market_excess` stamps at
    ``period`` because a measurement made afterwards pairs t with t, and an excess
    return is such a measurement too. With a real publication delay, this stamp
    would subtract a stale rate. Which reading excess returns take is for the lock.
    """
    return factor_series(panel, "ff_rf", index, stamp="available_at", fill=True)


def market_excess(panel: pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series:
    """``ff_mkt-rf`` on the session it accrued, for a realised beta.

    Stamped at ``period`` on purpose: a beta pairs the book's return on t with the
    market's return on t. It is a measurement made afterwards and informs no trade,
    so publication delay does not enter it; stamping it at ``available_at`` would
    misalign the pair the day the store encodes a real delay. No fill: a missing
    market return is missing, not yesterday's.
    """
    return factor_series(panel, "ff_mkt-rf", index, stamp="period", fill=False)


def excess_returns(returns: pd.DataFrame, rf: pd.Series) -> pd.DataFrame:
    """Instrument returns in excess of the daily risk-free rate, ``r_i - rf``."""
    return returns.sub(rf.reindex(returns.index), axis=0)


# ---------------------------------------------------------------------------
# comparison
# ---------------------------------------------------------------------------
def realised_beta(book_returns: pd.Series, market: pd.Series) -> float:
    """OLS slope of the book on the market over their common sessions (§8 control 3)."""
    both = pd.concat([book_returns, market], axis=1, join="inner").dropna()
    if len(both) < 2:
        return float("nan")
    x = both.iloc[:, 1].to_numpy(float)
    y = both.iloc[:, 0].to_numpy(float)
    vx = x.var(ddof=1)
    if not vx > 0:
        return float("nan")
    return float(np.cov(y, x, ddof=1)[0, 1] / vx)


def placebo_percentile(value: float, null: np.ndarray) -> float:
    """Share of placebo draws below ``value``, ties counted half.

    NaN when ``value`` or ANY draw is not finite: the lock (§13.4) makes such a lock
    UNDECIDABLE, and no draw is dropped, replaced or redrawn. Dropping a broken draw
    would condition the null on the draws that happen to work.
    """
    draws = np.asarray(null, dtype=float)
    if not np.isfinite(value) or draws.size == 0 or not np.isfinite(draws).all():
        return float("nan")
    return float((draws < value).mean() + 0.5 * (draws == value).mean())


def placebo_p_value(value: float, null: np.ndarray) -> float:
    """One-sided placebo p-value, ``(1 + #{draws >= value}) / (1 + n)``.

    Ties count against the real partition. NaN when ``value`` or any draw is not
    finite, for the reason given in :func:`placebo_percentile`.
    """
    draws = np.asarray(null, dtype=float)
    if not np.isfinite(value) or draws.size == 0 or not np.isfinite(draws).all():
        return float("nan")
    return float((1 + (draws >= value).sum()) / (1 + draws.size))


def paired_hac_t(a: pd.Series, b: pd.Series, *, lags: int = HAC_LAGS) -> float:
    """HAC t-statistic of the mean of ``a - b`` on their common sessions.

    Newey–West, Bartlett kernel, ``lags`` = 6 by the inherited rule, the same fit as
    the programme's ``hac_t`` (no small-sample correction). With both legs held at
    the same realised volatility the mean difference is the Sharpe difference times
    that volatility, so this is the paired test of §6 A-1 up to the scale; it is not
    a delta-method test on the Sharpe ratios themselves. The equivalence fails where
    the leverage cap binds, since a capped leg runs below the target.
    """
    diff = (a - b).dropna()
    values = diff.to_numpy(float)
    if len(values) < 30 or not values.std(ddof=1) > 0:
        return float("nan")
    fit = sm.OLS(values, np.ones(len(values))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(fit.tvalues[0])


# ---------------------------------------------------------------------------
# the power convention
# ---------------------------------------------------------------------------
def mde_z(alpha: float = FAMILY_ALPHA, power: float = POWER) -> float:
    """Critical multiple of the standard error: ``z(1 - α/2) + z(power)``."""
    return float(stats.norm.ppf(1.0 - alpha / 2.0) + stats.norm.ppf(power))


def blinded_mde(
    a: np.ndarray,
    b: np.ndarray,
    *,
    mean_block: int,
    draws: int = 2_000,
    alpha: float = FAMILY_ALPHA,
    power: float = POWER,
    seed: int = 0,
) -> PowerResult:
    """Minimum detectable paired Sharpe difference, with both legs demeaned first.

    Sessions where either leg is missing are dropped from both. The means are
    removed before `analysis.power` sees the legs, so its ``observed`` field is zero
    by construction and only the bootstrap standard error — a function of second
    moments alone — leaves this function. If ``observed`` is not zero to within
    rounding, a mean reached the statistic and the call is refused.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("legs must be the same length")
    keep = np.isfinite(a) & np.isfinite(b)
    a, b = a[keep], b[keep]
    if len(a) < 2 or not (np.ptp(a) > 0 and np.ptp(b) > 0):
        raise ValueError("each leg needs two sessions and a positive variance")
    a = a - a.mean()
    b = b - b.mean()
    result = minimum_detectable_sharpe_difference(
        a, b, mean_block=mean_block, draws=draws, alpha=alpha, power=power, seed=seed
    )
    if not abs(result.observed) < BLIND_TOLERANCE:
        raise RuntimeError("blinding failed: a mean reached the power calculation")
    return result


def mde_at(result: PowerResult, alpha: float, power: float | None = None) -> float:
    """The same standard error, read at another α: what `analysis.power` gives on the same draws."""
    return float(result.se * mde_z(alpha, result.power if power is None else power))


def standalone_sharpe_threshold(
    years: float, *, alpha: float = FAMILY_ALPHA, power: float = POWER
) -> float:
    """Lo's standalone convention: the smallest Sharpe separable from zero in ``years``.

    Solves ``S = z · sqrt((1 + S²/2) / Y)``, i.e. ``S = z / sqrt(Y - z²/2)``, with
    ``z = z(1 - α/2) + z(power)``. Infinite below ``z²/2`` years (3.924 at the
    defaults), where no effect of any size separates from zero. ARBITRAGE §0
    reproduces 0.638 at 23.2 years with it.
    """
    z = mde_z(alpha, power)
    room = years - z * z / 2.0
    return float(z / np.sqrt(room)) if room > 0 else float("inf")

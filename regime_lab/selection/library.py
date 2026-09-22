"""The signal library — twelve candidates on the 49-industry panel, ten retained.

`PRESPEC_TWOSIGMA.md` §4 fixes the construction: every signal is a cross-sectional
z-score over the 49 industries, clipped at ±3, gross-normalised to 1 and lagged one
session, so the position held on *t* is known at the close of *t-1*. §1 P2 measured
twelve candidates and §4 excludes two before any contact with a return:

``IND_ACCEL``
    correlates −0.912 with ``IND_MOM_12_1`` in position space. It is the 3-month
    minus 12-month return, so it is 12-1 momentum with its sign flipped plus noise:
    a construction redundancy, not a second signal.
``IND_REV_1W``
    turns over 156.7 times a year, 7.83%/yr at the 5 bp round trip of §5 — ten
    times what a weekly sector reversal can plausibly earn gross.

That leaves the ten-signal inferential library, ``LIBRARY``.

**Position space only.** Nothing here multiplies a weight by a forward return. The
diagnostics below — the mean daily cross-sectional correlation of two signals'
weight vectors, the participation ratio of its eigenvalues, turnover ``|Δw|`` — say
whether two signals take the same bet and what trading them costs, never whether
either earns anything. That is the blindness the unlocked pre-registration requires.

**Ported, not rewritten.** Every construction reproduces the scripts that produced
the disclosed figures before they were committed, down to their NaN handling: the
multi-asset signals take ``log`` of the stored price even where it is not positive,
which yields NaN (CL=F on 2020-04-20) exactly as the original did.

The eleven signals of ``TREND_CANDIDATES`` exist for one purpose: §1 P1, the
declared result that the 46-instrument universe *cannot* host this study, its
eleven signals having an effective rank of 3.86. They are not a library.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from regime_lab.config import RAW
from regime_lab.data.pit import validate
from regime_lab.data.sources.kenfrench import MISSING

#: The twelve candidates of §1 P2, in the order the disclosed matrix was printed.
INDUSTRY_CANDIDATES: tuple[str, ...] = (
    "IND_MOM_12_1", "IND_MOM_6_1", "IND_REV_1M", "IND_REV_1W", "IND_LTREV", "IND_LOWVOL",
    "IND_VOLMOM", "IND_ACCEL", "IND_SKEW", "IND_LOWBETA", "IND_LOWCORR", "IND_SEASON",
)

#: §4, excluded before contact, with the reason written where the exclusion is made.
EXCLUDED: dict[str, str] = {
    "IND_ACCEL": "construction redundancy, -0.912 against IND_MOM_12_1 in position space",
    "IND_REV_1W": "turnover 156.7x/yr, 7.83%/yr at 5 bp round trip",
}

#: The ten-signal inferential library of §4.
LIBRARY: tuple[str, ...] = tuple(n for n in INDUSTRY_CANDIDATES if n not in EXCLUDED)

#: The eleven candidates of §1 P1 on the 46-instrument universe.
TREND_CANDIDATES: tuple[str, ...] = (
    "TSMOM_12_1", "TSMOM_3M", "TSMOM_1M", "TSREV_5D", "TSREV_1D", "BRKOUT_100", "BRKOUT_20",
    "XSMOM_12_1", "XSREV_1M", "XS_LOWVOL", "XS_CARRY_PX",
)

#: §1 P1: a session enters the 46-instrument measurement when at least this many
#: instruments carry all eleven signals.
TREND_MIN_LEGS = 30

Z_CLIP = 3.0
#: Floor on a simple return before taking its log, as in the disclosed scripts.
RETURN_FLOOR = -0.95


# --------------------------------------------------------------------------- inputs


def load_panel(name: str, *, root: Path = RAW) -> pd.DataFrame:
    """One Ken French daily panel as a wide frame of decimal returns.

    Rows are sessions, columns are series ids. The store keeps percent returns with
    ``available_at == period`` (the judgement documented in ``kenfrench.py`` and
    declared as this study's PIT weakness in §3); the library's T-1 rule is applied
    later, by ``positions``. Sentinels are masked even though §3 counts none.
    """
    frame = validate(pd.read_parquet(root / "panels" / f"{name}.parquet"))
    frame = frame.loc[~frame["value"].isin(MISSING)]
    wide = frame.pivot(index="period", columns="series_id", values="value").sort_index()
    wide.columns = wide.columns.astype(str)
    wide.columns.name = None
    return wide / 100.0


def log_returns(returns: pd.DataFrame) -> pd.DataFrame:
    """``log(1 + r)`` with ``r`` floored at −95%, the disclosed scripts' convention."""
    return np.log1p(returns.clip(lower=RETURN_FLOOR))


# ----------------------------------------------------------------------- transforms


def cross_sectional_z(scores: pd.DataFrame, *, clip: float = Z_CLIP) -> pd.DataFrame:
    """Z-score each session across columns (population sd), clipped at ``±clip``.

    A session with zero dispersion has no ranking and returns NaN throughout.
    """
    mean = scores.mean(axis=1)
    sd = scores.std(axis=1, ddof=0).replace(0.0, np.nan)
    return scores.sub(mean, axis=0).div(sd, axis=0).clip(-clip, clip)


def time_series_z(
    scores: pd.DataFrame, *, window: int = 252, clip: float = Z_CLIP
) -> pd.DataFrame:
    """Scale each column by its own trailing ``window`` standard deviation, clipped.

    Not demeaned: this is the multi-asset scripts' time-series normalisation, which
    keeps the sign of the raw score.
    """
    return (scores / scores.rolling(window).std()).clip(-clip, clip)


def window_return(log_ret: pd.DataFrame, lookback: int, skip: int = 0) -> pd.DataFrame:
    """Sum of log returns over sessions ``t-lookback+1 .. t-skip``.

    ``window_return(lr, 252, 21)`` is the 12-minus-1 month return: the 231 sessions
    ending 21 sessions ago.
    """
    return log_ret.rolling(lookback - skip).sum().shift(skip)


def rolling_beta(log_ret: pd.DataFrame, market: pd.Series, *, window: int = 252) -> pd.DataFrame:
    """Trailing OLS beta of each column on ``market``: rolling cov over rolling var."""
    market = market.reindex(log_ret.index)
    return log_ret.rolling(window).cov(market).div(market.rolling(window).var(), axis=0)


def same_month_seasonality(
    log_ret: pd.DataFrame, *, years: int = 10, min_years: int = 5
) -> pd.DataFrame:
    """Heston-Sadka seasonality: the mean return of the same calendar month over the
    previous ``years`` years, at least ``min_years`` of them, stamped on every session
    of the current month.

    Only past years enter, so the value on any session of month *m* of year *y* is
    known before year *y*'s month *m* begins.
    """
    index = log_ret.index
    monthly = log_ret.groupby([index.year, index.month]).sum()
    monthly.index = monthly.index.set_names(["year", "month"])
    by_year = monthly.unstack("month")
    past = by_year.shift(1).rolling(years, min_periods=min_years).mean()
    lookup = past.stack("month", future_stack=True).reindex(columns=log_ret.columns)
    key = pd.MultiIndex.from_arrays([index.year, index.month], names=["year", "month"])
    return pd.DataFrame(lookup.reindex(key).to_numpy(), index=index, columns=log_ret.columns)


def positions(scores: pd.DataFrame, *, lag: int = 1) -> pd.DataFrame:
    """Scores to held weights: lagged ``lag`` sessions, then unit gross each session.

    The T-1 rule lives here. A session whose scores are all zero or all missing has
    no position and returns NaN, never a silent zero.
    """
    lagged = scores.shift(lag)
    gross = lagged.abs().sum(axis=1, min_count=1).replace(0.0, np.nan)
    return lagged.div(gross, axis=0)


# -------------------------------------------------------------------------- signals


def industry_scores(returns: pd.DataFrame, market: pd.Series) -> dict[str, pd.DataFrame]:
    """The twelve candidates of §1 P2 as unlagged cross-sectional z-scores.

    ``returns`` are decimal simple returns of the 49 industries; ``market`` is
    ``ff_mkt-rf`` in decimals, used only as the regressor of the beta signal.
    Pass the result through ``positions`` before anything is held.
    """
    lr = log_returns(returns)
    vol63 = lr.rolling(63).std()
    vol252 = lr.rolling(252).std()
    raw = {
        "IND_MOM_12_1": window_return(lr, 252, 21),
        "IND_MOM_6_1": window_return(lr, 126, 21),
        "IND_REV_1M": -lr.rolling(21).sum(),
        "IND_REV_1W": -lr.rolling(5).sum(),
        "IND_LTREV": -window_return(lr, 1260, 252),
        "IND_LOWVOL": -vol63,
        "IND_VOLMOM": -(vol63 - vol252),
        "IND_ACCEL": lr.rolling(63).sum() - lr.rolling(252).sum(),
        "IND_SKEW": -lr.rolling(63).skew(),
        "IND_LOWBETA": -rolling_beta(lr, market, window=252),
        # Correlation with the equal-weight industry mean, a proxy for the average
        # pairwise correlation that is linear in the number of legs.
        "IND_LOWCORR": -lr.rolling(126).corr(lr.mean(axis=1)),
        "IND_SEASON": same_month_seasonality(lr, years=10, min_years=5),
    }
    return {name: cross_sectional_z(raw[name]) for name in INDUSTRY_CANDIDATES}


def industry_library(
    returns: pd.DataFrame, market: pd.Series, names: Sequence[str] = LIBRARY
) -> dict[str, pd.DataFrame]:
    """Held weights of the named industry signals: lagged one session, unit gross.

    Defaults to the ten-signal library of §4; pass ``INDUSTRY_CANDIDATES`` for the
    twelve measured in §1 P2.
    """
    unknown = [n for n in names if n not in INDUSTRY_CANDIDATES]
    if unknown:
        raise ValueError(f"not an industry candidate: {unknown}")
    scores = industry_scores(returns, market)
    return {name: positions(scores[name]) for name in names}


def trend_universe_scores(prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The eleven candidates of §1 P1 on the 46-instrument universe, unlagged.

    Kept only to reproduce the declared non-transposability result. ``prices`` are
    stored price levels, not returns — the programme's fourth trap.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.log(prices).diff()
        vol = lr.rolling(63).std()
        trend_12_1 = window_return(lr, 252, 21) / (vol * np.sqrt(231))
        trend_1m = lr.rolling(21).sum() / (vol * np.sqrt(21))

        def donchian(window: int) -> pd.DataFrame:
            low, high = prices.rolling(window).min(), prices.rolling(window).max()
            return (prices - low) / (high - low) * 2.0 - 1.0

        scores = {
            "TSMOM_12_1": time_series_z(trend_12_1),
            "TSMOM_3M": time_series_z(lr.rolling(63).sum() / (vol * np.sqrt(63))),
            "TSMOM_1M": time_series_z(trend_1m),
            "TSREV_5D": time_series_z(-lr.rolling(5).sum() / (vol * np.sqrt(5))),
            "TSREV_1D": time_series_z(-lr / vol),
            "BRKOUT_100": donchian(100),
            "BRKOUT_20": donchian(20),
            "XSMOM_12_1": cross_sectional_z(trend_12_1),
            "XSREV_1M": cross_sectional_z(-trend_1m),
            "XS_LOWVOL": cross_sectional_z(-np.log(vol)),
            # Distance to the one-year mean price, the scripts' value proxy.
            "XS_CARRY_PX": cross_sectional_z(-np.log(prices / prices.rolling(252).mean())),
        }
    return {name: scores[name] for name in TREND_CANDIDATES}


# ---------------------------------------------------------------------- diagnostics


def _stack(weights: Mapping[str, pd.DataFrame]) -> tuple[np.ndarray, pd.Index, pd.Index]:
    if not weights:
        raise ValueError("no signal given")
    first = next(iter(weights.values()))
    index, columns = first.index, first.columns
    cube = np.stack(
        [w.reindex(index=index, columns=columns).to_numpy(float) for w in weights.values()]
    )
    return cube, index, columns


def _cell_mask(cube: np.ndarray, min_legs: int | None) -> tuple[np.ndarray, np.ndarray]:
    cells = np.isfinite(cube).all(axis=0)
    need = cube.shape[2] if min_legs is None else min_legs
    return cells, cells.sum(axis=1) >= need


def complete_sessions(
    weights: Mapping[str, pd.DataFrame], *, min_legs: int | None = None
) -> pd.DatetimeIndex:
    """Sessions on which every signal is defined on at least ``min_legs`` common legs.

    ``min_legs=None`` requires every leg — the rule for the 49-industry panel, which
    §3 verifies is fully populated. A leg counts on a session only if every signal
    has it, so the signals are always compared on the same cross-section.
    """
    cube, index, _ = _stack(weights)
    _, keep = _cell_mask(cube, min_legs)
    return pd.DatetimeIndex(index[keep])


def position_correlation(
    weights: Mapping[str, pd.DataFrame], *, min_legs: int | None = None
) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """Mean over sessions of the cross-sectional correlation between weight vectors.

    For each session in ``complete_sessions`` and each pair of signals, the Pearson
    correlation of the two weight vectors across the legs both define; then the mean
    over sessions, skipping any session on which a vector has no dispersion. No
    return enters. Returns the matrix and the sessions it was measured on.
    """
    cube, index, _ = _stack(weights)
    cells, keep = _cell_mask(cube, min_legs)
    x = np.where(cells, cube, np.nan)[:, keep]
    x = np.nan_to_num(x - np.nanmean(x, axis=2, keepdims=True), nan=0.0)
    gram = np.einsum("itn,jtn->ijt", x, x)
    norm = np.sqrt(np.einsum("iit->it", gram))
    denominator = norm[:, None, :] * norm[None, :, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        daily = np.where(denominator == 0.0, np.nan, gram / denominator)
        mean = np.nanmean(daily, axis=2)
    names = list(weights)
    return pd.DataFrame(mean, index=names, columns=names), pd.DatetimeIndex(index[keep])


def eigenvalues(correlation: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Eigenvalues of a symmetric matrix, descending, negative values floored at 0."""
    values = np.linalg.eigvalsh(np.asarray(correlation, dtype=float))
    return np.clip(values, 0.0, None)[::-1]


def effective_rank(correlation: pd.DataFrame | np.ndarray) -> float:
    """Participation ratio ``(Σλ)² / Σλ²``: k for k orthogonal signals, 1 for one bet."""
    values = eigenvalues(correlation)
    return float(values.sum() ** 2 / (values**2).sum())


def components_for_share(correlation: pd.DataFrame | np.ndarray, share: float = 0.90) -> int:
    """How many leading principal components reach ``share`` of the trace."""
    values = eigenvalues(correlation)
    cumulative = np.cumsum(values) / values.sum()
    return int(np.searchsorted(cumulative, share)) + 1


@dataclass(frozen=True)
class Geometry:
    """How many distinct bets a set of signals takes, in position space."""

    correlation: pd.DataFrame
    sessions: pd.DatetimeIndex
    eigenvalues: np.ndarray
    effective_rank: float
    components_90: int
    mean_corr: float
    mean_abs_corr: float
    max_abs_corr: float

    @property
    def years(self) -> float:
        """Calendar span of the measurement sample."""
        return (self.sessions.max() - self.sessions.min()).days / 365.25

    def pair(self, a: str, b: str) -> float:
        """The correlation of one named pair."""
        return float(self.correlation.loc[a, b])

    def strongest_pairs(self, count: int) -> list[tuple[str, str, float]]:
        """The ``count`` pairs with the largest absolute correlation."""
        names = list(self.correlation.index)
        pairs = [
            (a, b, float(self.correlation.iloc[i, j]))
            for i, a in enumerate(names)
            for j, b in enumerate(names)
            if j > i
        ]
        return sorted(pairs, key=lambda p: -abs(p[2]))[:count]


def geometry(
    weights: Mapping[str, pd.DataFrame],
    names: Sequence[str] | None = None,
    *,
    min_legs: int | None = None,
) -> Geometry:
    """Position correlation, its eigen-structure and summary statistics, in one pass.

    ``names`` restricts the measurement to a subset while keeping the sessions and
    legs the full ``weights`` mapping defines, so that a 10-signal and a 12-signal
    reading are taken on the same cross-sections.
    """
    cube, index, columns = _stack(weights)
    cells, keep = _cell_mask(cube, min_legs)
    common = pd.DataFrame(cells, index=index, columns=columns)
    chosen = list(weights) if names is None else list(names)
    subset = {
        n: weights[n].reindex(index=index, columns=columns).where(common).loc[keep]
        for n in chosen
    }
    correlation, used = position_correlation(subset, min_legs=min_legs)
    c = correlation.to_numpy()
    off = c[~np.eye(len(c), dtype=bool)]
    return Geometry(
        correlation=correlation,
        sessions=used,
        eigenvalues=eigenvalues(c),
        effective_rank=effective_rank(c),
        components_90=components_for_share(c, 0.90),
        mean_corr=float(off.mean()),
        mean_abs_corr=float(np.abs(off).mean()),
        max_abs_corr=float(np.abs(off).max()),
    )


def equal_weight_blend(
    weights: Mapping[str, pd.DataFrame], names: Sequence[str] | None = None
) -> pd.DataFrame:
    """The §4 control: the plain average of the signals' unit-gross weights.

    Not renormalised: where signals disagree the blend's gross falls below one, and
    the portfolio volatility target, not this function, restores the risk.
    """
    chosen = list(weights) if names is None else list(names)
    if not chosen:
        raise ValueError("no signal given")
    total = weights[chosen[0]]
    for name in chosen[1:]:
        total = total + weights[name]
    return total / len(chosen)


def annual_turnover(weights: pd.DataFrame, *, periods: int = 252) -> float:
    """Annualised two-sided turnover: mean session-to-session ``Σ|Δw|``, times ``periods``.

    The mean runs over the T−1 changes of the frame given; the first session, having
    no predecessor, is not counted as a zero (``vehicle.turnover`` counts it, which
    differs by a factor (T−1)/T). A non-finite weight is refused: choose the sample
    explicitly, typically ``complete_sessions``, rather than let a gap read as a trade.
    """
    values = weights.to_numpy(float)
    if len(values) < 2:
        raise ValueError("turnover needs at least two sessions")
    if not np.isfinite(values).all():
        raise ValueError("non-finite weight: restrict to complete sessions first")
    return float(np.abs(np.diff(values, axis=0)).sum(axis=1).mean() * periods)

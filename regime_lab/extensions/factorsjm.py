"""One sparse jump model per factor, after Shu and Mulvey (arXiv 2410.14841).

The programme's single classifier is a switch on one book: ordered by volatility,
changing state about once every two years, and conditioning an object of effective
dimension below four. Shu and Mulvey's follow-up to the paper replicated in
`docs/REPLICATION_SHU2024.md` does something else. Each factor gets its own two-state
model, fitted on the factor's own active return (its trend, oscillators, downside
deviation and market beta) plus a handful of market variables; the states are
labelled bull and bear by the cumulative active return earned in each, not by
volatility; and the six decisions steer an allocation across factors.

This module holds the construction and nothing that reads a result:

- the paper's features, computed from a daily active return (`active_features`) and
  from the market (`market_features`);
- the walk-forward fit of one model at one penalty (`walk_forward`): expanding window
  capped at twelve years and floored at eight, refits on a fixed schedule, online
  inference in between with the centroids held fixed;
- the paper's tuning rule (`select_penalty`): at each refit, keep the penalty whose
  single-factor long-short strategy had the best Sharpe over the six preceding years
  of its own out-of-sample path;
- the two mappings from a state to an exposure: on/off, and the paper's
  ``clip(mu_state / 5%, -1, 1)``;
- the same machinery for any other partition (`partition_means`), so a volatility
  rule or a trailing-return rule can be put through exactly the same exposure code;
- the book and the descriptive measures (participation ratio, kappa, transitions).

Conventions. Returns are daily, in decimals, already in excess of cash (a long-short
factor is self-financing, the market factor is ``Mkt - RF``). Label 1 is bull, label 0
is bear, as the programme's "higher label is the good one". A state stamped on
session ``d`` is used on session ``d + LAG``: the paper's one-day delay.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from jumpmodels.preprocess import DataClipperStd, StandardScalerPD
from jumpmodels.sparse_jump import SparseJumpModel

PERIODS = 252

#: Half-lives and windows of the paper's factor-specific features, in sessions.
WINDOWS = (8, 21, 63)
MACD_PAIRS = ((8, 21), (21, 63))
HL_RISK = 21

#: The market variables use one medium window, as in the paper.
HL_MARKET = 21
#: Half-life of the realised volatility that stands in for the VIX (see `market_features`).
HL_REALISED = 10

#: Expanding training window, floored and capped (years), and the validation window.
MIN_TRAIN_YEARS = 8
MAX_TRAIN_YEARS = 12
VALIDATION_YEARS = 6

#: Winsorisation of the features, in training standard deviations.
CLIP_SD = 3.0

#: The paper's cap on a state's expected active return, per year.
CAP = 0.05

#: A state stamped on session d is traded on session d + LAG (signal known at the
#: close of d, executed at the close of d + 1, earning the return of d + 2).
LAG = 2

BULL = 1
BEAR = 0

_EPS = 1e-12


# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------
def _ewm(x: pd.Series, halflife: int) -> pd.Series:
    return x.ewm(halflife=halflife, min_periods=halflife).mean()


def rsi(returns: pd.Series, window: int) -> pd.Series:
    """Relative strength index of a return series, 0 to 100, simple rolling means."""
    gains = returns.clip(lower=0.0).rolling(window).mean()
    losses = (-returns).clip(lower=0.0).rolling(window).mean()
    total = gains + losses
    return (100.0 * gains / total.where(total > 0)).rename(f"rsi_{window}")


def stochastic_k(returns: pd.Series, window: int) -> pd.Series:
    """Stochastic oscillator %K of the cumulative index built from ``returns``, 0 to 100."""
    index = np.log1p(returns).cumsum()
    low = index.rolling(window).min()
    high = index.rolling(window).max()
    span = high - low
    k = 100.0 * (index - low) / span.where(span > 0)
    return k.clip(0.0, 100.0).rename(f"stoch_k_{window}")


def macd(returns: pd.Series, fast: int, slow: int) -> pd.Series:
    """Fast minus slow exponential average of the cumulative log index (half-lives)."""
    index = np.log1p(returns).cumsum()
    fast_avg = index.ewm(halflife=fast, min_periods=fast).mean()
    slow_avg = index.ewm(halflife=slow, min_periods=slow).mean()
    return (fast_avg - slow_avg).rename(f"macd_{fast}_{slow}")


def downside_deviation(returns: pd.Series, halflife: int = HL_RISK) -> pd.Series:
    """Exponentially weighted root mean square of the negative returns."""
    return np.sqrt(_ewm(returns.clip(upper=0.0) ** 2, halflife))


def ewm_beta(active: pd.Series, market: pd.Series, halflife: int = HL_RISK) -> pd.Series:
    """Exponentially weighted beta of ``active`` on ``market``."""
    frame = pd.concat({"a": active, "m": market}, axis=1)
    mean_a = _ewm(frame["a"], halflife)
    mean_m = _ewm(frame["m"], halflife)
    cov = _ewm(frame["a"] * frame["m"], halflife) - mean_a * mean_m
    var = _ewm(frame["m"] ** 2, halflife) - mean_m**2
    return cov / var.where(var > _EPS)


def active_features(active: pd.Series, market: pd.Series | None = None) -> pd.DataFrame:
    """The paper's factor-specific features, computed on one daily active return.

    EWMA of the active return (half-lives 8, 21, 63), RSI and stochastic %K (windows
    8, 21, 63), MACD (8, 21) and (21, 63), the log of the downside deviation (half-life
    21) and the active market beta (half-life 21). The beta is left out when ``market``
    is None, which is the case of the market factor itself.
    """
    cols: dict[str, pd.Series] = {}
    for w in WINDOWS:
        cols[f"ret_ewm_{w}"] = _ewm(active, w)
    for w in WINDOWS:
        cols[f"rsi_{w}"] = rsi(active, w)
    for w in WINDOWS:
        cols[f"stoch_k_{w}"] = stochastic_k(active, w)
    for fast, slow in MACD_PAIRS:
        cols[f"macd_{fast}_{slow}"] = macd(active, fast, slow)
    cols["log_downside"] = np.log(downside_deviation(active) + _EPS)
    if market is not None:
        cols["beta_mkt"] = ewm_beta(active, market.reindex(active.index))
    return pd.DataFrame(cols, index=active.index)


def market_features(
    market: pd.Series, short_yield: pd.Series, long_yield: pd.Series
) -> pd.DataFrame:
    """The paper's market-environment variables, with long-history stand-ins.

    - market return, EWMA half-life 21 (as in the paper);
    - the VIX line (log, difference, EWMA 21) is taken on the market's own realised
      volatility, ``sqrt(EWMA(r^2, half-life 10))``, because the VIX starts in 1990;
    - the 2-year yield line (difference, EWMA 21) is taken on the 1-year yield, and
      the 10-year minus 2-year line on the 10-year minus 1-year spread, because the
      2-year series starts in 1976.

    Yields are in percent and are forward-filled onto the market's sessions from the
    last value published before each session.
    """
    index = market.index
    realised = np.sqrt(_ewm(market**2, HL_REALISED))
    short = short_yield.reindex(index.union(short_yield.index)).ffill().reindex(index)
    long = long_yield.reindex(index.union(long_yield.index)).ffill().reindex(index)
    return pd.DataFrame(
        {
            "mkt_ret_ewm_21": _ewm(market, HL_MARKET),
            "mkt_logvol_diff_ewm_21": _ewm(np.log(realised + _EPS).diff(), HL_MARKET),
            "y1_diff_ewm_21": _ewm(short.diff(), HL_MARKET),
            "slope_diff_ewm_21": _ewm((long - short).diff(), HL_MARKET),
        },
        index=index,
    )


# ---------------------------------------------------------------------------
# one model, one penalty, walk-forward
# ---------------------------------------------------------------------------
def refit_schedule(
    index: pd.DatetimeIndex, first: pd.Timestamp, months: int = 6
) -> pd.DatetimeIndex:
    """First session on or after each ``months``-month calendar step from ``first``."""
    steps = pd.date_range(first, index.max(), freq=f"{months}MS")
    out = [index[index >= s].min() for s in steps]
    return pd.DatetimeIndex([d for d in out if pd.notna(d)]).unique()


def training_window(
    index: pd.DatetimeIndex,
    refit: pd.Timestamp,
    *,
    min_years: int = MIN_TRAIN_YEARS,
    max_years: int = MAX_TRAIN_YEARS,
) -> pd.DatetimeIndex | None:
    """Sessions strictly before ``refit``, at most ``max_years`` back.

    None when fewer than ``min_years`` of history precede ``refit``.
    """
    before = index[index < refit]
    if len(before) == 0 or before.min() > refit - pd.DateOffset(years=min_years):
        return None
    return before[before >= refit - pd.DateOffset(years=max_years)]


@dataclass(frozen=True)
class Fit:
    """One refit: its in-sample labels and the per-state means they imply."""

    refit: pd.Timestamp
    labels: pd.Series
    n_selected: int


def fit_one(
    features: pd.DataFrame,
    active: pd.Series,
    *,
    jump_penalty: float,
    max_feats: float,
    seed: int = 0,
) -> tuple[SparseJumpModel, DataClipperStd, StandardScalerPD, pd.Series]:
    """Fit the sparse jump model on one training block; labels 1 = bull, 0 = bear.

    The winsoriser and the scaler are fitted on the block and returned so that the
    out-of-sample rows are transformed with training statistics only. States are
    sorted by the cumulative active return earned in each on the block, as in the
    paper: the reference implementation puts the highest first, so its label 0 is bull
    and is mapped here to 1.
    """
    clipper = DataClipperStd(mul=CLIP_SD)
    scaler = StandardScalerPD()
    x = scaler.fit_transform(clipper.fit_transform(features))
    model = SparseJumpModel(
        n_components=2, max_feats=max_feats, jump_penalty=jump_penalty, random_state=seed
    )
    model.fit(x, ret_ser=active.reindex(x.index), sort_by="cumret")
    raw = np.asarray(model.labels_, dtype=int)
    labels = pd.Series(1 - raw, index=x.index, name="label")
    return model, clipper, scaler, labels


def walk_forward(
    features: pd.DataFrame,
    active: pd.Series,
    refits: Sequence[pd.Timestamp],
    *,
    jump_penalty: float,
    max_feats: float,
    targets: Mapping[str, pd.Series] | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Out-of-sample states of one model at one penalty, refit on ``refits``.

    Each refit is fitted on its training window (`training_window`) and filters
    forward, centroids fixed, from the start of that window to the session before the
    next refit; only the rows from the refit onwards are kept. Every kept row carries
    the refit it came from and, for each series in ``targets`` (the active return at
    least), the annualised mean of that series on the training sessions labelled bull
    and bear, which is what the paper's exposure rule needs.
    """
    frame = features.dropna()
    targets = dict(targets or {})
    targets.setdefault("active", active)
    refits = [pd.Timestamp(r) for r in refits]
    out: list[pd.DataFrame] = []
    for i, refit in enumerate(refits):
        train = training_window(frame.index, refit)
        if train is None:
            continue
        end = refits[i + 1] if i + 1 < len(refits) else frame.index.max() + pd.Timedelta(days=1)
        block = frame.index[(frame.index >= refit) & (frame.index < end)]
        if len(block) == 0:
            continue
        model, clipper, scaler, labels = fit_one(
            frame.loc[train], active.reindex(train), jump_penalty=jump_penalty,
            max_feats=max_feats, seed=seed,
        )
        path = frame.loc[train.union(block)]
        x = scaler.transform(clipper.transform(path))
        online = 1 - np.asarray(model.predict_online(x), dtype=int)
        states = pd.Series(online, index=path.index).reindex(block)
        rows = pd.DataFrame({"state": states.astype(float), "refit": refit}, index=block)
        for name, series in targets.items():
            mu = state_means(series.reindex(train), labels)
            rows[f"mu_bull_{name}"] = mu[BULL]
            rows[f"mu_bear_{name}"] = mu[BEAR]
        rows["n_selected"] = int((np.asarray(model.w) > 1e-8).sum())
        out.append(rows)
    if not out:
        return pd.DataFrame(columns=["state", "refit"])
    return pd.concat(out).sort_index()


def state_means(returns: pd.Series, labels: pd.Series) -> dict[int, float]:
    """Annualised mean of ``returns`` on the sessions of each label.

    A label with no session takes the mean over all sessions: a model that collapses
    to one state expresses no view beyond the unconditional one.
    """
    aligned = pd.concat({"r": returns, "s": labels}, axis=1).dropna()
    overall = float(aligned["r"].mean() * PERIODS) if len(aligned) else float("nan")
    out: dict[int, float] = {}
    for state in (BEAR, BULL):
        part = aligned.loc[aligned["s"] == state, "r"]
        out[state] = float(part.mean() * PERIODS) if len(part) else overall
    return out


# ---------------------------------------------------------------------------
# exposures
# ---------------------------------------------------------------------------
def onoff_exposure(states: pd.Series) -> pd.Series:
    """1 in the bull state, 0 in the bear state; NaN where the state is unknown."""
    return states.where(states.isna(), (states == BULL).astype(float))


def paper_exposure(mu: pd.Series, *, cap: float = CAP) -> pd.Series:
    """The paper's position: ``clip(mu / cap, -1, 1)``, linear between the caps."""
    return (mu / cap).clip(-1.0, 1.0)


def current_mu(path: pd.DataFrame, target: str) -> pd.Series:
    """The expected annual return of ``target`` in the state each row is in."""
    bull = path[f"mu_bull_{target}"]
    bear = path[f"mu_bear_{target}"]
    return bull.where(path["state"] == BULL, bear).where(path["state"].notna())


def long_short_returns(
    path: pd.DataFrame, active: pd.Series, *, target: str = "active", lag: int = LAG
) -> pd.Series:
    """The paper's single-factor long-short strategy on ``active``, zero cost.

    Position ``clip(mu_state / 5%, -1, 1)`` from the state stamped ``lag`` sessions
    before; the return is position times the active return. Rows before the first
    usable position are NaN.
    """
    position = paper_exposure(current_mu(path, target))
    held = position.reindex(active.index).shift(lag)
    return (held * active).where(held.notna())


# ---------------------------------------------------------------------------
# the tuning rule
# ---------------------------------------------------------------------------
def sharpe(x: pd.Series) -> float:
    v = x.dropna().to_numpy(float)
    if len(v) < 2 or not v.std(ddof=1) > 0:
        return float("nan")
    return float(v.mean() / v.std(ddof=1) * np.sqrt(PERIODS))


def select_penalty(
    paths: Mapping[float, pd.DataFrame],
    active: pd.Series,
    tune_dates: Sequence[pd.Timestamp],
    *,
    validation_years: int = VALIDATION_YEARS,
    lag: int = LAG,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stitch one path from several penalties by the paper's validation rule.

    At each tune date ``tau`` every penalty is scored by the Sharpe of its
    single-factor long-short strategy over the ``validation_years`` before ``tau``,
    on its own out-of-sample path; a penalty whose path does not cover the whole
    validation window is not eligible. The best one supplies the rows from ``tau`` to
    the next tune date. Ties go to the first penalty in ``paths`` order. Returns the
    stitched path and the table of scores.
    """
    tune_dates = [pd.Timestamp(t) for t in tune_dates]
    strategies = {lam: long_short_returns(p, active, lag=lag) for lam, p in paths.items()}
    chosen: list[pd.DataFrame] = []
    scores: list[dict] = []
    for i, tau in enumerate(tune_dates):
        start = tau - pd.DateOffset(years=validation_years)
        end = tune_dates[i + 1] if i + 1 < len(tune_dates) else None
        row: dict = {"tune_date": tau}
        best, best_score = None, -np.inf
        for lam, path in paths.items():
            covered = len(path) and path.index.min() <= start + pd.Timedelta(days=7)
            ls = strategies[lam]
            window = ls[(ls.index >= start) & (ls.index < tau)]
            score = sharpe(window) if covered and window.notna().mean() > 0.95 else float("nan")
            row[f"sharpe_{lam:g}"] = score
            if np.isfinite(score) and score > best_score:
                best, best_score = lam, score
        row["chosen"] = best
        scores.append(row)
        if best is None:
            continue
        path = paths[best]
        mask = path.index >= tau if end is None else (path.index >= tau) & (path.index < end)
        part = path.loc[mask].copy()
        part["penalty"] = best
        chosen.append(part)
    stitched = pd.concat(chosen).sort_index() if chosen else pd.DataFrame()
    return stitched, pd.DataFrame(scores).set_index("tune_date")


# ---------------------------------------------------------------------------
# any other partition, through the same exposure code
# ---------------------------------------------------------------------------
def partition_means(
    labels: pd.Series,
    returns: pd.Series,
    refits: Sequence[pd.Timestamp],
    index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Per-state training means of ``returns`` for a causal partition, refit by refit.

    ``labels`` is a 0/1 series (1 = the "good" side) stamped on the session it is
    known. For each refit, the means are taken on its training window
    (`training_window`) and carried on every session of ``index`` up to the next
    refit; the row's ``state`` is the partition's own label on that session. The
    output has the columns `current_mu` and `long_short_returns` expect.
    """
    refits = [pd.Timestamp(r) for r in refits]
    frame = pd.concat({"s": labels, "r": returns}, axis=1).dropna()
    out: list[pd.DataFrame] = []
    for i, refit in enumerate(refits):
        train = training_window(frame.index, refit)
        if train is None:
            continue
        end = refits[i + 1] if i + 1 < len(refits) else index.max() + pd.Timedelta(days=1)
        block = index[(index >= refit) & (index < end)]
        mu = state_means(frame.loc[train, "r"], frame.loc[train, "s"])
        rows = pd.DataFrame({"state": labels.reindex(block), "refit": refit}, index=block)
        rows["mu_bull_target"] = mu[BULL]
        rows["mu_bear_target"] = mu[BEAR]
        out.append(rows)
    if not out:
        return pd.DataFrame(columns=["state", "refit"])
    return pd.concat(out).sort_index()


def trailing_return_rule(returns: pd.Series, window: int = PERIODS) -> pd.Series:
    """1 when the sum of the last ``window`` returns, the session included, is positive."""
    total = returns.rolling(window).sum()
    return (total > 0).astype(float).where(total.notna()).rename("trailing_rule")


# ---------------------------------------------------------------------------
# the book
# ---------------------------------------------------------------------------
def book_returns(
    sleeves: pd.DataFrame, exposure: pd.DataFrame, *, lag: int = LAG
) -> pd.Series:
    """Equal-budget book: ``mean_k exposure_k(t - lag) * sleeve_k(t)``.

    ``sleeves`` holds the volatility-targeted daily return of each factor (weight
    already lagged); ``exposure`` is stamped on the session its state is known and is
    shifted here. A session where any factor's exposure or sleeve is missing is NaN.
    """
    held = exposure.reindex(index=sleeves.index, columns=sleeves.columns).shift(lag)
    contrib = held * sleeves
    out = contrib.mean(axis=1)
    return out.where(contrib.notna().all(axis=1))


# ---------------------------------------------------------------------------
# descriptive measures
# ---------------------------------------------------------------------------
def participation_ratio(corr: np.ndarray) -> float:
    """``(sum eigenvalues)^2 / sum eigenvalues^2`` of a correlation matrix."""
    eig = np.linalg.eigvalsh(np.asarray(corr, dtype=float))
    return float(eig.sum() ** 2 / (eig**2).sum())


def cohen_kappa(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a).astype(bool)
    b = np.asarray(b).astype(bool)
    observed = float((a == b).mean())
    chance = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
    return (observed - chance) / (1.0 - chance) if chance < 1 else float("nan")


def transitions_per_year(states: pd.Series) -> float:
    clean = states.dropna()
    if len(clean) < 2:
        return float("nan")
    years = len(clean) / PERIODS
    return float((clean.diff().abs() > 0).sum() / years)

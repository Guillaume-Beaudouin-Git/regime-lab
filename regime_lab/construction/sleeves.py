"""Bridgewater study — the risk sleeves, and the book held without the quadrant.

`docs/PRESPEC_BRIDGEWATER.md` §1.3 and §2 declare five risk sleeves over 30
instruments, equal risk inside each sleeve, risk parity across sleeves, a daily 10%
volatility target, a quarterly rebalance and the vehicle's cost schedule. This
module builds that book from the stored prices for ANY sleeve-weight schedule, so
the evaluation legs (blind, balanced, the placebos, the B2 map) all go through one
construction and differ only in the schedule they pass.

**Nothing here reads the quadrant.** The module has no notion of a label. It turns
prices into excess returns, a sleeve-weight schedule into held instrument weights,
and held weights into a volatility-targeted, costed book.

**Conventions, each one the programme's, named with its source.**

- *Returns are simple returns* ``p_t / p_{t-1} - 1``, as `extensions.vehicle`
  computes them (``prices.pct_change()``). A price that is missing or not positive
  gives no return on that session or the next: CL=F settled at -37.63 on
  2020-04-20, where a simple return would read -306% then -127%. This is the
  treatment `extensions.risque.log_returns` gives the same session. A return after a
  gap is missing too (the move across the gap is not booked on one session), which
  is also what ``pct_change`` does under pandas 3.
- *A missing return on a held session contributes zero* to the book, as
  ``(weights * returns).sum(axis=1)`` does in `extensions.vehicle`. The position is
  held; its P&L on that session is unobserved. PL=F is the one instrument this
  touches materially (540 missing sessions in the headline sample).
- *Funding.* `scripts/fetch_trend_universe.py` downloads with ``auto_adjust=True``,
  so every ETF column is a total return and is funded at the 3-month cash rate
  (`extensions.vehicle.FUNDED_ON_CASH`: SHY IEF TLT AGG TIP, and LQD HYG EMB BWX).
  The rate is ``data/raw/macro/rate_cash_3m.parquet`` stamped at ``available_at``
  and carried forward, divided by 100 and by 252, exactly as
  `scripts/run_m3_evaluation.py` builds it. It is subtracted at the instrument
  level, which is the programme's "signed" funding reading: a book pays cash on its
  signed exposure to the funded columns only.
- *Futures and price indices are not funded.* A futures return is excess by
  construction. An equity price index is read as the excess return of its future,
  which is the programme's convention; the error of that reading is the index's
  dividend yield minus its local cash rate, not the whole dividend. ``^GDAXI`` is
  the exception the convention never named: the DAX is a *performance* (total
  return) index, so its unfunded return overstates the excess return by the euro
  cash rate. Spot FX carries no coupon and is not funded either; its interest
  differential is ignored, as in `extensions.vehicle`.
- *Within a sleeve, equal standalone risk*: weights proportional to ``1/σ_i``,
  summing to one, σ being the sample standard deviation of the instrument's excess
  return over the estimation window. Intra-sleeve correlation is ignored.
- *Across sleeves*: `blind_sleeve_weights` offers ``"erc"`` (equal risk
  contribution on the sleeve covariance, `selection.protocol.risk_parity_mix`) and
  ``"inverse_vol"`` (its diagonal special case). The draft does not say which one
  "risk parity" means. Its balanced leg equalises *contributions to variance*, so
  ERC is the reading under which the two legs differ only in the conditioning.
- *Rebalance and timing.* A schedule row dated τ uses information up to the close
  of the last session at or before τ, and is first held on the first session
  strictly after τ (signal at T-1, trade at T). Between rebalances the held
  fractions are constant: the book is rebalanced back to them every session, and,
  as everywhere in the programme, that drift rebalancing is not charged.
- *Volatility target*: `extensions.vehicle.portfolio_scalar`, 10% on 63 sessions of
  the unscaled book, lagged one session, the multiplier capped at 3.0 — the reading
  of `docs/PROTOCOL_FREEZE.md`, 2026-09-21. The sleeve book is long only and its
  unscaled weights sum to one, so here **the multiplier IS the gross exposure**:
  the two readings that row separated coincide on this object.
- *Costs*: `extensions.vehicle.cost_bps` round trip per instrument, charged on
  ``|Δw|`` of the weights actually held (after the multiplier) through
  `extensions.vehicle.charge_by_instrument`. The entry into the book is charged.
  Four columns: ``headline``, ``conservative``, ``stress`` and ``all_cash``.
- *Panel*: ``data/cache/trend_universe_m1.parquet``. Its 30 headline columns and the
  three credit columns are identical to ``trend_universe.parquet``; the two panels
  differ only on the eight spot FX series that M1 realigned, which only the FX
  sensitivity uses, and there the realigned series are the repaired ones.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from regime_lab.config import CACHE, RAW
from regime_lab.data.pit import validate
from regime_lab.extensions.trend import MAX_LEVERAGE, VOL_TARGET
from regime_lab.extensions.vehicle import (
    FUNDED_ON_CASH,
    charge_by_instrument,
    cost_bps,
    portfolio_scalar,
)
from regime_lab.selection.protocol import risk_parity_mix

PERIODS = 252
VOL_WINDOW = 63
DAYS_PER_YEAR = 365.25

PANEL = CACHE / "trend_universe_m1.parquet"
CASH = RAW / "macro" / "rate_cash_3m.parquet"

#: §1.3, the headline: five sleeves, 30 instruments.
HEADLINE_SLEEVES: dict[str, tuple[str, ...]] = {
    "equity": (
        "^GSPC", "^NDX", "^RUT", "^GDAXI", "^FTSE", "^N225",
        "^AXJO", "^GSPTSE", "^HSI", "^AEX", "^FCHI", "^IBEX",
    ),
    "duration": ("SHY", "IEF", "TLT", "AGG"),
    "inflation_linked": ("TIP",),
    "commodity": (
        "CL=F", "HO=F", "NG=F", "HG=F", "PL=F", "ZC=F",
        "ZS=F", "ZW=F", "KC=F", "SB=F", "CT=F",
    ),
    "precious": ("GC=F", "SI=F"),
}

#: §1.3, the two sensitivities. Never the headline: they cost 4.2 years.
CREDIT_SLEEVE: tuple[str, ...] = ("LQD", "HYG", "EMB")
FX_SLEEVE: tuple[str, ...] = (
    "DX-Y.NYB", "EURUSD=X", "GBPUSD=X", "AUDUSD=X", "JPY=X", "CHF=X",
    "CAD=X", "NZDUSD=X", "SEK=X", "NOK=X", "MXN=X",
)

CONFIGURATIONS: dict[str, dict[str, tuple[str, ...]]] = {
    "headline": dict(HEADLINE_SLEEVES),
    "with_credit": {**HEADLINE_SLEEVES, "credit": CREDIT_SLEEVE},
    "with_fx": {**HEADLINE_SLEEVES, "credit": CREDIT_SLEEVE, "fx": FX_SLEEVE},
}

#: Quoted as units of foreign currency per dollar (the dollar index is the dollar
#: itself). The draft gives the FX sleeve no orientation; the one taken here holds
#: every currency LONG against the dollar, so these seven are inverted and the
#: sleeve is short the dollar throughout. Only the FX sensitivity is affected.
USD_BASE_QUOTES: tuple[str, ...] = ("DX-Y.NYB", "JPY=X", "CHF=X", "CAD=X", "SEK=X", "NOK=X",
                                    "MXN=X")

#: Equity columns that are total-return indices and not price indices.
TOTAL_RETURN_INDICES: tuple[str, ...] = ("^GDAXI",)

COST_COLUMNS: tuple[str, ...] = ("headline", "conservative", "stress", "all_cash")

DEFAULT_LOOKBACK = 252
#: An instrument enters a sleeve at a rebalance only if at least this share of the
#: estimation window carries a return. PL=F loses up to 50 consecutive sessions.
DEFAULT_MIN_VALID = 0.5

Method = Literal["erc", "inverse_vol"]
Frequency = Literal["daily", "weekly"]


# ---------------------------------------------------------------------------
# the universe
# ---------------------------------------------------------------------------
def sleeve_map(configuration: str = "headline") -> dict[str, tuple[str, ...]]:
    """The sleeves of one declared configuration: ``headline``, ``with_credit``, ``with_fx``."""
    if configuration not in CONFIGURATIONS:
        raise ValueError(f"unknown configuration {configuration!r}; use {list(CONFIGURATIONS)}")
    return dict(CONFIGURATIONS[configuration])


def instruments(sleeves: Mapping[str, Sequence[str]]) -> list[str]:
    """Every instrument of ``sleeves``, in sleeve order; an instrument may sit in one sleeve."""
    names = [name for members in sleeves.values() for name in members]
    if len(set(names)) != len(names):
        raise ValueError("an instrument appears in two sleeves")
    return names


def load_prices(
    sleeves: Mapping[str, Sequence[str]] | None = None, path: Path | None = None
) -> pd.DataFrame:
    """Stored prices of the sleeves' instruments, sessions x instruments, sleeve order."""
    sleeves = HEADLINE_SLEEVES if sleeves is None else sleeves
    names = instruments(sleeves)
    panel = pd.read_parquet(path or PANEL)
    missing = [n for n in names if n not in panel.columns]
    if missing:
        raise KeyError(f"instruments absent from the panel: {missing}")
    out = panel.loc[:, names].astype(float)
    out.index = pd.DatetimeIndex(out.index).astype("datetime64[ns]")
    out.index.name = "session"
    return out


def coverage(prices: pd.DataFrame, sleeves: Mapping[str, Sequence[str]]) -> pd.DataFrame:
    """Per sleeve: size, the latest first price, and which instrument sets it."""
    rows = []
    for sleeve, members in sleeves.items():
        first = {name: prices[name].first_valid_index() for name in members}
        binding = max(first, key=lambda name: first[name])
        rows.append({"sleeve": sleeve, "n": len(members), "start": first[binding],
                     "binding": binding})
    return pd.DataFrame(rows).set_index("sleeve")


def configuration_start(prices: pd.DataFrame, sleeves: Mapping[str, Sequence[str]]) -> pd.Timestamp:
    """The first session on which every instrument has started trading (§1.3's start)."""
    return pd.Timestamp(coverage(prices, sleeves)["start"].max())


# ---------------------------------------------------------------------------
# returns and funding
# ---------------------------------------------------------------------------
def price_returns(
    prices: pd.DataFrame, *, invert: Sequence[str] = USD_BASE_QUOTES
) -> pd.DataFrame:
    """Simple returns, missing wherever this price or the previous one is missing or ≤ 0.

    Columns named in ``invert`` are quoted per dollar and are turned into the dollar
    value of the foreign currency: their return is ``p_{t-1} / p_t - 1``.
    """
    positive = prices.where(prices > 0)
    previous = positive.shift(1)
    out = positive / previous - 1.0
    flipped = [c for c in prices.columns if c in set(invert)]
    if flipped:
        out[flipped] = previous[flipped] / positive[flipped] - 1.0
    return out


def cash_rate(index: pd.DatetimeIndex, frame: pd.DataFrame | None = None) -> pd.Series:
    """Daily 3-month cash rate in decimal per session, stamped at ``available_at``.

    Session t carries the latest value published at or before t. The stored series
    is published the day after the date it describes, so a session never uses the
    rate of its own date. Identical to `scripts/run_m3_evaluation.cash_rate_daily`.
    """
    raw = validate(frame if frame is not None else pd.read_parquet(CASH))
    series = (
        raw.sort_values("available_at")
        .drop_duplicates("available_at", keep="last")
        .set_index("available_at")["value"]
    )
    target = pd.DatetimeIndex(index).astype("datetime64[ns]")
    carried = series.reindex(series.index.union(target)).ffill().reindex(target)
    return (carried / 100.0 / PERIODS).rename("cash")


def excess_returns(
    returns: pd.DataFrame, cash: pd.Series, *, funded: Sequence[str] = FUNDED_ON_CASH
) -> pd.DataFrame:
    """Subtract the cash rate from the total-return columns only (``funded``)."""
    out = returns.copy()
    columns = [c for c in returns.columns if c in set(funded)]
    if columns:
        rate = cash.reindex(returns.index)
        if rate.loc[returns[columns].notna().any(axis=1)].isna().any():
            raise ValueError("the cash rate is missing on a session where a funded column trades")
        out[columns] = returns[columns].sub(rate, axis=0)
    return out


def funding_charge(
    weights: pd.DataFrame, cash: pd.Series, *, funded: Sequence[str] = FUNDED_ON_CASH
) -> pd.Series:
    """The cash a book pays on its signed exposure to the funded columns, per session.

    Already inside a book built on `excess_returns`; reported separately so the
    financing of the bond ETFs can be read as a cost line.
    """
    columns = [c for c in weights.columns if c in set(funded)]
    if not columns:
        return pd.Series(0.0, index=weights.index, name="funding")
    exposure = weights[columns].sum(axis=1, min_count=1)
    return (exposure * cash.reindex(weights.index)).rename("funding")


# ---------------------------------------------------------------------------
# estimation at the rebalance dates
# ---------------------------------------------------------------------------
def anchor_sessions(sessions: pd.DatetimeIndex, dates: Sequence[pd.Timestamp]) -> pd.DatetimeIndex:
    """For each rebalance date, the last session at or before it (NaT if none)."""
    sessions = pd.DatetimeIndex(sessions)
    stamps = pd.DatetimeIndex(pd.to_datetime(list(dates))).astype(sessions.dtype)
    position = sessions.searchsorted(stamps, side="right") - 1
    return pd.DatetimeIndex(
        [sessions[p] if p >= 0 else pd.NaT for p in position]
    ).astype(sessions.dtype)


def _window(
    returns: pd.DataFrame,
    anchor: pd.Timestamp,
    *,
    lookback: int | None,
    start: pd.Timestamp | None,
    min_sessions: int,
) -> pd.DataFrame | None:
    """Rows of ``returns`` in the estimation window ending at ``anchor``, or None if short."""
    if pd.isna(anchor):
        return None
    eligible = returns.loc[:anchor]
    if start is not None:
        eligible = eligible.loc[pd.Timestamp(start):]
    if lookback is not None:
        if len(eligible) < lookback:
            return None
        eligible = eligible.iloc[-lookback:]
    return eligible if len(eligible) >= min_sessions else None


def _aggregate(frame: pd.DataFrame, frequency: Frequency) -> pd.DataFrame:
    """Daily rows as they are, or summed by calendar week ending Friday."""
    if frequency == "daily":
        return frame
    if frequency == "weekly":
        return frame.groupby(pd.Grouper(freq="W-FRI")).sum(min_count=1)
    raise ValueError(f"frequency must be 'daily' or 'weekly', not {frequency!r}")


def within_sleeve_weights(
    returns: pd.DataFrame,
    sleeves: Mapping[str, Sequence[str]],
    dates: Sequence[pd.Timestamp],
    *,
    lookback: int | None = DEFAULT_LOOKBACK,
    start: pd.Timestamp | None = None,
    min_sessions: int | None = None,
    min_valid: float = DEFAULT_MIN_VALID,
    frequency: Frequency = "daily",
) -> pd.DataFrame:
    """Inverse-volatility weights inside each sleeve, one row per rebalance date.

    Row τ is estimated on the ``lookback`` sessions ending at the last session at or
    before τ (``None`` = expanding), never before ``start``. An instrument with fewer
    than ``min_valid`` of the window's sessions carrying a return, or a zero
    variance, gets weight zero and the rest of its sleeve is renormalised; a sleeve
    left with no qualified instrument, or a window shorter than ``min_sessions``,
    gives NaN for that sleeve's columns. Each sleeve's weights sum to one.
    ``frequency="weekly"`` takes σ from calendar-week sums of the same window, the
    answer to closes that are not synchronous; the eligibility count stays daily.
    """
    names = instruments(sleeves)
    need = min_sessions if min_sessions is not None else (lookback or DEFAULT_LOOKBACK)
    anchors = anchor_sessions(returns.index, dates)
    out = pd.DataFrame(np.nan, index=pd.DatetimeIndex(pd.to_datetime(list(dates))),
                       columns=names)
    out.index.name = "rebalance"
    for tau, anchor in zip(out.index, anchors, strict=True):
        window = _window(returns[names], anchor, lookback=lookback, start=start,
                         min_sessions=need)
        if window is None:
            continue
        valid = window.notna().sum() >= min_valid * len(window)
        sd = _aggregate(window, frequency).std(ddof=1).where(valid)
        sd = sd.where(sd > 0)
        for members in sleeves.values():
            inv = 1.0 / sd[list(members)]
            if inv.notna().any():
                out.loc[tau, list(members)] = (inv / inv.sum()).fillna(0.0).to_numpy()
    return out


def sleeve_returns(
    returns: pd.DataFrame,
    within: pd.Series | pd.DataFrame,
    sleeves: Mapping[str, Sequence[str]],
) -> pd.DataFrame:
    """Sleeve returns from instrument returns and within-sleeve weights.

    ``within`` is either one row of weights (a Series), applied to every session,
    or a sessions x instruments frame of weights held on each session. A missing
    instrument return contributes zero, as it does in the book.
    """
    names = instruments(sleeves)
    r = returns.reindex(columns=names).fillna(0.0)
    if isinstance(within, pd.Series):
        product = r.mul(within.reindex(names), axis=1)
    else:
        product = r * within.reindex(index=r.index, columns=names)
    return pd.DataFrame(
        {s: product[list(m)].sum(axis=1, min_count=len(m)) for s, m in sleeves.items()},
        index=r.index,
    )


def blind_sleeve_weights(
    returns: pd.DataFrame,
    sleeves: Mapping[str, Sequence[str]],
    within: pd.DataFrame,
    *,
    method: Method = "erc",
    lookback: int | None = DEFAULT_LOOKBACK,
    start: pd.Timestamp | None = None,
    min_sessions: int | None = None,
    frequency: Frequency = "daily",
) -> pd.DataFrame:
    """Unconditional risk parity across sleeves at each rebalance date of ``within``.

    At τ the sleeves are formed with row τ of ``within`` over the same window as
    `within_sleeve_weights`, and their sample covariance (ddof 1) gives ERC weights
    (``"erc"``) or inverse-volatility weights (``"inverse_vol"``). Rows sum to one;
    a date where any sleeve is undefined gives a NaN row. Nothing here reads a label.
    ``frequency="weekly"`` estimates the covariance on calendar-week sums.
    """
    if method not in ("erc", "inverse_vol"):
        raise ValueError(f"method must be 'erc' or 'inverse_vol', not {method!r}")
    need = min_sessions if min_sessions is not None else (lookback or DEFAULT_LOOKBACK)
    names = instruments(sleeves)
    anchors = anchor_sessions(returns.index, within.index)
    out = pd.DataFrame(np.nan, index=within.index, columns=list(sleeves))
    for tau, anchor in zip(within.index, anchors, strict=True):
        row = within.loc[tau, names]
        if row.isna().any():
            continue
        window = _window(returns[names], anchor, lookback=lookback, start=start,
                         min_sessions=need)
        if window is None:
            continue
        cov = _aggregate(sleeve_returns(window, row, sleeves), frequency).cov(ddof=1)
        if method == "inverse_vol":
            inv = 1.0 / np.sqrt(np.diag(cov.to_numpy()))
            out.loc[tau] = inv / inv.sum()
        else:
            out.loc[tau] = risk_parity_mix(cov).to_numpy()
    return out


def instrument_schedule(
    sleeve_weights: pd.DataFrame,
    within: pd.DataFrame,
    sleeves: Mapping[str, Sequence[str]],
) -> pd.DataFrame:
    """Instrument weights ``W_i(τ) = S_s(τ) · w_i|s(τ)``, one row per rebalance date.

    ``sleeve_weights`` is ANY schedule, dates x sleeves, rows summing to one — the
    blind leg's, a balanced leg's, a placebo draw's or an ex-ante map's. Its dates
    must be rows of ``within``. A row with any missing sleeve weight stays NaN. A
    sleeve at zero weight holds nothing, even where its within weights are undefined.
    """
    names = instruments(sleeves)
    missing = [d for d in sleeve_weights.index if d not in within.index]
    if missing:
        raise KeyError(f"no within-sleeve weights for {missing[:3]}")
    complete = sleeve_weights.notna().all(axis=1)
    if not np.allclose(sleeve_weights.loc[complete].sum(axis=1), 1.0, atol=1e-9):
        raise ValueError("sleeve-weight rows must sum to one")
    if (sleeve_weights.loc[complete] < 0).to_numpy().any():
        raise ValueError("sleeve weights must be non-negative")
    inner = within.loc[sleeve_weights.index, names]
    scale = pd.DataFrame(
        {name: sleeve_weights[s] for s, members in sleeves.items() for name in members},
        index=sleeve_weights.index,
    )[names]
    out = (inner * scale).mask(scale == 0.0, 0.0)
    out.loc[~complete, :] = np.nan
    return out


def hold(schedule: pd.DataFrame, sessions: pd.DatetimeIndex) -> pd.DataFrame:
    """Weights held on each session: the last schedule row dated STRICTLY before it.

    A row dated τ is known at the close of τ and first held on the next session.
    Sessions before the first row hold NaN.
    """
    if not (schedule.index.is_monotonic_increasing and schedule.index.is_unique):
        raise ValueError("the schedule must be sorted by date without duplicates")
    sessions = pd.DatetimeIndex(sessions)
    stamps = pd.DatetimeIndex(schedule.index).astype(sessions.dtype)
    position = stamps.searchsorted(sessions, side="left") - 1
    values = np.vstack([schedule.to_numpy(float), np.full((1, schedule.shape[1]), np.nan)])
    out = values[np.where(position >= 0, position, len(schedule))]
    return pd.DataFrame(out, index=sessions, columns=schedule.columns)


# ---------------------------------------------------------------------------
# the book
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SleeveBook:
    """A sleeve book at the volatility target, gross of trading cost, in excess of cash."""

    #: Held fractions before the multiplier; long only, each row sums to one.
    unscaled_weights: pd.DataFrame
    #: Weights actually held after the multiplier: what costs are charged on.
    weights: pd.DataFrame
    #: Unscaled book return, defined from the first held row on.
    unscaled_returns: pd.Series
    #: Book return at the target, NaN during the 63-session warm-up.
    returns: pd.Series
    multiplier: pd.Series
    cap_binds: pd.Series

    @property
    def live(self) -> pd.DatetimeIndex:
        """Sessions on which the book exists at its target."""
        return self.returns.index[self.returns.notna()]


def build_book(
    excess: pd.DataFrame,
    held: pd.DataFrame,
    *,
    target: float = VOL_TARGET,
    window: int = VOL_WINDOW,
    cap: float = MAX_LEVERAGE,
) -> SleeveBook:
    """Hold ``held`` (sessions x instruments, from `hold`) at the daily volatility target.

    The multiplier on session t is ``target / σ`` of the unscaled book over the 63
    sessions ending at t-1, capped at ``cap`` (`extensions.vehicle.portfolio_scalar`).
    It exists once 63 unscaled returns have been observed; before that the book is
    NaN, never zero.
    """
    r = excess.reindex(index=held.index, columns=held.columns)
    defined = held.notna().all(axis=1)
    unscaled = (held * r.fillna(0.0)).sum(axis=1).where(defined).rename("unscaled")
    multiplier, binds = portfolio_scalar(unscaled, window=window, target=target, cap=cap)
    live = unscaled.rolling(window).count().shift(1) >= window
    multiplier = multiplier.where(live).rename("multiplier")
    binds = binds.astype(float).where(live).rename("cap_binds")
    return SleeveBook(
        unscaled_weights=held,
        weights=held.mul(multiplier, axis=0),
        unscaled_returns=unscaled,
        returns=(unscaled * multiplier).rename("book"),
        multiplier=multiplier,
        cap_binds=binds,
    )


def cost_rates(columns: pd.Index, column: str = "headline") -> pd.Series:
    """Round-trip bps per instrument for one of the four `COST_COLUMNS`."""
    if column == "all_cash":
        return cost_bps(columns, column="headline", all_cash=True)
    if column not in COST_COLUMNS:
        raise ValueError(f"unknown cost column {column!r}; use {COST_COLUMNS}")
    return cost_bps(columns, column=column)


def cost_drag(weights: pd.DataFrame, column: str = "headline") -> pd.Series:
    """Per-session cost ``Σ_i |Δw_i| · bps_i / 10,000`` on the held weights.

    A weight that is not yet defined counts as zero, so entering the book is paid.
    """
    zero = pd.Series(0.0, index=weights.index, name="cost")
    return -charge_by_instrument(zero, weights.fillna(0.0), cost_rates(weights.columns, column))


def net_returns(book: SleeveBook, column: str = "headline") -> pd.Series:
    """Book return net of trading cost at one cost column, in excess of cash."""
    return (book.returns - cost_drag(book.weights, column)).rename(f"net_{column}")


def blended_cost_bps(
    columns: Sequence[str], column: str = "headline", weights: pd.Series | None = None
) -> float:
    """Average round-trip bps over ``columns``, equal-weighted unless ``weights`` given."""
    rates = cost_rates(pd.Index(list(columns)), column)
    if weights is None:
        return float(rates.mean())
    w = weights.reindex(rates.index).fillna(0.0).abs()
    return float((rates * w).sum() / w.sum())


def annual_turnover(weights: pd.DataFrame) -> float:
    """Mean ``Σ|Δw|`` per session times 252, over sessions where both rows are defined."""
    diff = weights.diff().abs()
    changes = diff.sum(axis=1).where(diff.notna().all(axis=1)).iloc[1:]
    return float(changes.mean() * PERIODS) if changes.notna().any() else float("nan")


def book_diagnostics(
    book: SleeveBook,
    *,
    sessions: pd.DatetimeIndex | None = None,
    cash: pd.Series | None = None,
) -> dict[str, object]:
    """What a state-blind book may disclose before the lock. No mean return is read.

    Realised standard deviation (daily x √252, and from calendar-week sums x √52),
    cap-binding share, gross exposure, turnover and the whole trading cost in Sharpe
    units at the realised sd and at the nominal target, per cost column, and the
    cash paid on the funded columns. ``sessions`` restricts every figure (e.g. to the
    test folds); the default is every live session.
    """
    idx = book.live if sessions is None else book.live.intersection(sessions)
    if len(idx) < 2:
        raise ValueError("fewer than two live sessions")
    r = book.returns.loc[idx]
    sd = float(r.std(ddof=1) * np.sqrt(PERIODS))
    # Demeaned first: weeks of four and five sessions would otherwise let the mean in.
    weekly = (r - r.mean()).groupby(pd.Grouper(freq="W-FRI")).sum(min_count=1).dropna()
    gross = book.weights.loc[idx].abs().sum(axis=1)
    out: dict[str, object] = {
        "sessions": len(idx),
        "first": idx[0],
        "last": idx[-1],
        "years_sessions": len(idx) / PERIODS,
        "years_calendar": (idx[-1] - idx[0]).days / DAYS_PER_YEAR,
        "sessions_per_calendar_year": len(idx) / ((idx[-1] - idx[0]).days / DAYS_PER_YEAR),
        "realised_sd": sd,
        "realised_sd_weekly": float(weekly.std(ddof=1) * np.sqrt(52)),
        "cap_share": float(book.cap_binds.loc[idx].mean()),
        "multiplier_median": float(book.multiplier.loc[idx].median()),
        "gross_median": float(gross.median()),
        "gross_p95": float(gross.quantile(0.95)),
        "gross_max": float(gross.max()),
        "gross_max_date": gross.idxmax(),
        "turnover": annual_turnover(book.weights.loc[idx]),
        "turnover_unscaled": annual_turnover(book.unscaled_weights.loc[idx]),
    }
    traded = book.weights.fillna(0.0).diff().abs().loc[idx].sum()
    for column in COST_COLUMNS:
        annual = float(cost_drag(book.weights, column).loc[idx].mean() * PERIODS)
        out[f"cost_{column}"] = annual
        out[f"cost_sharpe_{column}"] = annual / sd
        out[f"cost_sharpe_nominal_{column}"] = annual / VOL_TARGET
        out[f"traded_bps_{column}"] = blended_cost_bps(list(traded.index), column, traded)
    if cash is not None:
        paid = float(funding_charge(book.weights, cash).loc[idx].mean() * PERIODS)
        out["funding"] = paid
        out["funding_sharpe"] = paid / sd
    return out


def blind_book(
    excess: pd.DataFrame,
    sleeves: Mapping[str, Sequence[str]],
    dates: Sequence[pd.Timestamp],
    *,
    method: Method = "erc",
    lookback: int | None = DEFAULT_LOOKBACK,
    start: pd.Timestamp | None = None,
    min_valid: float = DEFAULT_MIN_VALID,
    frequency: Frequency = "daily",
    target: float = VOL_TARGET,
    window: int = VOL_WINDOW,
    cap: float = MAX_LEVERAGE,
) -> tuple[SleeveBook, pd.DataFrame, pd.DataFrame]:
    """The state-blind leg end to end: (book, within-sleeve weights, sleeve weights).

    Sessions before ``start`` are neither estimated on nor held. ``frequency`` applies
    to the weight estimates only; the volatility target stays daily.
    """
    within = within_sleeve_weights(excess, sleeves, dates, lookback=lookback, start=start,
                                   min_valid=min_valid, frequency=frequency)
    across = blind_sleeve_weights(excess, sleeves, within, method=method, lookback=lookback,
                                  start=start, frequency=frequency)
    schedule = instrument_schedule(across, within, sleeves)
    sessions = excess.index if start is None else excess.index[excess.index >= start]
    book = build_book(excess, hold(schedule, sessions), target=target, window=window, cap=cap)
    return book, within, across

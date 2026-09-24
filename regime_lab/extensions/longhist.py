"""A reduced sparse jump model on a hundred years: features, labels, validation, verdict.

The programme's classifier (A' sparse jump, 50 features, 2002-2026 out of sample) was
validated on two NBER recessions and three stress episodes. Every verdict built on it
hits the same wall: too few episodes. This module rebuilds the same *method* on the
subset of its features that can be computed since 1926 from free data, so that the
classifier can be scored on about fourteen recessions and the one use that showed
something (stopping momentum in stress) can be re-read with four times the crises.

It is a cousin of A', not A': no VIX before 1990, no first-release macro data, no
NFCI, no oil, no dollar. Feature definitions are the repository's own
(`features.market`, `features.asymmetry`) wherever the data allow; the cross-sectional
ones are re-implemented here only because the panels of the 1920s-1960s have missing
industries, which the repository versions (written for complete panels) turn into
missing features. See `docs/PRESPEC_LONGHIST.md`.

Conventions, as everywhere in the programme: a label is 1 in calm and 0 in stress
(`extensions.crisis.STRESS`); every rolling quantity uses the past only; a label
formed at the close of d is traded on d+1 (`extensions.crisis.lagged_state`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from regime_lab.data import store
from regime_lab.features import asymmetry, market
from regime_lab.features.standardise import expanding_zscore, winsorise

SOURCE = "longhist"
STRESS = 0.0
CALM = 1.0
PERIODS = 252
EQUITY = market.EQUITY

#: Features of the reduced model, grouped as in `features.build.FAMILIES`.
MARKET_FEATURES = (
    "mom_eq_21", "mom_eq_63", "mom_eq_252", "mom_eq_accel",
    "vol_rv_5", "vol_rv_21", "vol_rv_63", "vol_term", "vol_of_vol", "vol_semi_ratio",
    "vol_jump_share",
    "asy_skew_63", "asy_kurt_63", "asy_drawdown_252", "asy_hurst", "asy_vr_5", "asy_vr_20",
)
CROSS_FEATURES = (
    "xs_dispersion_ind", "xs_dispersion_szbm", "xs_avg_corr", "xs_absorption",
    "xs_absorption_chg", "xs_breadth_63", "xs_ff_smb_63", "xs_ff_hml_63",
)
RATE_FEATURES = (
    "cre_quality", "cre_quality_chg63", "rat_slope_term", "rat_slope_term_chg63",
    "rat_cash_chg12m",
)
MACRO_FEATURES = ("mac_indpro_yoy", "mac_indpro_3m")
HEADLINE_FEATURES = MARKET_FEATURES + CROSS_FEATURES + RATE_FEATURES


# ---------------------------------------------------------------------------
# speed: the jump model's dynamic programme, bit for bit, without numpy per row
# ---------------------------------------------------------------------------
def fast_dp(loss_mx: np.ndarray, penalty_mx: np.ndarray, return_value_mx: bool = False):
    """Drop-in replacement for `jumpmodels.jump.dp`, identical to the bit for two states.

    The reference implementation runs one numpy reduction per row, which costs about
    as much as the arithmetic of a whole row a hundred times over; on a hundred years
    of sessions it makes a walk-forward take days. This version performs the same
    IEEE operations in the same order on Python floats: the value matrix, the path
    and the optimum are equal to the reference's exactly (`tests/test_longhist.py`).
    Ties resolve to the lower state, as `numpy.argmin` does. Other state counts fall
    back to the reference.
    """
    from jumpmodels import jump as reference

    n_s, n_c = loss_mx.shape
    if n_c != 2:
        return _REFERENCE_DP(loss_mx, penalty_mx, return_value_mx=return_value_mx)
    assert penalty_mx.shape == (n_c, n_c)
    loss = reference.replace_nan_by_inf(loss_mx)
    p00, p01 = float(penalty_mx[0, 0]), float(penalty_mx[0, 1])
    p10, p11 = float(penalty_mx[1, 0]), float(penalty_mx[1, 1])
    l0, l1 = loss[:, 0].tolist(), loss[:, 1].tolist()
    out0, out1 = [0.0] * n_s, [0.0] * n_s
    v0, v1 = l0[0], l1[0]
    out0[0], out1[0] = v0, v1
    for t in range(1, n_s):
        a, b = v0 + p00, v1 + p10
        m0 = b if b < a else a
        a, b = v0 + p01, v1 + p11
        m1 = b if b < a else a
        v0, v1 = l0[t] + m0, l1[t] + m1
        out0[t], out1[t] = v0, v1
    values = np.column_stack([np.asarray(out0), np.asarray(out1)])
    if return_value_mx:
        return values
    assign = np.empty(n_s, dtype=int)
    last = 1 if out1[-1] < out0[-1] else 0
    assign[-1] = last
    column = ((p00, p10), (p01, p11))
    for t in range(n_s - 1, 0, -1):
        c0, c1 = column[last]
        last = 1 if out1[t - 1] + c1 < out0[t - 1] + c0 else 0
        assign[t - 1] = last
    return assign, values[-1, assign[-1]]


def _reference_dp():
    from jumpmodels import jump as reference

    return getattr(reference, "_longhist_reference_dp", reference.dp)


_REFERENCE_DP = _reference_dp()


def use_fast_dp() -> None:
    """Route `jumpmodels`' E-step through `fast_dp`. Idempotent; affects this process only."""
    from jumpmodels import jump as reference

    if not hasattr(reference, "_longhist_reference_dp"):
        reference._longhist_reference_dp = reference.dp
    reference.dp = fast_dp


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def load_wide(name: str) -> pd.DataFrame:
    """One stored long-history frame as a wide panel indexed by ``period``."""
    frame = store.read(SOURCE, name)
    wide = frame.pivot(index="period", columns="series_id", values="value")
    wide.index = pd.DatetimeIndex(wide.index)
    wide.columns = [str(c) for c in wide.columns]
    return wide.sort_index()


def market_returns(ff3: pd.DataFrame) -> pd.DataFrame:
    """Daily market total return, excess return and cash, as decimals.

    Ken French's ``Mkt-RF`` is the CRSP value-weighted market in excess of the
    one-month bill, ``RF`` the bill's daily return, both in percent.
    """
    excess = ff3["ff_mkt_rf"] / 100.0
    cash = ff3["ff_rf"] / 100.0
    return pd.DataFrame({"total": excess + cash, "excess": excess, "cash": cash})


def monthly_asof(frame: pd.DataFrame, series_id: str, index: pd.DatetimeIndex) -> pd.Series:
    """A monthly point-in-time series as known at each session of ``index``.

    Each value enters on its ``available_at`` stamp and is carried forward; a session
    before the first publication is missing.
    """
    one = frame[frame["series_id"] == series_id].sort_values("available_at")
    known = pd.Series(one["value"].to_numpy(float), index=pd.DatetimeIndex(one["available_at"]))
    known = known[~known.index.duplicated(keep="last")]
    return known.reindex(known.index.union(index)).ffill().reindex(index).rename(series_id)


def short_rate(cash: pd.Series) -> pd.Series:
    """Annualised one-month bill yield, in percent, known from each month's first session.

    Ken French spreads the month's bill return evenly over its sessions; the bill is
    bought at the start of the month, so the month's sum is known on its first day.
    """
    month = cash.index.to_period("M")
    return (cash.groupby(month).transform("sum") * 12.0 * 100.0).rename("short_rate")


# ---------------------------------------------------------------------------
# cross-section, robust to missing industries
# ---------------------------------------------------------------------------
def _complete_block(values: np.ndarray) -> np.ndarray:
    """The columns of a window observed on every one of its rows."""
    return values[:, ~np.isnan(values).any(axis=0)]


def average_correlation(returns: pd.DataFrame, *, window: int = 63) -> pd.Series:
    """Mean off-diagonal pairwise correlation over the columns complete in each window.

    Same statistic as `features.crosssection.average_correlation`, which drops every
    row with a missing column and therefore returns nothing while an industry is not
    yet populated (39 to 43 of 49 before 1964).
    """
    values = returns.to_numpy(dtype=float)
    out = np.full(len(values), np.nan)
    for end in range(window, len(values) + 1):
        block = _complete_block(values[end - window:end])
        k = block.shape[1]
        if k < 10:
            continue
        corr = np.corrcoef(block, rowvar=False)
        out[end - 1] = (np.nansum(corr) - k) / (k * (k - 1))
    return pd.Series(out, index=returns.index, name="xs_avg_corr")


def absorption_ratio(returns: pd.DataFrame, *, window: int = 252, factors: int = 5) -> pd.Series:
    """Kritzman's absorption ratio over the columns complete in each window."""
    values = returns.to_numpy(dtype=float)
    out = np.full(len(values), np.nan)
    for end in range(window, len(values) + 1):
        block = _complete_block(values[end - window:end])
        if block.shape[1] < 10:
            continue
        eigenvalues = np.linalg.eigvalsh(np.cov(block, rowvar=False))[::-1]
        total = eigenvalues.sum()
        if total > 0:
            out[end - 1] = eigenvalues[:factors].sum() / total
    return pd.Series(out, index=returns.index, name="xs_absorption")


def breadth(returns: pd.DataFrame, *, window: int = 63) -> pd.Series:
    """Share of the populated industries whose ``window``-session return is positive."""
    growth = np.log1p(returns / 100.0).rolling(window, min_periods=window).sum()
    positive = (growth > 0).astype(float).where(growth.notna())
    return positive.mean(axis=1).rename("xs_breadth_63")


def cross_section(
    industries: pd.DataFrame, size_bm: pd.DataFrame, factors: pd.DataFrame
) -> pd.DataFrame:
    """The cross-sectional block of `features.crosssection.build`, on 1926 panels."""
    out = pd.DataFrame(index=industries.index)
    out["xs_dispersion_ind"] = industries.std(axis=1).rolling(21).mean()
    out["xs_dispersion_szbm"] = size_bm.std(axis=1).rolling(21).mean()
    out["xs_avg_corr"] = average_correlation(industries)
    out["xs_absorption"] = absorption_ratio(industries)
    out["xs_absorption_chg"] = out["xs_absorption"].diff(63)
    out["xs_breadth_63"] = breadth(industries)
    out["xs_ff_smb_63"] = factors["ff_smb"].rolling(63).sum()
    out["xs_ff_hml_63"] = factors["ff_hml"].rolling(63).sum()
    return out


# ---------------------------------------------------------------------------
# the feature matrix
# ---------------------------------------------------------------------------
def raw_features(
    ff3: pd.DataFrame,
    industries: pd.DataFrame,
    size_bm: pd.DataFrame,
    fred: pd.DataFrame,
    *,
    with_indpro: bool = False,
) -> pd.DataFrame:
    """Every feature of the reduced model, before standardisation.

    ``fred`` is the stored point-in-time frame of `scripts/longhist_fetch.py`
    (``fred_aaa``, ``fred_baa``, ``fred_indpro``). The market is Ken French's CRSP
    value-weighted total return, not the S&P 500 price index A' used.
    """
    index = ff3.index
    r = market_returns(ff3)
    price = (1.0 + r["total"]).cumprod().rename(EQUITY)
    panel = price.to_frame()

    momentum = market.momentum(panel)[list(MARKET_FEATURES[:4])]
    volatility = market.volatility(panel)
    shape = asymmetry.build(panel)

    aaa = monthly_asof(fred, "fred_aaa", index)
    baa = monthly_asof(fred, "fred_baa", index)
    cash = short_rate(r["cash"])
    rates = pd.DataFrame(index=index)
    # Fama and French (1989): DEF = Baa - Aaa, TERM = Aaa - one-month bill.
    rates["cre_quality"] = baa - aaa
    rates["cre_quality_chg63"] = rates["cre_quality"].diff(63)
    rates["rat_slope_term"] = aaa - cash
    rates["rat_slope_term_chg63"] = rates["rat_slope_term"].diff(63)
    rates["rat_cash_chg12m"] = cash.diff(252)

    blocks = [momentum, volatility, shape, cross_section(industries, size_bm, ff3), rates]
    if with_indpro:
        level = monthly_asof(fred, "fred_indpro", index)
        blocks.append(pd.DataFrame({"mac_indpro_yoy": level.pct_change(252, fill_method=None),
                                    "mac_indpro_3m": level.pct_change(63, fill_method=None)}))
    columns = list(HEADLINE_FEATURES) + (list(MACRO_FEATURES) if with_indpro else [])
    return pd.concat(blocks, axis=1).reindex(index)[columns]


def standardise(raw: pd.DataFrame, *, min_periods: int = 252) -> pd.DataFrame:
    """Expanding z-score then winsorisation at ±5, as `features.build.build` does."""
    return winsorise(expanding_zscore(raw, min_periods=min_periods))


# ---------------------------------------------------------------------------
# labels: 1 = calm, 0 = stress
# ---------------------------------------------------------------------------
def realised_vol(returns: pd.Series, window: int) -> pd.Series:
    return returns.rolling(window).std() * np.sqrt(PERIODS)


def confirmed(flag: pd.Series, k: int) -> pd.Series:
    """True on a session once ``flag`` has held on each of the last ``k`` sessions."""
    return flag.astype(float).rolling(k, min_periods=k).min().eq(1.0)


def asymmetric_exit(state: pd.Series, returns: pd.Series, *, k: int = 10) -> pd.Series:
    """Idea 2's convalescence rule: the model's stress, released once volatility normalises.

    On a session where the model says stress, the label is calm if the 21-session
    realised volatility of ``returns`` has been below the 63-session one on each of
    the last ``k`` sessions (the term structure of realised volatility is back in its
    normal order). Entry is the model's own; exit is the earlier of the model's exit
    and the release. Where the model is calm the label is the model's. ``k`` and the
    two windows are fixed in advance; nothing is estimated.
    """
    rv21 = realised_vol(returns, 21).reindex(state.index)
    rv63 = realised_vol(returns, 63).reindex(state.index)
    release = confirmed((rv21 < rv63) & rv63.notna(), k)
    out = state.where(~(state.eq(STRESS) & release), CALM)
    return out.where(state.notna()).rename(f"asymmetric_exit_{k}")


def bear_market(price: pd.Series, *, months: int = 24) -> pd.Series:
    """1 when the market sits below its level ``months`` calendar months earlier.

    Daniel and Moskowitz (2016) define the bear state on the cumulative market return
    over the past 24 months. Calendar months, not sessions: the NYSE traded on
    Saturdays until 1952, so 504 sessions were about 20 months then.
    """
    known = price.sort_index()
    past_dates = known.index - pd.DateOffset(months=months)
    # The last session on or before each past date (29 February maps onto the 28th,
    # so past dates can repeat: a positional lookup, not a reindex).
    position = np.searchsorted(known.index.to_numpy(), past_dates.to_numpy(), side="right") - 1
    valid = past_dates >= known.index[0]
    past = known.to_numpy()[np.clip(position, 0, None)]
    bear = pd.Series((known.to_numpy() < past).astype(float), index=known.index)
    return bear.where(valid).rename("bear_24m")


def panic_state(price: pd.Series, returns: pd.Series, *, window: int = 126,
                min_periods: int = 252) -> pd.Series:
    """Daniel and Moskowitz's panic state as a label: stress = bear market x high variance.

    Stress (0) when the market is below its level 24 calendar months earlier **and**
    its variance over the last ``window`` sessions (their variance window) is above
    its own expanding median. Calm (1) otherwise. Causal.
    """
    bear = bear_market(price)
    variance = returns.rolling(window).var()
    median = variance.expanding(min_periods=min_periods).median()
    high = variance > median
    ok = bear.notna() & median.notna() & variance.notna()
    label = pd.Series(np.where((bear == 1.0) & high, STRESS, CALM), index=price.index)
    return label.where(ok).rename("dm_panic")


# ---------------------------------------------------------------------------
# validation against NBER, no return involved
# ---------------------------------------------------------------------------
def recessions(usrec: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """(first day of the first recession month, last day of the last) for each run of 1."""
    flag = usrec.sort_index().astype(int)
    runs = flag.ne(flag.shift()).cumsum()
    out = []
    for _, run in flag.groupby(runs):
        if run.iloc[0] == 1:
            out.append((run.index[0].to_period("M").to_timestamp(),
                        run.index[-1].to_period("M").to_timestamp(how="end").normalize()))
    return out


def daily_reference(usrec: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """The NBER flag of each session's own calendar month (no publication lag: validation)."""
    monthly = usrec.copy()
    monthly.index = pd.DatetimeIndex(monthly.index).to_period("M")
    values = monthly.reindex(index.to_period("M")).to_numpy(dtype=float)
    return pd.Series(values, index=index, name="nber")


def transitions_per_year(label: pd.Series) -> float:
    clean = label.dropna()
    if len(clean) < 2:
        return float("nan")
    changes = int((clean != clean.shift()).sum()) - 1
    years = (clean.index[-1] - clean.index[0]).days / 365.25
    return changes / years


def detection(
    label: pd.Series,
    episodes: list[tuple[pd.Timestamp, pd.Timestamp]],
    *,
    min_sessions: int = 21,
    lead_days: int = 183,
) -> pd.DataFrame:
    """Per recession: stress sessions inside it, detected or not, and entry latency.

    A recession is detected when at least ``min_sessions`` stress sessions fall in its
    NBER months. Latency is the first stress session in [peak - ``lead_days``, trough]
    minus the first day of the first recession month, in calendar days: negative
    means the label was in stress before the recession began. No stress session in
    that window: latency missing.
    """
    rows = []
    stress = label.dropna().eq(STRESS)
    for peak, trough in episodes:
        inside = stress.loc[peak:trough]
        window = stress.loc[peak - pd.Timedelta(days=lead_days):trough]
        first = window[window].index.min() if window.any() else pd.NaT
        rows.append({
            "peak": peak.date(), "trough": trough.date(), "sessions": int(len(inside)),
            "stress_sessions": int(inside.sum()),
            "detected": bool(inside.sum() >= min_sessions),
            "latency_days": float((first - peak).days) if pd.notna(first) else float("nan"),
        })
    return pd.DataFrame(rows)


def kappa(a: pd.Series, b: pd.Series) -> float:
    """Cohen's kappa of two 0/1 labels on their common sessions (stress = 0)."""
    pair = pd.concat([a, b], axis=1).dropna()
    if len(pair) < 100 or pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
        return float("nan")
    return float(cohen_kappa_score(pair.iloc[:, 0].astype(int), pair.iloc[:, 1].astype(int)))


def nber_kappa(label: pd.Series, reference: pd.Series) -> float:
    """Kappa of "stress" against "recession" (both coded 1 for the event)."""
    pair = pd.concat([label.rename("s"), reference.rename("r")], axis=1).dropna()
    if len(pair) < 100 or pair["s"].nunique() < 2 or pair["r"].nunique() < 2:
        return float("nan")
    return float(cohen_kappa_score(pair["r"].astype(int), pair["s"].eq(STRESS).astype(int)))


def kappa_difference_ci(
    a: pd.Series,
    b: pd.Series,
    reference: pd.Series,
    *,
    block: int = 252,
    draws: int = 2000,
    seed: int = 0,
    level: float = 0.95,
) -> dict[str, float]:
    """Moving-block bootstrap of kappa_NBER(a) - kappa_NBER(b), paired by session.

    Blocks of ``block`` consecutive sessions are drawn with replacement until the
    sample length is reached; the same draw serves both labels, so the interval is
    that of the paired difference.
    """
    frame = pd.concat([a.rename("a"), b.rename("b"), reference.rename("r")], axis=1).dropna()
    n = len(frame)
    if n < 2 * block:
        return {"delta": float("nan"), "low": float("nan"), "high": float("nan"), "n": n}
    ra = frame["a"].eq(STRESS).to_numpy(int)
    rb = frame["b"].eq(STRESS).to_numpy(int)
    rr = frame["r"].to_numpy(int)

    def k(x: np.ndarray, y: np.ndarray) -> float:
        po = float((x == y).mean())
        pe = float(x.mean() * y.mean() + (1 - x.mean()) * (1 - y.mean()))
        return (po - pe) / (1.0 - pe) if pe < 1 else float("nan")

    delta = k(ra, rr) - k(rb, rr)
    rng = np.random.default_rng(seed)
    starts_count = int(np.ceil(n / block))
    out = np.empty(draws)
    for i in range(draws):
        starts = rng.integers(0, n - block + 1, size=starts_count)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]
        out[i] = k(ra[idx], rr[idx]) - k(rb[idx], rr[idx])
    tail = (1.0 - level) / 2.0
    return {"delta": float(delta), "low": float(np.nanquantile(out, tail)),
            "high": float(np.nanquantile(out, 1.0 - tail)), "n": int(n)}


# ---------------------------------------------------------------------------
# the test: measures and the verdict
# ---------------------------------------------------------------------------
def sharpe(x: pd.Series) -> float:
    """Annualised Sharpe of a daily excess return; NaN if any value is not finite."""
    v = np.asarray(x, dtype=float)
    if len(v) < 2 or not np.isfinite(v).all() or not v.std(ddof=1) > 0:
        return float("nan")
    return float(v.mean() / v.std(ddof=1) * np.sqrt(PERIODS))


def fold_deltas(a: pd.Series, b: pd.Series, *, folds: int = 5) -> list[float]:
    """Sharpe(a) - Sharpe(b) on ``folds`` consecutive blocks of equal session count."""
    pair = pd.concat([a, b], axis=1)
    edges = np.linspace(0, len(pair), folds + 1).round().astype(int)
    return [sharpe(pair.iloc[s:e, 0]) - sharpe(pair.iloc[s:e, 1])
            for s, e in zip(edges[:-1], edges[1:], strict=True)]


def verdict(
    delta: float,
    mde: float,
    t: float,
    t_critical: float,
    placebo_pct: float,
    folds_positive: int,
    controls: dict[str, float],
    *,
    min_folds: int = 3,
) -> str:
    """The decision of `docs/PRESPEC_LONGHIST.md` §7, stricter than the crisis study's.

    USEFUL needs all of: delta >= MDE; a HAC t of the same sign **and** beyond the
    family's critical value; the rotation placebo at or above its 95th percentile;
    at least ``min_folds`` positive folds of five; and delta above every control in
    ``controls`` (the same coupling driven by each one-line rule). Below the MDE a
    positive delta is UNDERPOWERED, never useful. Any non-finite input: no verdict.
    """
    values = [delta, mde, t, t_critical, placebo_pct, *controls.values()]
    if not all(np.isfinite(values)):
        return "NOT FINITE — no verdict"
    if delta >= mde:
        ok = (np.sign(t) == np.sign(delta) and abs(t) >= t_critical and placebo_pct >= 0.95
              and folds_positive >= min_folds and all(delta > c for c in controls.values()))
        if ok:
            return "USEFUL — all conditions hold"
        return ("NOT SHOWN — above the MDE, fails the t, the placebo, the folds or a "
                "one-line rule")
    if delta > 0:
        return "UNDERPOWERED — positive, below the MDE, never useful"
    if delta > -mde:
        return "NOT USEFUL — does not raise the Sharpe"
    if np.sign(t) == np.sign(delta) and abs(t) >= t_critical and placebo_pct <= 0.05:
        return "HARMFUL — measurably lowers the Sharpe"
    return "NOT USEFUL — negative beyond the MDE, not confirmed"

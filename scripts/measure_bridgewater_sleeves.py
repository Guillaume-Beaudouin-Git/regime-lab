"""Bridgewater study — the sleeves and the state-blind book, measured before the lock.

Re-measures from committed code every figure `docs/PRESPEC_BRIDGEWATER.md` §1.3,
§2 and §5 disclose about the sleeve panel, then measures the object §2 declares:
the blind risk-parity book at its volatility target, net of cost, funded.

**Blind.** No quantity here is conditioned on a quadrant label, and no mean return
of any instrument, sleeve or book is computed or printed. What is printed: dates,
counts, realised standard deviations, correlations of instrument returns, weights,
gross exposure, cap binding, turnover, trading cost and the cash paid on funded
ETFs (the mean of a rate times a weight), and label COUNTS for the fold layout.

Usage:
    .venv/bin/python scripts/measure_bridgewater_sleeves.py [--json PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from regime_lab.config import CACHE, RAW  # noqa: E402
from regime_lab.construction import sleeves as S  # noqa: E402
from regime_lab.extensions.vehicle import cost_class  # noqa: E402
from regime_lab.selection.protocol import standalone_sharpe_threshold  # noqa: E402

RULE = "=" * 78
SPF_DIR = RAW / "spf"

#: What the draft discloses, to be reproduced.
DRAFT = {
    "start": "2003-12-05",
    "years": 23.6,
    "start_6_7": "2007-12-19",
    "years_6_7": 19.4,
    "n": {"headline": 30, "with_credit": 33, "with_fx": 44},
    "resolvable": {"headline": 0.632, "with_credit": 0.712, "with_fx": 0.712},
    "blended_bp": 1.217,
    "fold_quarters": 18.2,
    "fold_sessions": 1187,
    "quarters": 91,
    "sessions": 5938,
    "occupancy_quarters": {0: 13, 1: 28, 2: 25, 3: 25},
}

ASIA = ("^N225", "^HSI", "^AXJO")
EUROPE = ("^GDAXI", "^FTSE", "^AEX", "^FCHI", "^IBEX")
AMERICAS = ("^GSPC", "^NDX", "^RUT", "^GSPTSE")


def section(title: str) -> None:
    print(f"\n{RULE}\n{title}\n{RULE}")


# ---------------------------------------------------------------------------
# 1. panel and §1.3
# ---------------------------------------------------------------------------
def panel_check() -> dict:
    section("1. Which panel")
    stored = pd.read_parquet(CACHE / "trend_universe.parquet")
    m1 = pd.read_parquet(CACHE / "trend_universe_m1.parquet")
    head = S.instruments(S.HEADLINE_SLEEVES)
    differ = [c for c in stored.columns if not stored[c].equals(m1[c])]
    same_head = all(stored[c].equals(m1[c]) for c in head)
    same_credit = all(stored[c].equals(m1[c]) for c in S.CREDIT_SLEEVE)
    print(f"trend_universe     {stored.shape}, {stored.index[0]:%Y-%m-%d} -> "
          f"{stored.index[-1]:%Y-%m-%d}")
    print(f"trend_universe_m1  {m1.shape}, same index: {stored.index.equals(m1.index)}")
    print(f"30 headline columns identical in both: {same_head}")
    print(f"3 credit columns identical in both:    {same_credit}")
    print(f"columns that differ ({len(differ)}): {differ}")
    print("-> the module reads trend_universe_m1: identical on the headline, repaired FX.")
    return {"same_headline": same_head, "same_credit": same_credit, "differ": differ}


def sample_check(prices: pd.DataFrame) -> dict:
    section("2. §1.3 — coverage, sample length, resolvable Sharpe (Lo, 80% power, 5%)")
    out: dict = {}
    full = S.load_prices(S.CONFIGURATIONS["with_fx"])
    cov = S.coverage(full, S.CONFIGURATIONS["with_fx"])
    print(cov.to_string())
    end = full.index[-1]
    print(f"\n{'configuration':12} {'start':>10} {'n':>3} {'sessions':>8} {'yrs=s/252':>9} "
          f"{'yrs cal':>7} {'SR s/252':>8} {'SR cal':>7} {'draft':>6}")
    for name, sleeves in S.CONFIGURATIONS.items():
        start = S.configuration_start(full, sleeves)
        sessions = full.index[(full.index >= start)]
        ys = len(sessions) / 252
        yc = (end - start).days / 365.25
        n = len(S.instruments(sleeves))
        sr_s, sr_c = standalone_sharpe_threshold(ys), standalone_sharpe_threshold(yc)
        print(f"{name:12} {start:%Y-%m-%d} {n:3d} {len(sessions):8d} {ys:9.2f} {yc:7.2f} "
              f"{sr_s:8.3f} {sr_c:7.3f} {DRAFT['resolvable'][name]:6.3f}")
        out[name] = {"start": str(start.date()), "n": n, "sessions": len(sessions),
                     "years_sessions": ys, "years_calendar": yc,
                     "resolvable_sessions": sr_s, "resolvable_calendar": sr_c}

    start = S.configuration_start(prices, S.HEADLINE_SLEEVES)
    sample = prices.loc[start:]
    complete = sample.notna().all(axis=1)
    per_year = len(prices) / ((prices.index[-1] - prices.index[0]).days / 365.25)
    print(f"\nthe panel is a UNION calendar: {per_year:.1f} sessions per calendar year, so "
          f"'years = sessions / 252' overstates length by {per_year / 252 - 1:.1%}")
    print(f"first session with all 30 prices present at once: "
          f"{complete.idxmax():%Y-%m-%d} (the latest first price is {start:%Y-%m-%d})")
    print(f"sessions of the headline sample missing at least one of the 30: "
          f"{int((~complete).sum())} of {len(sample)}")
    missing = sample.isna().sum()
    print(f"missing prices by instrument: {missing[missing > 0].to_dict()}")
    pl = sample["PL=F"].isna()
    runs = pl.ne(pl.shift()).cumsum()[pl]
    lengths = runs.value_counts()
    print(f"PL=F: {len(lengths)} gaps, longest {int(lengths.max())} sessions, last gap ends "
          f"{runs.index[-1]:%Y-%m-%d}")
    bad = (sample <= 0).sum()
    print(f"non-positive prices: {bad[bad > 0].to_dict()} "
          f"(CL=F on {sample.index[(sample['CL=F'] <= 0)][0]:%Y-%m-%d}: no return that "
          f"session or the next)")
    out.update({"sessions_per_calendar_year": per_year,
                "first_complete": str(complete.idxmax().date()),
                "sessions_incomplete": int((~complete).sum()),
                "pl_gaps": int(len(lengths)), "pl_longest_gap": int(lengths.max()),
                "pl_missing": int(missing["PL=F"])})
    return out


def cost_check(names: list[str]) -> dict:
    section("3. §5 — blended round-trip cost over the 30 instruments")
    classes = pd.Series({n: cost_class(n) for n in names})
    print(classes.value_counts().to_string())
    cash_names = list(classes[classes == "cash_vehicle"].index)
    print(f"cash_vehicle instruments in the headline: {cash_names}")
    draft_formula = (1 * 1.0 + 13 * 1.5 + 12 * 1.0 + 4 * 1.0) / 30
    print(f"\ndraft formula (TIP and AGG at 1.0 bp): {draft_formula:.3f} bp  <- reproduces 1.217")
    out = {"draft_formula": draft_formula}
    for column in S.COST_COLUMNS:
        value = S.blended_cost_bps(names, column)
        out[f"equal_weight_{column}"] = value
        print(f"vehicle.cost_bps, equal weight, {column:12}: {value:.3f} bp")
    print("the draft priced TIP and AGG at the fixed-income 1.0 bp; vehicle.py prices both at "
          "the 7.5 bp cash-vehicle rate (neither has a futures contract).")
    return out


# ---------------------------------------------------------------------------
# 4. the declared blind book
# ---------------------------------------------------------------------------
def rebalance_dates(sessions: pd.DatetimeIndex) -> tuple[list[pd.Timestamp], str, object]:
    """SPF release stamps from the quadrant module if on disk, else the draft's rule."""
    try:
        from regime_lab.construction import quadrant as Q

        rgdp = Q.read_spf(SPF_DIR / Q.SPF_LEVEL_FILES["rgdp"][0], Q.SPF_LEVEL_FILES["rgdp"][1])
        cpi = Q.read_spf(SPF_DIR / Q.SPF_LEVEL_FILES["cpi"][0], Q.SPF_LEVEL_FILES["cpi"][1])
        release = Q.read_release_dates(SPF_DIR / Q.RELEASE_DATES_CSV)
        panel = Q.surprise_panel(rgdp, cpi)
        stamps = Q.availability(panel.index, rule="release", release_dates=release)
        labels = Q.quarterly_labels(panel, stamps)
        source = "quadrant module, SPF release dates (rule='release')"
    except (ImportError, OSError, AttributeError, KeyError, TypeError, ValueError) as err:
        print(f"quadrant module or SPF files unavailable ({type(err).__name__}: {err}); "
              "falling back to the draft's stamp rule")
        quarters = pd.period_range("2003Q1", "2026Q2", freq="Q")
        stamp = quarters.start_time + pd.DateOffset(months=4, days=14)
        labels = None
        dates = [d for d in stamp if sessions[0] <= d <= sessions[-1]]
        return dates, "draft rule: start(q) + 4 months + 14 days", labels
    dates = [pd.Timestamp(d) for d in labels["available_at"]
             if sessions[0] <= pd.Timestamp(d) <= sessions[-1]]
    return dates, source, labels


def diagnostics_row(name: str, book: S.SleeveBook, cash: pd.Series) -> dict:
    d = S.book_diagnostics(book, cash=cash)
    unscaled = book.unscaled_returns.loc[book.live]
    d["unscaled_sd"] = float(unscaled.std(ddof=1) * np.sqrt(252))
    d["name"] = name
    return d


def print_rows(rows: list[dict]) -> None:
    print(f"\n{'reading':34} {'live':>10} {'sess':>5} {'sd':>6} {'sd wk':>6} {'unsc':>5} "
          f"{'cap%':>5} {'g med':>5} {'g p95':>5} {'g max':>5} {'TO':>5} {'TOu':>5}")
    for d in rows:
        print(f"{d['name']:34} {d['first']:%Y-%m-%d} {d['sessions']:5d} "
              f"{d['realised_sd']:6.2%} {d['realised_sd_weekly']:6.2%} {d['unscaled_sd']:5.2%} "
              f"{d['cap_share']:5.1%} {d['gross_median']:5.2f} {d['gross_p95']:5.2f} "
              f"{d['gross_max']:5.2f} {d['turnover']:5.2f} {d['turnover_unscaled']:5.2f}")
    print(f"\n{'reading':34} " + " ".join(f"{c:>18}" for c in S.COST_COLUMNS)
          + f" {'funding':>14}")
    print(f"{'':34} " + " ".join(f"{'%/yr  SR(real)':>18}" for _ in S.COST_COLUMNS)
          + f" {'%/yr  SR':>14}")
    for d in rows:
        cells = " ".join(
            f"{d[f'cost_{c}']:8.3%} {d[f'cost_sharpe_{c}']:9.4f}" for c in S.COST_COLUMNS
        )
        print(f"{d['name']:34} {cells} {d['funding']:7.3%} {d['funding_sharpe']:6.3f}")
    print(f"\n{'reading':34} bp per unit traded on held weights, by cost column:")
    for d in rows:
        print(f"{d['name']:34} " + " ".join(
            f"{c} {d[f'traded_bps_{c}']:.2f}" for c in S.COST_COLUMNS))


def blind_book_check(excess: pd.DataFrame, cash: pd.Series, dates: list[pd.Timestamp],
                     start: pd.Timestamp) -> tuple[dict, S.SleeveBook]:
    section("4. The declared blind book (state-blind; no mean read)")
    print("within sleeve: inverse volatility; across sleeves: ERC on the sleeve covariance;")
    print("estimation: trailing 252 sessions inside the sample, at each stamp; held from the")
    print("next session; 10% target on 63 sessions lagged one, multiplier capped at 3.0;")
    print("costs on |Δw| of held weights; ETFs funded at rate_cash_3m (available_at).")
    readings = {
        "DECLARED erc, 252, cap 3": dict(method="erc", lookback=252),
        "inverse_vol, 252, cap 3": dict(method="inverse_vol", lookback=252),
        "erc, 63, cap 3": dict(method="erc", lookback=63),
        "erc, expanding, cap 3": dict(method="erc", lookback=None),
        "erc, 252, weekly estimates, cap 3": dict(method="erc", lookback=252,
                                                  frequency="weekly"),
        "erc, 252, no cap": dict(method="erc", lookback=252, cap=np.inf),
    }
    rows, books = [], {}
    for name, kw in readings.items():
        book, within, across = S.blind_book(excess, S.HEADLINE_SLEEVES, dates, start=start, **kw)
        books[name] = (book, within, across)
        rows.append(diagnostics_row(name, book, cash))
    print_rows(rows)

    book, within, across = books["DECLARED erc, 252, cap 3"]
    declared = rows[0]
    print("\n§5's cost kill (0.10 Sharpe), re-derived on held weights at the realised sd")
    print("(the draft: 82 extra round trips a year at headline, 13.3 at all_cash):")
    kill = {}
    for column in S.COST_COLUMNS:
        bp = declared[f"traded_bps_{column}"]
        kill[column] = 0.10 * declared["realised_sd"] * 10_000 / bp
        print(f"  {column:12} {bp:5.2f} bp per unit traded -> {kill[column]:5.1f} extra units "
              "of held-weight turnover a year")
    weekly = books["erc, 252, weekly estimates, cap 3"][2].dropna()
    print("\nsleeve weights, median, daily vs weekly estimates:")
    for s in weekly.columns:
        print(f"  {s:18} daily {across[s].median():.3f}  weekly {weekly[s].median():.3f}")
    held = across.dropna()
    print(f"\nsleeve weights at the {len(held)} rebalances (unscaled, rows sum to one):")
    print(held.describe().loc[["min", "50%", "max"]].round(3).to_string())
    iv = books["inverse_vol, 252, cap 3"][2].dropna()
    print("inverse-vol sleeve weights, median: "
          + ", ".join(f"{k} {v:.3f}" for k, v in iv.median().items()))
    inner = within.dropna(how="all")
    zero = (inner == 0).sum()
    print(f"instrument-rebalances excluded for too few returns: {zero[zero > 0].to_dict()}")
    print("median within-sleeve weight: " + ", ".join(
        f"{k} {v:.3f}" for k, v in inner.median().items() if k in ("SHY", "AGG", "TLT", "IEF")))

    # How much of the book's variance dispersion survives the target, label-free.
    stamps = pd.DatetimeIndex(dates)
    live = book.live
    q = pd.Series(stamps.searchsorted(live, side="left"), index=live)
    by_q = pd.DataFrame({"t": book.returns.loc[live], "u": book.unscaled_returns.loc[live],
                         "q": q}).groupby("q")
    counts = by_q.size()
    keep = counts[counts >= 40].index
    log_var_t = np.log(by_q["t"].var(ddof=1).loc[keep])
    log_var_u = np.log(by_q["u"].var(ddof=1).loc[keep])
    cap_q = pd.Series(book.cap_binds.loc[live].to_numpy(), index=live).groupby(q).mean().loc[keep]
    print(f"\nquarter-level log variance across {len(keep)} complete quarters (unconditional):")
    print(f"  unscaled book: sd {log_var_u.std(ddof=1):.3f}, range "
          f"{log_var_u.max() - log_var_u.min():.3f}")
    print(f"  targeted book: sd {log_var_t.std(ddof=1):.3f}, range "
          f"{log_var_t.max() - log_var_t.min():.3f}")
    print(f"  corr(quarter cap share, quarter log var of the targeted book): "
          f"{np.corrcoef(cap_q, log_var_t)[0, 1]:+.3f}")
    print(f"  quarters with the cap binding on >50% of sessions: {(cap_q > 0.5).sum()} of "
          f"{len(keep)}")
    out = {r["name"]: {k: (str(v.date()) if isinstance(v, pd.Timestamp) else v)
                       for k, v in r.items()} for r in rows}
    out["cost_kill_turnover"] = kill
    out["quarter_log_var_sd"] = {"unscaled": float(log_var_u.std(ddof=1)),
                                 "targeted": float(log_var_t.std(ddof=1)),
                                 "quarters": int(len(keep))}
    out["sleeve_weight_median"] = held.median().to_dict()
    return out, book


# ---------------------------------------------------------------------------
# 5. time zones
# ---------------------------------------------------------------------------
def timezone_check(returns: pd.DataFrame, book: S.SleeveBook, start: pd.Timestamp) -> dict:
    section("5. Time zones — same-day vs lagged correlation with ^GSPC (label-free)")
    r = returns.loc[start:, list(S.HEADLINE_SLEEVES["equity"])]
    r = r - r.mean()  # correlations are mean-free; demeaning keeps the weekly sums so too
    us = r["^GSPC"]
    weekly = r.groupby(pd.Grouper(freq="W-FRI")).sum(min_count=1)
    print(f"{'index':8} {'same':>6} {'i(t+1)~us(t)':>13} {'i(t)~us(t+1)':>13} "
          f"{'sum':>6} {'weekly':>7}")
    out = {}
    for name in r.columns:
        if name == "^GSPC":
            continue
        same = r[name].corr(us)
        lead = r[name].shift(-1).corr(us)
        lag = r[name].corr(us.shift(-1))
        wk = weekly[name].corr(weekly["^GSPC"])
        out[name] = {"same": same, "next": lead, "us_next": lag, "sum": same + lead + lag,
                     "weekly": wk}
        print(f"{name:8} {same:6.3f} {lead:13.3f} {lag:13.3f} {same + lead + lag:6.3f} "
              f"{wk:7.3f}")

    # Variance ratio of sleeves: weekly variance against summed daily variance.
    held = book.unscaled_weights.loc[book.live]
    x = returns.reindex(index=held.index, columns=held.columns).fillna(0.0)
    parts = {
        "equity sleeve": list(S.HEADLINE_SLEEVES["equity"]),
        "  US + Canada only": list(AMERICAS),
        "  Europe only": list(EUROPE),
        "  Asia only": list(ASIA),
        "duration sleeve": list(S.HEADLINE_SLEEVES["duration"]),
        "commodity sleeve": list(S.HEADLINE_SLEEVES["commodity"]),
        "whole unscaled book": list(held.columns),
    }
    print("\nvariance ratio = var(weekly sum) / (sessions per week x var(daily)); >1 means")
    print("daily returns understate risk (asynchronous closes, or autocorrelation):")
    per_week = x.groupby(pd.Grouper(freq="W-FRI")).size()
    per_week = float(per_week[per_week > 0].mean())
    vr = {}
    for label, cols in parts.items():
        w = held[cols]
        daily = (w * x[cols]).sum(axis=1) / w.sum(axis=1)
        daily = daily - daily.mean()
        wk = daily.groupby(pd.Grouper(freq="W-FRI")).sum(min_count=1).dropna()
        vr[label.strip()] = float(wk.var(ddof=1) / (per_week * daily.var(ddof=1)))
        print(f"  {label:22} {vr[label.strip()]:.3f}")
    eq = (held[parts["equity sleeve"]] * x[parts["equity sleeve"]]).sum(axis=1)
    eq = eq - eq.mean()
    other = {s: (held[list(m)] * x[list(m)]).sum(axis=1)
             for s, m in S.HEADLINE_SLEEVES.items() if s != "equity"}
    other = {s: v - v.mean() for s, v in other.items()}
    print("\ncorrelation of the equity sleeve with each other sleeve, daily vs weekly:")
    cross = {}
    for s, series in other.items():
        d = eq.corr(series)
        wk = eq.groupby(pd.Grouper(freq="W-FRI")).sum().corr(
            series.groupby(pd.Grouper(freq="W-FRI")).sum())
        cross[s] = {"daily": d, "weekly": wk}
        print(f"  {s:18} daily {d:+.3f}  weekly {wk:+.3f}")
    return {"pairs": out, "variance_ratio": vr, "cross": cross, "sessions_per_week": per_week}


# ---------------------------------------------------------------------------
# 6. dividends
# ---------------------------------------------------------------------------
def dividend_check(returns: pd.DataFrame, cash: pd.Series, book: S.SleeveBook) -> dict:
    section("6. Dividends (draft killer 5) — inventory and size, label-free")
    inventory = {
        "data/cache/shu_sp500_tr.parquet": "S&P 500 price return + Shiller D/P accrual (daily)",
        "data/cache/shu_sp500.parquet": "S&P 500 price and 3-month bill",
        "data/raw/prices/cross_asset.parquet": "eq_us_large, eq_us_small, eq_us_tech, eq_jp "
                                               "prices, no dividends",
    }
    for k, v in inventory.items():
        print(f"  {k:40} {'present' if (REPO / k).exists() else 'ABSENT':8} {v}")
    print("  nothing on disk carries a dividend for the 11 non-S&P indices;")
    print("  ^GDAXI is the DAX PERFORMANCE index (dividends reinvested): total return already.")
    out: dict = {}
    path = CACHE / "shu_sp500_tr.parquet"
    if not path.exists():
        print("  shu_sp500_tr.parquet absent: nothing measured")
        return out
    tr = pd.read_parquet(path)["total_return"]
    live = book.live
    common = tr.index.intersection(returns.index)
    accrual = (tr.loc[common] - returns.loc[common, "^GSPC"]).dropna()
    yield_annual = accrual * 252
    steps = int((yield_annual.round(10).diff().abs() > 1e-9).sum())
    sample = yield_annual.loc[live[0]:live[-1]]
    rf = cash.loc[live] * 252
    print(f"\nShiller S&P 500 yield recovered from the cache: {len(accrual)} sessions, "
          f"{steps} changes of level (a monthly step, as built)")
    print(f"  over the blind book's live sample {live[0]:%Y-%m-%d} -> {live[-1]:%Y-%m-%d}:")
    print(f"  mean dividend yield {sample.mean():.3%}, mean cash rate {rf.mean():.3%}, "
          f"mean (d - r) {sample.mean() - rf.mean():+.3%}")
    for a, b in (("2005", "2008"), ("2009", "2015"), ("2016", "2021"), ("2022", "2026")):
        d, c = sample.loc[a:b].mean(), rf.loc[a:b].mean()
        print(f"    {a}-{b}: d {d:.2%}  r {c:.2%}  d - r {d - c:+.2%}")
    x = returns.loc[live, "^GSPC"].dropna()
    with_div = x + accrual.reindex(x.index).ffill()
    rel = with_div.std(ddof=1) / x.std(ddof=1) - 1.0
    print(f"  sd(^GSPC + accrual) / sd(^GSPC) - 1 = {rel:+.2e}: a smooth accrual leaves every")
    print("  variance, every risk-parity weight and the statistic D unchanged; it moves means.")
    equity = book.weights.loc[live, list(S.HEADLINE_SLEEVES["equity"])].sum(axis=1)
    sd = book.returns.loc[live].std(ddof=1) * np.sqrt(252)
    gap = sample.mean() - rf.mean()
    print(f"  held equity notional in the blind book: mean {equity.mean():.3f}, median "
          f"{equity.median():.3f}")
    print("  if all 12 indices carried the S&P's d - r, the blind book's excess return would")
    print(f"  move by {equity.mean() * gap:+.3%}/yr = {equity.mean() * gap / sd:+.4f} Sharpe; "
          f"adding d alone (no r): {equity.mean() * sample.mean() / sd:+.4f} Sharpe")
    out.update({"mean_yield": float(sample.mean()), "mean_cash": float(rf.mean()),
                "sd_change": float(rel), "equity_notional_mean": float(equity.mean()),
                "sharpe_shift_d_minus_r": float(equity.mean() * gap / sd),
                "sharpe_shift_d": float(equity.mean() * sample.mean() / sd)})
    return out


# ---------------------------------------------------------------------------
# 7. walk-forward layout
# ---------------------------------------------------------------------------
def quarter_windows(labels: pd.DataFrame, sessions: pd.DatetimeIndex) -> pd.DataFrame:
    """Each labelled quarter's holding window on ``sessions``: from the session after its
    stamp to the session of the next stamp. Label COUNTS only."""
    table = labels.sort_values("available_at").drop_duplicates("available_at", keep="last")
    stamps = pd.DatetimeIndex(table["available_at"]).astype(sessions.dtype)
    first = stamps.searchsorted(sessions, side="left") - 1
    held = pd.Series(first, index=sessions)
    rows = []
    for k in range(len(table)):
        on = held.index[held.to_numpy() == k]
        if len(on):
            rows.append({"quarter": str(table.index[k]), "label": int(table["label"].iloc[k]),
                         "first": on[0], "last": on[-1], "sessions": len(on)})
    return pd.DataFrame(rows)


def fold_layout(labels: pd.DataFrame | None, sample_start: pd.Timestamp,
                live_start: pd.Timestamp, sessions: pd.DatetimeIndex) -> dict:
    section("7. Walk-forward layout (label counts only)")
    n = len(sessions[sessions >= sample_start])
    print(f"draft: 5 folds of {DRAFT['fold_quarters']} quarters and {DRAFT['fold_sessions']} "
          f"sessions = {DRAFT['quarters']} quarters / 5 and {DRAFT['sessions']} sessions / 5.")
    print(f"measured: {n} sessions from {sample_start:%Y-%m-%d}; {n / 5:.1f} per fold if the "
          "WHOLE sample is tested, which leaves fold 1 no training window. That is a K-fold")
    print("split, not an expanding walk-forward.")
    live = sessions[sessions >= live_start]
    print(f"the blind book is live from {live_start:%Y-%m-%d} (252 sessions of estimation, a "
          f"stamp, then 63 of volatility warm-up): {len(live)} sessions.")
    out: dict = {"sessions_from_start": n, "live_sessions": len(live)}
    if labels is None:
        occ = DRAFT["occupancy_quarters"]
        print(f"no labels on disk: the draft's occupancy {occ} gives totals, not timing, so")
        print("no first training window can be derived. Re-run once the quadrant module lands.")
        return out
    windows = quarter_windows(labels, sessions[sessions >= sample_start])
    inside = windows[windows["first"] >= sample_start].reset_index(drop=True)
    cells = inside["label"].value_counts().sort_index().to_dict()
    print(f"\nlabelled quarters held inside the sample: {len(inside)} (the first is held from "
          f"{inside['first'].iloc[0]:%Y-%m-%d} for {inside['sessions'].iloc[0]} sessions, stamped "
          f"before the sample); cell counts {cells}")
    off = [d for d in labels["available_at"] if sample_start <= pd.Timestamp(d) <= sessions[-1]
           and pd.Timestamp(d) not in sessions]
    print(f"stamps inside the sample that are not sessions of the panel: {len(off)} {off[:5]}")
    out["quarters_in_sample"] = int(len(inside))
    out["proposals"] = {}
    for m in (2, 3, 4, 5, 6):
        counts = pd.get_dummies(inside["label"]).reindex(columns=range(4), fill_value=0).cumsum()
        ok = (counts >= m).all(axis=1)
        if not ok.any():
            print(f"  min {m} quarters per cell: never reached")
            continue
        k0 = int(ok.idxmax())
        train_end = inside["last"].iloc[k0]
        test = inside.iloc[k0 + 1:]
        test = test[test["first"] >= live_start]
        if len(test) < 5:
            print(f"  min {m}: fewer than five test quarters left")
            continue
        per = np.array_split(np.arange(len(test)), 5)
        folds = []
        for j, idx in enumerate(per, start=1):
            block = test.iloc[idx]
            train_q = inside[inside["last"] < block["first"].iloc[0]]
            folds.append({
                "fold": j, "test_first": str(block["first"].iloc[0].date()),
                "test_last": str(block["last"].iloc[-1].date()), "quarters": len(block),
                "sessions": int(block["sessions"].sum()),
                "train_cells": train_q["label"].value_counts().reindex(range(4), fill_value=0)
                .tolist(),
                "test_cells": block["label"].value_counts().reindex(range(4), fill_value=0)
                .tolist(),
            })
        tested = sum(f["sessions"] for f in folds)
        mde = 2.8016 * np.sqrt(1.0 / (2.0 * tested / 63.0))
        print(f"\n  min {m} quarters per cell: first training window ends {train_end:%Y-%m-%d} "
              f"({(train_end - sample_start).days / 365.25:.2f} years); tested "
              f"{sum(f['quarters'] for f in folds)} quarters, {tested} sessions; the draft's own "
              f"variance-MDE formula on the tested sessions: {mde:.3f} (draft 0.204 on all)")
        for f in folds:
            print(f"    fold {f['fold']}: {f['test_first']} -> {f['test_last']}  "
                  f"{f['quarters']:2d} q {f['sessions']:5d} s  train cells {f['train_cells']}  "
                  f"test cells {f['test_cells']}")
        out["proposals"][m] = {"train_end": str(train_end.date()), "folds": folds,
                               "tested_sessions": tested, "draft_formula_mde": float(mde)}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", type=Path, default=None, help="write the figures here")
    args = parser.parse_args()

    results: dict = {"panel": panel_check()}
    prices = S.load_prices(S.HEADLINE_SLEEVES)
    results["sample"] = sample_check(prices)
    names = S.instruments(S.HEADLINE_SLEEVES)
    results["cost"] = cost_check(names)

    returns = S.price_returns(prices)
    cash = S.cash_rate(prices.index)
    excess = S.excess_returns(returns, cash)
    start = S.configuration_start(prices, S.HEADLINE_SLEEVES)
    dates, source, labels = rebalance_dates(prices.index[prices.index >= start])
    print(f"\nrebalance stamps: {len(dates)}, {dates[0]:%Y-%m-%d} -> {dates[-1]:%Y-%m-%d}; "
          f"source: {source}")
    results["rebalance"] = {"n": len(dates), "source": source}
    results["book"], book = blind_book_check(excess, cash, dates, start)
    results["timezones"] = timezone_check(returns, book, start)
    results["dividends"] = dividend_check(returns, cash, book)
    results["folds"] = fold_layout(labels, start, book.live[0], prices.index)
    if args.json:
        args.json.write_text(json.dumps(results, indent=2, default=str))
        print(f"\nwritten: {args.json}")


if __name__ == "__main__":
    main()

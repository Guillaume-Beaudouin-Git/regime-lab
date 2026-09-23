"""Idea 6: does the sparse jump state improve a risk forecast on 46 markets?

The protocol is `docs/PRESPEC_RISQUE.md`. EVERYTHING BELOW IS FIXED, AND COMMITTED,
BEFORE ANY LOSS OR RETURN CONDITIONED ON THE STATE IS READ.

The question. The state carries information about forward variance beyond a volatility
quantile (+3.93 points of R2, `docs/RESULTS_FINAL.md`), and almost none beyond the VIX on
the S&P 500 (+0.20 point, `docs/RESULTS_CRISE.md` section 5). Most of the 46 markets of
the trend panel have no implied-volatility index in this repository. Does the state,
added to a standard volatility forecast estimated on the past alone, forecast the
variance of those markets better? And does a better forecast size a book better?

The forecasts (per market, `extensions.risque`)
    target  log of the realised variance over the next 21 sessions (log returns)
    HAR     log target ~ 1 + log mean r^2 over 5, 22 and 66 sessions
    HARVIX  HAR + log VIX
    + state   one dummy, 1 when A' sparse jump (filtered, `states.parquet`) is in stress
    + median  one dummy, 1 when the ^GSPC 21-session volatility is above its expanding
              median (`volatility_quantile_placebo`)
    + 80th    the same above the expanding 80th percentile (`crisis.volatility_tail_rule`)
    Every coefficient used at the close of t is an OLS on rows s <= t - 21 only, from
    2002-04-01 (the first out-of-sample state), at least 504 rows; refitted every session.
    Every arm trains on the same rows. Variance forecast exp(x'b + s2/2).

The panels
    P1  43 markets without an own implied-volatility index (the 46 minus ^GSPC, ^NDX,
        ^RUT), state beyond HAR
    P2  the same 43, state beyond HARVIX
    P3  the 3 US equity indices, state beyond HARVIX (where the VIX applies)
    C1-C8  four classes (non-US equities 10, commodities 13, currencies 11, bonds and
        credit 9) x the two baselines
    E1  the 46-market trend book, instruments sized by HAR+state against HAR
    E2  a long-only inverse-volatility book on the 35 non-currency markets, same pair

THE DECISION, WRITTEN BEFORE THE NUMBER
    d_t    cross-sectional mean over the panel's markets of QLIKE(base) - QLIKE(base+state)
    delta  mean of d_t over the evaluation sessions (positive: the state helps)
    MDE    blinded stationary block bootstrap SE of the mean of d_t, blocks 21/63/126,
           the largest, times z(1 - alpha/2) + z(0.80); alpha 0.05/3 for P, 0.05/8 for C
    t      HAC t of d_t, 21 lags (overlapping 21-session targets)
    USEFUL  delta >= MDE, t of the same sign, delta >= 95th pct of a 400-draw rotation
            placebo of the state, delta above the same gain of each volatility rule, and
            delta > 0 in at least 3 of 5 contiguous folds
    the rest of the ladder as in `scripts/run_crisis_coupling.py`; see `risque.verdict`.
    E1/E2: delta = Sharpe(state-sized) - Sharpe(HAR-sized), excess of cash, zero cost,
           MDE `selection.protocol.blinded_mde` at alpha 0.05/2, t HAC 6, same placebo,
           controls: the book sized by HAR+median, HAR+80th and HARVIX.
    A reading with a non-finite value gives no verdict.

Discipline, as in `scripts/run_crisis_coupling.py`: run plain, the script measures the
instrument and prints no loss or return that involves the state. `--read` prints the
results and logs the rows of family `risque_forecast` to `data/trials.parquet`; run it
once.

Usage:
    .venv/bin/python scripts/run_risque.py           # instrument only
    .venv/bin/python scripts/run_risque.py --read    # the reading, once
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from regime_lab.analysis import trials
from regime_lab.config import CACHE, RAW
from regime_lab.evaluation.predictive import volatility_quantile_placebo
from regime_lab.extensions import crisis, risque
from regime_lab.extensions.vehicle import vol_targeted_book
from regime_lab.selection.protocol import blinded_mde, mde_at, paired_hac_t

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_crisis_coupling import series, sha256  # noqa: E402
from run_m3_evaluation import cash_rate_daily, funding_charge  # noqa: E402

RULE = "=" * 78
FAMILY = "risque_forecast"
STATE_COLUMN = "A' sparse jump"
TRAIN_START = pd.Timestamp("2002-04-01")
HORIZON = risque.HORIZON
MIN_N = 30
FORECAST_LAGS = 21
BOOK_LAGS = 6
BLOCKS = (21, 63, 126)
DRAWS = 2000
ROTATIONS = 400
MIN_SHIFT = 252
FOLDS = 5
FOLDS_NEEDED = 3
ALPHA_PRIMARY = 0.05 / 3
ALPHA_CLASS = 0.05 / 8
ALPHA_BOOK = 0.05 / 2
HOLM_ALPHA = 0.05
PLACEBO_SEED = 20260925

US_EQUITY = ("^GSPC", "^NDX", "^RUT")
CLASSES = {
    "equity_non_us": ("^AEX", "^AXJO", "^FCHI", "^FTSE", "^GDAXI", "^GSPTSE", "^HSI",
                      "^IBEX", "^N225", "^STOXX50E"),
    "commodity": ("CL=F", "CT=F", "GC=F", "HG=F", "HO=F", "KC=F", "NG=F", "PL=F", "SB=F",
                  "SI=F", "ZC=F", "ZS=F", "ZW=F"),
    "currency": ("AUDUSD=X", "CAD=X", "CHF=X", "DX-Y.NYB", "EURUSD=X", "GBPUSD=X", "JPY=X",
                 "MXN=X", "NOK=X", "NZDUSD=X", "SEK=X"),
    "bond_credit": ("AGG", "BWX", "EMB", "HYG", "IEF", "LQD", "SHY", "TIP", "TLT"),
}
NON_US = tuple(m for members in CLASSES.values() for m in members)
RP_MEMBERS = US_EQUITY + CLASSES["equity_non_us"] + CLASSES["commodity"] + CLASSES["bond_credit"]

BASES = ("har", "harvix")
DUMMIES = ("state", "median", "80th")
ARMS = tuple(b for b in BASES) + tuple(f"{b}+{d}" for b in BASES for d in DUMMIES)
BOOK_ARMS = ("har", "har+state", "har+median", "har+80th", "harvix")

# (code, panel label, members, base, alpha)
FORECAST_TESTS = (
    ("P1", "non_us_43", NON_US, "har", ALPHA_PRIMARY),
    ("P2", "non_us_43", NON_US, "harvix", ALPHA_PRIMARY),
    ("P3", "us_equity_3", US_EQUITY, "harvix", ALPHA_PRIMARY),
    *(
        (f"C{2 * i + j + 1}", name, members, base, ALPHA_CLASS)
        for i, (name, members) in enumerate(CLASSES.items())
        for j, base in enumerate(BASES)
    ),
)
BOOK_TESTS = (("E1", "trend_46"), ("E2", "risk_parity_35"))
INPUTS = (
    CACHE / "trend_universe_m1.parquet",
    RAW / "prices" / "cross_asset.parquet",
    RAW / "macro" / "rate_cash_3m.parquet",
    CACHE / "states.parquet",
)


# ---------------------------------------------------------------------------
# data and forecasts
# ---------------------------------------------------------------------------
def ns(index: pd.Index) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(index).as_unit("ns")


def asof(labels: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """The last value stamped at or before each date of ``index``."""
    known = labels.dropna().sort_index()
    known.index = ns(known.index)
    return known.reindex(known.index.union(index)).ffill().reindex(index)


def stress_dummy(labels: pd.Series) -> np.ndarray:
    """1 on stress (label 0), 0 on calm, NaN where the label is unknown."""
    return labels.eq(0.0).astype(float).where(labels.notna()).to_numpy()


def build() -> dict:
    prices = pd.read_parquet(CACHE / "trend_universe_m1.parquet")
    prices.index = ns(prices.index)
    index = prices.index
    r = risque.log_returns(prices)
    target = risque.forward_variance(r)
    y = np.log(target.where(target > 0))

    path = RAW / "prices" / "cross_asset.parquet"
    spx = series(path, "eq_us_large")
    vix = series(path, "vol_vix")
    simple = spx.pct_change()
    states = pd.read_parquet(CACHE / "states.parquet")[STATE_COLUMN]
    state = asof(states, index)
    labels = {
        "state": state,
        "median": asof(volatility_quantile_placebo(simple), index),
        "80th": asof(crisis.volatility_tail_rule(simple), index),
    }
    dummies = {k: stress_dummy(v) for k, v in labels.items()}
    log_vix = np.log(asof(vix, index)).to_numpy()
    eligible = np.asarray(index >= TRAIN_START)

    f = {arm: {} for arm in (*ARMS, "ewma", "ewma+state")}
    store = {}
    for name in prices.columns:
        har = risque.har_components(r[name]).to_numpy()
        ewma = risque.ewma_log_variance(r[name]).to_numpy()
        yi = y[name].to_numpy()
        regs = (np.isfinite(har).all(axis=1) & np.isfinite(log_vix)
                & np.all([np.isfinite(d) for d in dummies.values()], axis=0))
        rows = eligible & np.isfinite(yi) & regs
        one = np.ones(len(index))
        designs = {"har": np.column_stack([one, har]),
                   "harvix": np.column_stack([one, har, log_vix])}
        mom = {}
        for base, x in designs.items():
            mom[base] = risque.moments(yi, x, rows, horizon=HORIZON)
            f[base][name] = risque.forecast(mom[base], usable=regs)
            for d in DUMMIES:
                f[f"{base}+{d}"][name] = risque.forecast_with_dummy(
                    mom[base], dummies[d], usable=regs, base=f[base][name])
        rows_e = rows & np.isfinite(ewma)
        me = risque.moments(yi, np.column_stack([one, ewma]), rows_e, horizon=HORIZON)
        usable_e = regs & np.isfinite(ewma)
        f["ewma"][name] = risque.forecast(me, usable=usable_e)
        f["ewma+state"][name] = risque.forecast_with_dummy(
            me, dummies["state"], usable=usable_e, base=f["ewma"][name])
        squared = r[name] ** 2
        stale = np.column_stack([squared.rolling(w, min_periods=w).mean().eq(0.0).to_numpy()
                                 for w in risque.HAR_WINDOWS])
        store[name] = {"har": mom["har"], "harvix": mom["harvix"], "regs": regs,
                       "rows": rows, "zero_windows": int(stale.any(axis=1).sum())}
    forecasts = {arm: pd.DataFrame(v, index=index)[prices.columns] for arm, v in f.items()}

    defined = np.all([forecasts[a].notna().to_numpy() for a in ARMS], axis=0)
    common = defined & target.notna().to_numpy()
    common_df = pd.DataFrame(common, index=index, columns=prices.columns)
    count = common_df[list(NON_US)].sum(axis=1)
    end_pos = int(np.flatnonzero(count.to_numpy() >= MIN_N).max())
    below = np.flatnonzero(count.to_numpy()[: end_pos + 1] < MIN_N)
    start_pos = int(below.max()) + 1 if below.size else 0
    evaluation = index[start_pos: end_pos + 1]

    losses = {arm: risque.qlike(target, forecasts[arm]).where(common_df)
              for arm in forecasts}
    return {
        "prices": prices, "index": index, "returns": r, "target": target,
        "forecasts": forecasts, "losses": losses, "common": common_df,
        "defined": pd.DataFrame(defined, index=index, columns=prices.columns),
        "evaluation": evaluation, "dummies": dummies, "labels": labels,
        "log_vix": log_vix, "store": store, "spx": spx,
    }


def differential(p: dict, members, base: str, alt: str,
                 losses: dict | None = None) -> pd.Series:
    """d_t: cross-sectional mean of loss(base) - loss(alt) over ``members``."""
    L = p["losses"] if losses is None else losses
    d = (L[base][list(members)] - L[alt][list(members)]).loc[p["evaluation"]]
    return d.mean(axis=1)


def book_legs(p: dict, sigma: dict[str, pd.DataFrame]) -> dict[str, dict[str, pd.Series]]:
    """Excess-of-cash book returns, zero cost, for each sizing arm and each book."""
    prices = p["prices"]
    simple = prices.pct_change()
    rate = cash_rate_daily(p["index"])
    out = {"trend_46": {}, "risk_parity_35": {}}
    weights = {"trend_46": {}, "risk_parity_35": {}}
    for arm, s in sigma.items():
        raw = risque.trend_raw_weights(prices, s)
        gross, w = risque.targeted_book(raw, simple)
        out["trend_46"][arm] = gross - funding_charge(w, rate, mode="signed")
        weights["trend_46"][arm] = w
        rp = list(RP_MEMBERS)
        raw = risque.risk_parity_raw_weights(s[rp])
        gross, w = risque.targeted_book(raw, simple[rp])
        out["risk_parity_35"][arm] = gross - funding_charge(w, rate, mode="signed")
        weights["risk_parity_35"][arm] = w
    return {"returns": out, "weights": weights}


def book_sigma(p: dict, forecasts: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Annualised forecast volatility, set missing wherever any sizing arm is missing."""
    mask = p["defined"]
    return {arm: risque.annualised_vol(f.where(mask)) for arm, f in forecasts.items()}


def book_window(p: dict) -> pd.DatetimeIndex:
    """From the session after the last one with fewer than 30 sized instruments, to the end.

    The M3 multiplier needs 63 sessions of the unscaled book plus one lag; the window
    starts no earlier than 65 sessions after the first sized session.
    """
    defined = p["defined"]
    index = p["index"]
    starts = []
    for members in (list(p["prices"].columns), list(RP_MEMBERS)):
        count = defined[members].sum(axis=1).to_numpy()
        below = np.flatnonzero(count < MIN_N)
        last_below = int(below[below < len(count) - 1].max()) if below.size else -1
        starts.append(last_below + 2)  # weights are held the session after the forecast
    first_sized = int(np.flatnonzero(defined.any(axis=1).to_numpy()).min())
    start = max(*starts, first_sized + 65)
    return index[start:]


# ---------------------------------------------------------------------------
# measures
# ---------------------------------------------------------------------------
def sharpe(x: pd.Series) -> float:
    v = x.to_numpy(float)
    if not np.isfinite(v).all() or len(v) < 2 or v.std(ddof=1) <= 0:
        return float("nan")
    return float(v.mean() / v.std(ddof=1) * np.sqrt(risque.PERIODS))


def max_drawdown(x: pd.Series) -> float:
    curve = (1.0 + x.fillna(0.0)).cumprod()
    return float((curve / curve.cummax() - 1.0).min())


def transitions_per_year(label: pd.Series) -> float:
    v = label.dropna()
    return float((v.diff().abs() > 0).sum() / (len(v) / risque.PERIODS))


def cohen_kappa(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(bool), b.astype(bool)
    observed = float((a == b).mean())
    chance = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
    return (observed - chance) / (1.0 - chance) if chance < 1 else float("nan")


def dimension(prices: pd.DataFrame, start, end) -> dict[str, float]:
    """Participation ratio of month-end log RV21 across the panel (the advisor's measure)."""
    px = prices.loc[start:end]
    rv = np.log(px.where(px > 0)).diff().rolling(21, min_periods=15).std()
    monthly = np.log(rv.resample("ME").last()).dropna(axis=0, how="any")
    eig = np.sort(np.linalg.eigvalsh(monthly.corr().to_numpy()))[::-1]
    return {
        "level": risque.participation_ratio(monthly.corr().to_numpy()),
        "change": risque.participation_ratio(monthly.diff().dropna().corr().to_numpy()),
        "first_eig": float(eig[0]), "months": int(len(monthly)), "n": int(monthly.shape[1]),
        "first": str(monthly.index.min().date()), "last": str(monthly.index.max().date()),
    }


# ---------------------------------------------------------------------------
# the instrument
# ---------------------------------------------------------------------------
def instrument(p: dict) -> dict | None:
    print(RULE)
    print("IDEA 6 — THE REGIME AS AN INPUT OF A RISK FORECAST, 46 MARKETS — INSTRUMENT")
    print(RULE)
    print("\n0.  INPUTS (SHA-256)\n")
    for path in INPUTS:
        print(f"   {sha256(path)[:16]}  {path.relative_to(RAW.parent)}")

    ev = p["evaluation"]
    common = p["common"].loc[ev]
    print(f"\n{RULE}\n1.  COVERAGE — no loss, no return\n")
    print(f"   training rows from {TRAIN_START:%Y-%m-%d}, at least {risque.MIN_ROWS}, "
          f"targets realised by t; HAR windows {risque.HAR_WINDOWS}; horizon {HORIZON}")
    print(f"   evaluation sessions {ev.min():%Y-%m-%d} -> {ev.max():%Y-%m-%d}, {len(ev):,}")
    counts = {"non_us_43": common[list(NON_US)].sum(axis=1),
              "us_equity_3": common[list(US_EQUITY)].sum(axis=1)}
    for name, members in CLASSES.items():
        counts[name] = common[list(members)].sum(axis=1)
    print(f"\n   {'panel':<15}{'markets':>8}{'min':>6}{'median':>8}{'max':>6}{'empty days':>12}")
    for name, c in counts.items():
        size = {"non_us_43": 43, "us_equity_3": 3}.get(name, len(CLASSES.get(name, ())))
        print(f"   {name:<15}{size:>8}{int(c.min()):>6}{int(c.median()):>8}{int(c.max()):>6}"
              f"{int((c == 0).sum()):>12}")
    print(f"\n   {'market':<10}{'first forecast':>15}{'defined share':>15}{'stale windows':>15}")
    for name in p["prices"].columns:
        col = p["common"][name]
        first = col[col].index.min()
        share = float(common[name].mean())
        print(f"   {name:<10}{first:%Y-%m-%d}{share:>15.1%}{p['store'][name]['zero_windows']:>15}")

    print(f"\n{RULE}\n2.  THE STATE AND THE RULES ON THE EVALUATION SESSIONS — descriptive\n")
    lab = {k: v.loc[ev] for k, v in p["labels"].items()}
    for k, v in lab.items():
        print(f"   {k:<7} stress share {v.eq(0).mean():>6.1%}   transitions/yr "
              f"{transitions_per_year(v):>5.2f}")
    s = lab["state"].eq(0).to_numpy()
    for k in ("median", "80th"):
        print(f"   kappa(state, {k} rule) {cohen_kappa(s, lab[k].eq(0).to_numpy()):+.2f}")

    print(f"\n{RULE}\n3.  AXIS 4 OF THE OPPOSABLE RULE — effective dimension of the volatility "
          "panel,\n    remeasured before any reading (participation ratio, month-end log RV21)\n")
    prices = p["prices"]
    adv = dimension(prices, "2003-07-17", None)
    own = dimension(prices, ev.min(), ev.max())
    for label, d in (("advisor's sample, from 2003-07-17", adv),
                     ("this study's evaluation window", own)):
        print(f"   {label:<36} level {d['level']:.2f}   monthly change {d['change']:.2f}   "
              f"first eigenvalue {d['first_eig']:.1f}/{d['n']}   {d['months']} months "
              f"{d['first']} -> {d['last']}")
    seventh = own["level"] < 4.0
    print("   => the economic part (E1, E2) is "
          + ("DECLARED A SEVENTH DEVICE before reading: level below 4" if seventh
             else "not a seventh device on axis 4: level at or above 4"))

    print(f"\n{RULE}\n4.  MINIMUM DETECTABLE EFFECT — forecast panels, blinded (demeaned d_t)\n")
    diffs = {}
    for code, _, members, base, _ in FORECAST_TESTS:
        diffs[code] = differential(p, members, base, f"{base}+state")
    matrix = pd.DataFrame(diffs)
    bad = int((~np.isfinite(matrix.to_numpy())).sum())
    print(f"   sessions {len(matrix):,}, non-finite cells {bad}")
    if bad:
        print("\n   A READING IS NOT FINITE. No verdict.")
        return None
    ses = {b: risque.blinded_mean_se(matrix.to_numpy(), mean_block=b, draws=DRAWS, seed=0)
           for b in BLOCKS}
    mdes: dict[str, float] = {}
    print(f"   {'test':<5}{'panel':<16}{'base':<8}{'alpha':>9}   MDE blocks 21 / 63 / 126"
          "        threshold   base QLIKE   MDE %")
    for j, (code, panel, members, base, alpha) in enumerate(FORECAST_TESTS):
        values = [ses[b][j] * risque.mde_z(alpha) for b in BLOCKS]
        mdes[code] = max(values)
        joined = " / ".join(f"{v:.5f}" for v in values)
        level = float(p["losses"][base][list(members)].loc[ev].mean(axis=1).mean())
        print(f"   {code:<5}{panel:<16}{base:<8}{alpha:>9.5f}   {joined}   {mdes[code]:.5f}"
              f"   {level:>10.4f}   {mdes[code] / level:>5.2%}")
    print("   units: mean QLIKE per market-session. The base QLIKE is the loss of the model")
    print("   WITHOUT the state, printed only to express the MDE as a share of it.")

    print(f"\n{RULE}\n5.  THE BOOKS — coverage and blinded MDE, no Sharpe\n")
    window = book_window(p)
    sigma = book_sigma(p, {a: p["forecasts"][a] for a in BOOK_ARMS})
    legs = book_legs(p, sigma)
    p["book_window"], p["book_sigma"], p["book_legs"] = window, sigma, legs
    for members, label in ((list(prices.columns), "trend_46"), (list(RP_MEMBERS),
                                                                  "risk_parity_35")):
        c = p["defined"][members].sum(axis=1).shift(1).loc[window]
        print(f"   {label:<15} {window.min():%Y-%m-%d} -> {window.max():%Y-%m-%d}, "
              f"{len(window):,} sessions, sized instruments min {int(c.min())} "
              f"median {int(c.median())}")
    for code, book in BOOK_TESTS:
        a = legs["returns"][book]["har+state"].loc[window].to_numpy()
        b = legs["returns"][book]["har"].loc[window].to_numpy()
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            print("\n   A BOOK LEG IS NOT FINITE. No verdict.")
            return None
        values = [mde_at(blinded_mde(a, b, mean_block=blk, draws=DRAWS, seed=0), ALPHA_BOOK)
                  for blk in BLOCKS]
        mdes[code] = max(values)
        joined = " / ".join(f"{v:.3f}" for v in values)
        print(f"   {code:<4}{book:<16} MDE (Sharpe) {joined}  ->  threshold {mdes[code]:.3f}")
    print(f"\n   families: P at 0.05/3, C at 0.05/8, E at 0.05/2; per market Holm at "
          f"{HOLM_ALPHA} over 46 per baseline")
    return {"mdes": mdes, "dimension": own, "dimension_advisor": adv, "seventh": seventh}


# ---------------------------------------------------------------------------
# the reading
# ---------------------------------------------------------------------------
def rotation_draws(p: dict) -> dict[str, np.ndarray]:
    """400 circular rotations of the stress dummy, every forecast refitted on each."""
    index = p["index"]
    stress = p["dummies"]["state"]
    seg = np.flatnonzero(np.isfinite(stress))
    rng = np.random.default_rng(PLACEBO_SEED)
    shifts = rng.integers(MIN_SHIFT, len(seg) - MIN_SHIFT, size=ROTATIONS)
    names = list(p["prices"].columns)
    target = p["target"]
    common = p["common"]
    window = p["book_window"]
    base_sharpe = {book: sharpe(p["book_legs"]["returns"][book]["har"].loc[window])
                   for _, book in BOOK_TESTS}
    out = {code: np.empty(ROTATIONS) for code, *_ in FORECAST_TESTS}
    out.update({code: np.empty(ROTATIONS) for code, _ in BOOK_TESTS})
    for i, k in enumerate(shifts):
        rot = stress.copy()
        rot[seg] = np.roll(stress[seg], int(k))
        f = {"har+state": {}, "harvix+state": {}}
        for name in names:
            st = p["store"][name]
            for base in BASES:
                f[f"{base}+state"][name] = risque.forecast_with_dummy(
                    st[base], rot, usable=st["regs"], base=p["forecasts"][base][name].to_numpy())
        losses = dict(p["losses"])
        rot_f = {}
        for arm, v in f.items():
            rot_f[arm] = pd.DataFrame(v, index=index)[names]
            losses[arm] = risque.qlike(target, rot_f[arm]).where(common)
        for code, _, members, base, _ in FORECAST_TESTS:
            out[code][i] = differential(p, members, base, f"{base}+state", losses).mean()
        sig = risque.annualised_vol(rot_f["har+state"].where(p["defined"]))
        legs = book_legs(p, {"har+state": sig})
        for code, book in BOOK_TESTS:
            rotated = sharpe(legs["returns"][book]["har+state"].loc[window])
            out[code][i] = rotated - base_sharpe[book]
    return out


def fold_deltas(d: pd.Series) -> list[float]:
    return [float(d.iloc[a:b].mean()) for a, b in risque.fold_bounds(len(d), FOLDS)]


def read(p: dict, inst: dict) -> None:
    mdes = inst["mdes"]
    ev = p["evaluation"]
    L = p["losses"]
    print(f"\n{RULE}\nTHE READING — zero cost, excess of cash, {ev.min():%Y-%m-%d} -> "
          f"{ev.max():%Y-%m-%d}\n")
    print("   rotation placebo: 400 draws, every forecast refitted on each ...", flush=True)
    null = rotation_draws(p)

    # ---------------- forecast panels ----------------
    print(f"\n{RULE}\nA.  THE FORECAST PANELS — mean QLIKE per market-session\n")
    print(f"   {'test':<5}{'panel':<15}{'base':<7}{'QLIKE base':>11}{'+state':>9}{'gain %':>8}"
          f"{'+median %':>10}{'+80th %':>9}{'VIX adds %':>11}")
    rows = {}
    for code, panel, members, base, _ in FORECAST_TESTS:
        m = list(members)
        q = {arm: float(L[arm][m].loc[ev].mean(axis=1).mean())
             for arm in (base, f"{base}+state", f"{base}+median", f"{base}+80th", "har",
                         "harvix")}
        vix_adds = (q["har"] - q["harvix"]) / q["har"]
        print(f"   {code:<5}{panel:<15}{base:<7}{q[base]:>11.4f}{q[base + '+state']:>9.4f}"
              f"{(q[base] - q[base + '+state']) / q[base]:>+8.2%}"
              f"{(q[base] - q[base + '+median']) / q[base]:>+10.2%}"
              f"{(q[base] - q[base + '+80th']) / q[base]:>+9.2%}{vix_adds:>+11.2%}")
        rows[code] = q

    print(f"\n{RULE}\nB.  DECISIONS — forecast panels\n")
    verdicts = {}
    for code, panel, members, base, alpha in FORECAST_TESTS:
        d = differential(p, members, base, f"{base}+state")
        delta = float(d.mean())
        t = risque.hac_mean_t(d, lags=FORECAST_LAGS)
        t6 = risque.hac_mean_t(d.iloc[::HORIZON], lags=BOOK_LAGS)
        dist = null[code]
        pct = float((dist < delta).mean()) if np.isfinite(dist).all() else float("nan")
        d_med = float(differential(p, members, base, f"{base}+median").mean())
        d_tail = float(differential(p, members, base, f"{base}+80th").mean())
        folds = fold_deltas(d)
        positive = int(sum(f > 0 for f in folds))
        v = risque.verdict(delta, mdes[code], t, pct, [d_med, d_tail], positive,
                           folds_needed=FOLDS_NEEDED)
        stress = pd.Series(p["dummies"]["state"], index=p["index"]).loc[ev].eq(1.0)
        print(f"   {code} {panel} beyond {base}: delta {delta:+.5f}  MDE {mdes[code]:.5f}  "
              f"t HAC21 {t:+.2f}  (every 21st, HAC6: {t6:+.2f})")
        print(f"        placebo pct {pct:.1%} (p5 {np.quantile(dist, 0.05):+.5f}, "
              f"p95 {np.quantile(dist, 0.95):+.5f})   rules: median {d_med:+.5f}, "
              f"80th {d_tail:+.5f}")
        print(f"        folds {' '.join(f'{f:+.5f}' for f in folds)}  ({positive}/5 positive)")
        print(f"        by lagged state (descriptive): stress sessions {d[stress].mean():+.5f} "
              f"({int(stress.sum())}), calm {d[~stress].mean():+.5f} ({int((~stress).sum())})")
        print(f"        => {v}")
        verdicts[code] = v
        base_q = rows[code][base]
        trials.log(
            FAMILY,
            {"test": code, "panel": panel, "markets": len(members), "base": base,
             "addition": "A' sparse jump stress dummy, filtered, known at t",
             "target": "log forward 21-session realised variance", "loss": "QLIKE",
             "estimation": "expanding OLS from 2002-04-01, rows s <= t-21, >= 504 rows",
             "alpha": alpha, "cost": "none",
             "sample": [str(ev.min().date()), str(ev.max().date())]},
            {"delta": delta, "relative_gain": delta / base_q, "threshold": mdes[code],
             "t_hac": t, "t_nonoverlap_hac6": t6, "placebo_pct": pct,
             "delta_median_rule": d_med, "delta_tail_rule": d_tail,
             "folds_positive": positive, "sessions": int(len(d)), "verdict": v},
        )

    # ---------------- per market ----------------
    print(f"\n{RULE}\nC.  PER MARKET — DM t (HAC 21) of QLIKE(base) - QLIKE(base+state), Holm "
          f"at {HOLM_ALPHA} over 46\n")
    names = list(p["prices"].columns)
    klass = {m: "us_equity" for m in US_EQUITY}
    klass.update({m: k for k, members in CLASSES.items() for m in members})
    per_market = {}
    for base in BASES:
        ts, ds = [], []
        for name in names:
            d = (L[base][name] - L[f"{base}+state"][name]).loc[ev].dropna()
            ds.append(float(d.mean()))
            ts.append(risque.hac_mean_t(d, lags=FORECAST_LAGS))
        ts = np.array(ts)
        pvals = 2.0 * (1.0 - stats.norm.cdf(np.abs(ts)))
        reject = risque.holm_reject(pvals, HOLM_ALPHA)
        table = pd.DataFrame({"class": [klass[n] for n in names], "delta": ds, "t": ts,
                              "better": reject & (ts > 0), "worse": reject & (ts < 0),
                              "positive": np.array(ds) > 0}, index=names)
        per_market[base] = table
        summary = table.groupby("class").agg(markets=("delta", "size"),
                                             positive=("positive", "sum"),
                                             better=("better", "sum"), worse=("worse", "sum"))
        print(f"   beyond {base}:  better (Holm) {int(table['better'].sum())}, worse (Holm) "
              f"{int(table['worse'].sum())}, positive delta {int(table['positive'].sum())}/46")
        print("   " + summary.to_string().replace("\n", "\n   "))
        print()
        trials.log(
            FAMILY,
            {"test": f"per_market_{base}", "markets": 46, "base": base,
             "correction": f"Holm {HOLM_ALPHA} over 46", "hac": FORECAST_LAGS,
             "sample": [str(ev.min().date()), str(ev.max().date())]},
            {"better_holm": int(table["better"].sum()), "worse_holm": int(table["worse"].sum()),
             "positive_delta": int(table["positive"].sum()), "sessions": int(len(ev)),
             "verdict": "count only, no verdict"},
        )
    for base in BASES:
        t = per_market[base]
        print(f"   beyond {base}, every market (delta x 1e4, t):")
        line = [f"{n} {1e4 * t.loc[n, 'delta']:+.1f} ({t.loc[n, 't']:+.1f})" for n in names]
        for i in range(0, len(line), 5):
            print("     " + "   ".join(line[i:i + 5]))

    # ---------------- other losses, descriptive ----------------
    print(f"\n{RULE}\nD.  OTHER CRITERIA — descriptive, never deciding\n")
    target = p["target"]
    common = p["common"]
    scale = target.where(common).loc[ev].mean()
    print("   MSE of the variance, each market scaled by its mean realised variance;")
    print("   Mincer-Zarnowitz in levels per market (median slope, median R2)")
    for code, panel, members, base, _ in FORECAST_TESTS[:3]:
        m = list(members)
        err = {}
        mz = {}
        for arm in (base, f"{base}+state"):
            e = ((target - p["forecasts"][arm]).where(common)[m].loc[ev] / scale[m]) ** 2
            err[arm] = e.mean(axis=1)
            slopes, r2s = [], []
            for name in m:
                both = pd.concat([target[name], p["forecasts"][arm][name]], axis=1).where(
                    common[name], axis=0).loc[ev].dropna()
                x = both.iloc[:, 1].to_numpy()
                yv = both.iloc[:, 0].to_numpy()
                b = np.cov(yv, x, ddof=1)[0, 1] / x.var(ddof=1)
                slopes.append(b)
                r2s.append(np.corrcoef(yv, x)[0, 1] ** 2)
            mz[arm] = (float(np.median(slopes)), float(np.median(r2s)))
        dm = err[base] - err[f"{base}+state"]
        print(f"   {code} {panel:<12} MSE gain {dm.mean() / err[base].mean():+.2%} "
              f"(t HAC21 {risque.hac_mean_t(dm, lags=FORECAST_LAGS):+.2f})   MZ {base}: "
              f"b {mz[base][0]:.2f} R2 {mz[base][1]:.3f}   +state: b {mz[base + '+state'][0]:.2f} "
              f"R2 {mz[base + '+state'][1]:.3f}")

    # ---------------- sensitivities ----------------
    print(f"\n{RULE}\nE.  SENSITIVITIES — declared, never deciding\n")
    sens = {}
    for code, _, members, base, _ in FORECAST_TESTS[:2]:
        m = [x for x in members if x != "CL=F"]
        d = differential(p, m, base, f"{base}+state")
        sens[f"{code}_without_CL"] = float(d.mean())
        dmed = (L[base][list(members)] - L[f"{base}+state"][list(members)]).loc[ev].median(axis=1)
        sens[f"{code}_cross_median"] = float(dmed.mean())
        print(f"   {code} without CL=F: delta {d.mean():+.5f} "
              f"(t {risque.hac_mean_t(d, lags=21):+.2f})"
              f"   cross-sectional median: {dmed.mean():+.5f} "
              f"(t {risque.hac_mean_t(dmed, lags=21):+.2f})")
    ew = p["losses"]
    ok = ew["ewma"].notna() & ew["ewma+state"].notna()
    d = (ew["ewma"] - ew["ewma+state"]).where(ok)[list(NON_US)].loc[ev].mean(axis=1)
    base_q = float(ew["ewma"].where(ok)[list(NON_US)].loc[ev].mean(axis=1).mean())
    sens["P1_ewma"] = float(d.mean())
    print(f"   P1 with a calibrated EWMA (0.94) as the base: delta {d.mean():+.5f} "
          f"({d.mean() / base_q:+.2%}), t {risque.hac_mean_t(d, lags=21):+.2f}, "
          f"sessions {int(d.notna().sum())}")
    d = differential(p, US_EQUITY, "har", "har+state")
    sens["US3_beyond_har"] = float(d.mean())
    q = float(L["har"][list(US_EQUITY)].loc[ev].mean(axis=1).mean())
    print(f"   US equities beyond HAR alone (the known result, descriptive): delta "
          f"{d.mean():+.5f} ({d.mean() / q:+.2%}), t {risque.hac_mean_t(d, lags=21):+.2f}")
    trials.log(FAMILY, {"test": "sensitivities", "declared": "PRESPEC_RISQUE section 8"},
               {**sens, "verdict": "sensitivities, never deciding"})

    # ---------------- the books ----------------
    window = p["book_window"]
    legs = p["book_legs"]
    print(f"\n{RULE}\nF.  THE BOOKS — zero cost, excess of cash, {window.min():%Y-%m-%d} -> "
          f"{window.max():%Y-%m-%d}\n")
    ref = vol_targeted_book(p["prices"])
    rate = cash_rate_daily(p["index"])
    m3 = (ref["returns"] - funding_charge(ref["weights"], rate, mode="signed")).loc[window]
    for code, book in BOOK_TESTS:
        print(f"   {code} {book}")
        print(f"   {'sizing':<14}{'Sharpe':>8}{'return':>9}{'vol':>8}{'maxDD':>9}{'turn/yr':>9}"
              f"{'gross':>7}{'track err':>10}")
        arms = dict(legs["returns"][book])
        if book == "trend_46":
            arms["M3 sigma63 (ref.)"] = m3
        for arm, x in arms.items():
            x = x.loc[window]
            w = legs["weights"][book].get(arm, ref["weights"]).loc[window]
            monthly_vol = x.groupby(x.index.to_period("M")).std() * np.sqrt(risque.PERIODS)
            track = float(np.log(monthly_vol[monthly_vol > 0] / 0.10).std())
            print(f"   {arm:<14}{sharpe(x):>+8.3f}{x.mean() * 252:>+9.2%}"
                  f"{x.std() * np.sqrt(252):>8.2%}{max_drawdown(x):>9.1%}"
                  f"{float(w.diff().abs().sum(axis=1).mean() * 252):>9.1f}"
                  f"{float(w.abs().sum(axis=1).mean()):>7.2f}{track:>10.3f}")
        a = legs["returns"][book]["har+state"].loc[window]
        b = legs["returns"][book]["har"].loc[window]
        delta = sharpe(a) - sharpe(b)
        t = paired_hac_t(a, b, lags=BOOK_LAGS)
        dist = null[code]
        pct = float((dist < delta).mean()) if np.isfinite(dist).all() else float("nan")
        controls = {arm: sharpe(legs["returns"][book][arm].loc[window]) - sharpe(b)
                    for arm in ("har+median", "har+80th", "harvix")}
        diff = a - b
        folds = [sharpe(a.iloc[i:j]) - sharpe(b.iloc[i:j])
                 for i, j in risque.fold_bounds(len(diff), FOLDS)]
        positive = int(sum(f > 0 for f in folds))
        v = risque.verdict(delta, mdes[code], t, pct, list(controls.values()), positive,
                           folds_needed=FOLDS_NEEDED)
        if inst["seventh"]:
            v += " [declared a seventh device before reading]"
        print(f"\n   delta {delta:+.3f}  MDE {mdes[code]:.3f}  t HAC6 {t:+.2f}  placebo pct "
              f"{pct:.1%} (p5 {np.quantile(dist, 0.05):+.3f}, p95 {np.quantile(dist, 0.95):+.3f})")
        print("   controls: " + ", ".join(f"{k} {v_:+.3f}" for k, v_ in controls.items()))
        print(f"   folds {' '.join(f'{f:+.3f}' for f in folds)}  ({positive}/5 positive)")
        print(f"   => {v}\n")
        verdicts[code] = v
        trials.log(
            FAMILY,
            {"test": code, "book": book, "sizing": "HAR+state forecast vs HAR forecast",
             "portfolio_target": "M3, 10%, sigma63 of the unscaled book, cap 3",
             "cost": "zero", "excess": "cash on signed funded exposure",
             "alpha": ALPHA_BOOK, "sample": [str(window.min().date()), str(window.max().date())]},
            {"sharpe": sharpe(a), "sharpe_alone": sharpe(b), "delta": delta,
             "threshold": mdes[code], "t_hac": t, "placebo_pct": pct,
             **{f"delta_{k.replace('+', '_')}": v_ for k, v_ in controls.items()},
             "folds_positive": positive, "sessions": int(len(window)), "verdict": v},
        )

    print(f"{RULE}\nSUMMARY\n")
    for code, v in verdicts.items():
        print(f"   {code:<4} {v}")
    print(f"\n   logged to data/trials.parquet ({trials.summary()['n_distinct']} distinct)")
    print(RULE)


def main() -> None:
    p = build()
    inst = instrument(p)
    if inst is None:
        return
    if "--read" not in sys.argv:
        print(f"\n{RULE}\nNOT READ — re-run with --read, once.\n{RULE}")
        return
    read(p, inst)


if __name__ == "__main__":
    main()

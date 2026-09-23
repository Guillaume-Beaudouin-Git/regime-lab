"""Crisis objects, alone and coupled with the sparse jump model's state.

The protocol is `docs/PRESPEC_CRISE.md`. EVERYTHING BELOW IS FIXED, AND COMMITTED,
BEFORE ANY RETURN CONDITIONED ON THE STATE IS READ.

The question. The classifier (A' sparse jump) carries information about forward
variance and none about forward mean. Every object coupled to it so far had a payoff
roughly linear in the market, and lost to a one-line volatility rule. Does the state
matter more for objects whose return is itself a function of variance, or which are
known to crash in one kind of state?

The four strategies (the family; no fifth is added after reading)
    VRP   short one-month S&P 500 variance swap, synthetic: a ladder of 21 tranches,
          strike (VIX/100)^2 x 21/252, realised variance of ^GSPC log returns,
          marked to market with the VIX (`extensions.crisis.variance_swap_ladder`).
          Data 1990-2026, Yahoo, `data/raw/prices/cross_asset.parquet`
    VIXF  short one-month constant-maturity VIX futures, rebuilt from the CFE daily
          settlements of every monthly contract (`vix_futures_constant_maturity`),
          from 2006-09-01, the first date after which both front contracts always
          have a settle. Data `data/raw/crisis/vix_futures.parquet`
    UMD   Ken French's daily momentum factor, long-short, already an excess return.
          Data `data/raw/crisis/french_umd.parquet`
    BXM   CBOE S&P 500 BuyWrite index, daily from 2002-03-22, in excess of the
          3-month bill (`rate_cash_3m`, stamped at availability).
          Data `data/raw/crisis/cboe_indices.parquet`
    Each is sized as everywhere in the programme: weight min(0.10 / sigma_63, 3),
    sigma_63 the realised volatility of the strategy's own unit return over the 63
    sessions ending the session before. A swap, a future and a long-short factor are
    unfunded, so their returns are excess returns as they stand.

The state
    A' sparse jump, filtered, `data/cache/states.parquet`, 0 = stress, 1 = calm; the
    state used on session d is the last one stamped at or before d-1.

The two couplings (the family: 4 strategies x 2 couplings + test P = 9 tests)
    STOP  weight x 0 on sessions whose lagged state is stress
    HALF  weight x 0.5 on those sessions (regime-dependent leverage)

The controls, each through the same code
    - the programme's one-line rule: stress when the 21-session realised volatility
      of ^GSPC is above its expanding median (`volatility_quantile_placebo`);
    - the same rule at the classifier's order of stress share: above the expanding
      80th percentile (`extensions.crisis.volatility_tail_rule`);
    - a rotation placebo: the lagged state rotated circularly against the dates,
      400 rotations of at least 252 sessions.

Sample per strategy: from the later of 2002-04-01 (first out-of-sample state) and the
first session with a weight, to the earlier of the last state and the last datum.

THE DECISION, WRITTEN BEFORE THE NUMBER (per strategy x coupling, zero cost)
    delta = Sharpe(coupled) - Sharpe(alone), both in excess of cash, same sessions
    MDE   = `selection.protocol.blinded_mde`, paired stationary block bootstrap on
            demeaned legs, blocks 21 / 63 / 126, the largest, at alpha 0.05 / 9
    t     = HAC lag-6 t of the daily difference (`paired_hac_t`)
    USEFUL        delta >= MDE, t of the same sign, delta above the 95th percentile of
                  the rotation placebo, and delta above the same coupling built on
                  each of the two volatility rules
    NOT SHOWN     delta >= MDE but one of the three other conditions fails
    UNDERPOWERED  0 < delta < MDE — never useful
    NOT USEFUL    -MDE < delta <= 0
    HARMFUL       delta <= -MDE, t of the same sign, delta below the 5th percentile of
                  the rotation placebo: the filter measurably destroys value
    NOT USEFUL (negative beyond the MDE, not confirmed) otherwise
    A reading with a non-finite value gives no verdict.

TEST P, "a better use of the prediction", written before the number
    On every out-of-sample session t with 21 forward sessions:
        y_t = log( sum_{j=t+1..t+21} r_j^2 ),  r = ^GSPC log returns
        base   y ~ 1 + log K_t + vol_rank_t        (K_t = implied monthly variance)
        full   y ~ 1 + log K_t + vol_rank_t + state_t
    vol_rank is the percentile rank of 21-session realised volatility, as in
    `evaluation.predictive.incremental_information`. With log K_t in both, the
    state's coefficient is the same whether y is log RV or the log variance premium
    log(RV / K): the test asks whether the state predicts the variance premium
    beyond the VIX and a volatility quantile.
    PREDICTS  |t_state| >= z(1 - alpha/2) at alpha 0.05 / 9 (HAC, 21 lags, the
              programme's convention for overlapping 21-session targets), AND the
              incremental R2 above the 95th percentile of the rotation placebo.
    Otherwise DOES NOT PREDICT BEYOND THE VIX. Sensitivity, never deciding: the
    non-overlapping sample (every 21st session), HAC 6.

Reported beside the decisions, never deciding: annual return, volatility, maximum
drawdown, worst month, annual turnover of the weight, the P&L inside six windows
dated by the S&P 500 and not by the state, the Sharpe of each strategy alone on
stress and on calm sessions, and — for UMD and BXM, whose unit is a funded 1x
position — the same STOP coupling without volatility targeting.

Discipline, as in `scripts/run_b1_regime_coupling.py`: run plain, the script measures
the instrument and prints no return conditioned on the state. `--read` prints the
results and logs nine rows (family `crise_coupling`) to `data/trials.parquet`; run
it once.

Usage:
    .venv/bin/python scripts/run_crisis_coupling.py           # instrument only
    .venv/bin/python scripts/run_crisis_coupling.py --read    # the reading, 9 trials
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from regime_lab.analysis import trials
from regime_lab.config import CACHE, RAW
from regime_lab.evaluation.predictive import volatility_quantile_placebo
from regime_lab.extensions import crisis
from regime_lab.extensions.vehicle import vol_targeted_book
from regime_lab.selection.protocol import blinded_mde, mde_at, paired_hac_t
from regime_lab.strategies.book import base_book

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_m3_evaluation import SAMPLE, cash_rate_daily  # noqa: E402

RULE = "=" * 78
FAMILY = "crise_coupling"
STATE_COLUMN = "A' sparse jump"
OOS_START = pd.Timestamp("2002-04-01")
VIXF_START = pd.Timestamp("2006-09-01")
N_TESTS = 9
ALPHA = 0.05 / N_TESTS
BLOCKS = (21, 63, 126)
DRAWS = 2000
ROTATIONS = 400
MIN_SHIFT = 252
MODES = ("stop", "half")
STRATEGIES = ("VRP", "VIXF", "UMD", "BXM")
RAW_LEGS = ("UMD", "BXM")
HORIZON = 21
BEAR_WINDOW = 504
WINDOWS = {
    "GFC": ("2007-10-09", "2009-03-09"),
    "REB09": ("2009-03-10", "2009-12-31"),
    "FEB18": ("2018-01-26", "2018-02-08"),
    "COVID": ("2020-02-19", "2020-03-23"),
    "REB20": ("2020-03-24", "2020-12-31"),
    "2022": ("2022-01-03", "2022-10-12"),
}
INPUTS = (
    RAW / "prices" / "cross_asset.parquet",
    RAW / "crisis" / "vix_futures.parquet",
    RAW / "crisis" / "french_umd.parquet",
    RAW / "crisis" / "cboe_indices.parquet",
    RAW / "macro" / "rate_cash_3m.parquet",
    CACHE / "states.parquet",
)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def series(path: Path, series_id: str) -> pd.Series:
    frame = pd.read_parquet(path)
    one = frame[frame["series_id"] == series_id].set_index("period")["value"]
    one.index = pd.DatetimeIndex(one.index)
    return one.sort_index().astype(float)


def market() -> dict[str, pd.Series]:
    path = RAW / "prices" / "cross_asset.parquet"
    spx = series(path, "eq_us_large")
    vix = series(path, "vol_vix").reindex(spx.index)
    return {"spx": spx, "vix": vix, "log": np.log(spx).diff(), "simple": spx.pct_change()}


def vix_settles() -> pd.DataFrame:
    frame = pd.read_parquet(RAW / "crisis" / "vix_futures.parquet")
    wide = frame.pivot(index="period", columns="series_id", values="value")
    wide.index = pd.DatetimeIndex(wide.index)
    wide.columns = pd.DatetimeIndex([c[3:] for c in wide.columns])
    return wide.sort_index(axis=1)


def units(m: dict[str, pd.Series]) -> dict[str, pd.Series]:
    """The unit return of each strategy, before sizing: one unit, excess of cash."""
    sessions = m["spx"].index
    vrp = crisis.variance_swap_ladder(m["log"], m["vix"])
    vixf_sessions = sessions[sessions >= VIXF_START]
    vixf = -crisis.vix_futures_constant_maturity(vix_settles(), vixf_sessions)
    umd = series(RAW / "crisis" / "french_umd.parquet", "ff_umd")
    bxm_level = series(RAW / "crisis" / "cboe_indices.parquet", "cboe_bxm")
    bxm = bxm_level.pct_change() - cash_rate_daily(bxm_level.index)
    return {"VRP": vrp.rename("VRP"), "VIXF": vixf.rename("VIXF"),
            "UMD": umd.rename("UMD"), "BXM": bxm.rename("BXM")}


def build() -> dict:
    m = market()
    states = pd.read_parquet(CACHE / "states.parquet")[STATE_COLUMN]
    median_rule = volatility_quantile_placebo(m["simple"])
    tail_rule = crisis.volatility_tail_rule(m["simple"])
    out = {"market": m, "states": states, "strategies": {}}
    for name, unit in units(m).items():
        weight = crisis.vol_target_weight(unit.dropna())
        index = unit.dropna().index
        state = crisis.lagged_state(states, index)
        med = crisis.lagged_state(median_rule, index)
        tail = crisis.lagged_state(tail_rule, index)
        end = min(states.dropna().index.max(), index.max())
        ready = weight.notna() & state.notna() & med.notna() & tail.notna()
        start = max(OOS_START, ready[ready].index.min())
        window = index[(index >= start) & (index <= end)]
        out["strategies"][name] = {
            "unit": unit.reindex(window), "weight": weight.reindex(window),
            "state": state.reindex(window), "median_rule": med.reindex(window),
            "tail_rule": tail.reindex(window), "index": window,
            "raw_unit": unit, "raw_weight": weight,
        }
    return out


# ---------------------------------------------------------------------------
# measures
# ---------------------------------------------------------------------------
def sharpe(x: pd.Series) -> float:
    v = x.to_numpy(float)
    if not np.isfinite(v).all() or len(v) < 2 or v.std(ddof=1) <= 0:
        return float("nan")
    return float(v.mean() / v.std(ddof=1) * np.sqrt(252))


def max_drawdown(x: pd.Series) -> float:
    curve = (1.0 + x.fillna(0.0)).cumprod()
    return float((curve / curve.cummax() - 1.0).min())


def describe(x: pd.Series, weight: pd.Series | None = None) -> dict:
    monthly = (1.0 + x).groupby(x.index.to_period("M")).prod() - 1.0
    row = {
        "sharpe": sharpe(x), "ann_return": float(x.mean() * 252),
        "ann_vol": float(x.std(ddof=1) * np.sqrt(252)), "max_dd": max_drawdown(x),
        "worst_month": float(monthly.min()),
        "turnover": crisis.annual_turnover(weight) if weight is not None else float("nan"),
    }
    for name, (a, b) in WINDOWS.items():
        part = x.loc[a:b]
        row[name] = float((1.0 + part).prod() - 1.0) if len(part) else float("nan")
    return row


def legs(s: dict, state: pd.Series) -> dict[str, tuple[pd.Series, pd.Series]]:
    out = {"alone": (s["weight"] * s["unit"], s["weight"])}
    for mode in MODES:
        w = crisis.couple(s["weight"], state, mode)
        out[mode] = (w * s["unit"], w)
    return out


def cohen_kappa(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(bool), b.astype(bool)
    observed = float((a == b).mean())
    chance = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
    return (observed - chance) / (1.0 - chance) if chance < 1 else float("nan")


def episodes(state: pd.Series) -> int:
    stress = state.eq(crisis.STRESS).astype(int)
    return int((stress.diff() == 1).sum() + (stress.iloc[0] == 1))


def variance_loading(r: pd.Series, log_spx: pd.Series) -> dict[str, float]:
    """R2 of 21-session returns on the S&P 500's realised variance over the same 21.

    Non-overlapping blocks, unconditional: no state enters. It measures how far an
    object's return is a function of variance, the axis this study claims to change.
    """
    frame = pd.concat({"r": r, "rv": log_spx**2}, axis=1).dropna()
    block = np.arange(len(frame)) // HORIZON
    agg = frame.groupby(block).sum().iloc[:-1]
    fit = sm.OLS(agg["r"].to_numpy(), sm.add_constant(agg["rv"].to_numpy())).fit(
        cov_type="HAC", cov_kwds={"maxlags": 6})
    return {"r2": float(fit.rsquared), "t": float(fit.tvalues[1]), "n": int(len(agg))}


def p_frame(p: dict) -> pd.DataFrame:
    m = p["market"]
    log_r = m["log"]
    forward = (log_r**2).shift(-1).rolling(HORIZON).sum().shift(-(HORIZON - 1))
    realised = log_r.rolling(HORIZON).std() * np.sqrt(252)
    frame = pd.concat({
        "y": np.log(forward),
        "log_k": np.log(crisis.monthly_variance(m["vix"])),
        "realised": realised,
        "state": p["states"].reindex(log_r.index),
    }, axis=1)
    frame = frame.loc[OOS_START:].dropna()
    frame["vol_rank"] = frame["realised"].rank(pct=True)
    return frame


def p_regression(
    frame: pd.DataFrame, state: np.ndarray, lags: int,
    controls: tuple[str, ...] = ("log_k", "vol_rank"),
) -> dict[str, float]:
    """Incremental R2 and HAC t of the state over ``controls``, on log forward RV."""
    y = frame["y"].to_numpy()
    base_x = sm.add_constant(frame[list(controls)].to_numpy())
    base = sm.OLS(y, base_x).fit()
    full = sm.OLS(y, np.column_stack([base_x, state])).fit(
        cov_type="HAC", cov_kwds={"maxlags": lags})
    k = base_x.shape[1]
    return {"incremental": float(full.rsquared - base.rsquared), "coef": float(full.params[k]),
            "t": float(full.tvalues[k]), "r2_base": float(base.rsquared), "n": int(len(y))}


# ---------------------------------------------------------------------------
# the instrument
# ---------------------------------------------------------------------------
def instrument(p: dict) -> dict[tuple[str, str], float]:
    print(RULE)
    print("CRISIS OBJECTS, ALONE AND COUPLED WITH THE SPARSE JUMP STATE — INSTRUMENT")
    print(RULE)
    print("\n0.  INPUTS (SHA-256)\n")
    for path in INPUTS:
        print(f"   {sha256(path)[:16]}  {path.relative_to(RAW.parent)}")

    print(f"\n{RULE}\n1.  SAMPLES, STATE COVERAGE, SIZING — no mean, no Sharpe\n")
    print(f"   {'strategy':<6} {'first':>11} {'last':>11} {'sessions':>9} {'stress':>7} "
          f"{'episodes':>9} {'med rule':>9} {'80th rule':>10} {'NaN':>5} {'vol':>7} "
          f"{'lev p50':>8} {'lev p95':>8} {'cap':>6}")
    bad = False
    for name in STRATEGIES:
        s = p["strategies"][name]
        idx = s["index"]
        alone = s["weight"] * s["unit"]
        nan = int(alone.isna().sum() + s["state"].isna().sum() + s["median_rule"].isna().sum()
                  + s["tail_rule"].isna().sum())
        bad |= nan > 0
        w = s["weight"]
        print(f"   {name:<6} {idx.min():%Y-%m-%d} {idx.max():%Y-%m-%d} {len(idx):>9,} "
              f"{s['state'].eq(0).mean():>6.1%} {episodes(s['state']):>9} "
              f"{s['median_rule'].eq(0).mean():>8.1%} {s['tail_rule'].eq(0).mean():>9.1%} "
              f"{nan:>5} {alone.std() * np.sqrt(252):>6.1%} {w.median():>8.2f} "
              f"{w.quantile(0.95):>8.2f} {(w >= crisis.MAX_LEVERAGE).mean():>5.1%}")
    unit_v = p["strategies"]["VIXF"]["raw_unit"]
    late_nan = int(unit_v.iloc[1:].isna().sum())
    print(f"\n   VIXF unit: {late_nan} non-finite returns after {VIXF_START:%Y-%m-%d}")
    bad |= late_nan > 0

    print(f"\n{RULE}\n2.  THE AXIS — how far each object's return is a function of variance")
    print("    R2 of 21-session returns on the S&P 500's realised variance over the same")
    print("    21 sessions, non-overlapping blocks, no state involved\n")
    log_spx = p["market"]["log"]
    refs = {"60/40 book (T1-T3, decomposition)": base_book()}
    prices = pd.read_parquet(CACHE / "trend_universe_m1.parquet").loc[SAMPLE]
    refs["trend book, 46 instruments (AHL, H-b)"] = vol_targeted_book(prices)["returns"]
    for label, r in refs.items():
        r = r.loc[OOS_START:]
        v = variance_loading(r, log_spx)
        print(f"   {label:<40} R2 {v['r2']:>6.1%}   t {v['t']:>+6.2f}   n {v['n']}")
    for name in STRATEGIES:
        s = p["strategies"][name]
        v = variance_loading(s["weight"] * s["unit"], log_spx)
        print(f"   {name + ' (vol-targeted)':<40} R2 {v['r2']:>6.1%}   t {v['t']:>+6.2f}   "
              f"n {v['n']}")

    print(f"\n{RULE}\n2b. THE LATENT — overlap with the Daniel-Moskowitz bear state, no return of")
    print("    any strategy involved: bear = ^GSPC below its level 504 sessions earlier")
    print("    (24 months), out of sample\n")
    spx = p["market"]["spx"]
    bear = (spx / spx.shift(BEAR_WINDOW) - 1.0 < 0).astype(float).where(
        spx.shift(BEAR_WINDOW).notna())
    index = spx.index[spx.index >= OOS_START]
    labels = {
        "A' sparse jump": crisis.lagged_state(p["states"], index),
        "median rule": crisis.lagged_state(volatility_quantile_placebo(p["market"]["simple"]),
                                           index),
        "80th rule": crisis.lagged_state(crisis.volatility_tail_rule(p["market"]["simple"]),
                                         index),
    }
    b = bear.reindex(index)
    print(f"   bear share of sessions {b.mean():.1%}")
    for label, lab in labels.items():
        ok = lab.notna() & b.notna()
        stress = lab[ok].eq(crisis.STRESS)
        bb = b[ok].eq(1.0)
        kappa = cohen_kappa(stress.to_numpy(), bb.to_numpy())
        print(f"   {label:<15} stress {stress.mean():>6.1%}   P(bear | stress) "
              f"{bb[stress].mean():>6.1%}   P(stress | bear) {stress[bb].mean():>6.1%}   "
              f"kappa {kappa:+.2f}")

    print(f"\n{RULE}\n3.  MINIMUM DETECTABLE EFFECT — demeaned legs, alpha 0.05/{N_TESTS} = "
          f"{ALPHA:.4f}\n")
    mdes: dict[tuple[str, str], float] = {}
    for name in STRATEGIES:
        s = p["strategies"][name]
        lg = legs(s, s["state"])
        for mode in MODES:
            values = []
            for block in BLOCKS:
                res = blinded_mde(lg[mode][0].to_numpy(), lg["alone"][0].to_numpy(),
                                  mean_block=block, draws=DRAWS, seed=0)
                values.append(mde_at(res, ALPHA))
            mdes[(name, mode)] = max(values)
            joined = " / ".join(f"{v:.3f}" for v in values)
            print(f"   {name:<5} {mode:<5} MDE {joined}  ->  threshold {mdes[(name, mode)]:.3f}")

    frame = p_frame(p)
    z = float(stats.norm.ppf(1.0 - ALPHA / 2.0))
    print(f"\n{RULE}\n4.  TEST P — design only\n")
    print(f"   sessions {frame.index.min():%Y-%m-%d} -> {frame.index.max():%Y-%m-%d}, "
          f"n {len(frame):,}, stress share {frame['state'].eq(0).mean():.1%}")
    print(f"   threshold |t| >= {z:.2f} (HAC 21) and placebo p95; sensitivity every 21st "
          "session, HAC 6")
    if bad or not all(np.isfinite(v) for v in mdes.values()):
        print("\n   A READING IS NOT FINITE. No verdict.")
        return {}
    return mdes


# ---------------------------------------------------------------------------
# the reading
# ---------------------------------------------------------------------------
def rotation_null(s: dict, mode: str, shifts: np.ndarray, base_sharpe: float) -> np.ndarray:
    values = s["state"].to_numpy()
    out = []
    for k in shifts:
        rotated = pd.Series(np.roll(values, k), index=s["index"])
        w = crisis.couple(s["weight"], rotated, mode)
        out.append(sharpe(w * s["unit"]) - base_sharpe)
    return np.array(out)


def verdict(delta: float, mde: float, t: float, pct: float, d_med: float, d_tail: float) -> str:
    if not all(np.isfinite([delta, mde, t, pct, d_med, d_tail])):
        return "NOT FINITE — no verdict"
    if delta >= mde:
        if np.sign(t) == np.sign(delta) and pct >= 0.95 and delta > d_med and delta > d_tail:
            return "USEFUL — all conditions hold"
        return "NOT SHOWN — above the MDE, fails the t, the placebo or a volatility rule"
    if delta > 0:
        return "UNDERPOWERED — positive, below the MDE, never useful"
    if delta > -mde:
        return "NOT USEFUL — the coupling does not raise the Sharpe"
    if np.sign(t) == np.sign(delta) and pct <= 0.05:
        return "HARMFUL — the filter measurably lowers the Sharpe"
    return "NOT USEFUL — negative beyond the MDE, not confirmed by the t or the placebo"


def read(p: dict, mdes: dict[tuple[str, str], float]) -> None:
    fmt = {"sharpe": "{:+.2f}", "ann_return": "{:+.1%}", "ann_vol": "{:.1%}",
           "max_dd": "{:.1%}", "worst_month": "{:+.1%}", "turnover": "{:.1f}",
           **{c: "{:+.1%}" for c in WINDOWS}}
    rng = np.random.default_rng(20260923)
    for name in STRATEGIES:
        s = p["strategies"][name]
        idx = s["index"]
        real = legs(s, s["state"])
        med = legs(s, s["median_rule"])
        tail = legs(s, s["tail_rule"])
        print(f"\n{RULE}\n{name}  {idx.min():%Y-%m-%d} -> {idx.max():%Y-%m-%d}, "
              f"{len(idx):,} sessions, zero cost, excess of cash\n")
        rows = {
            "alone": describe(*real["alone"]),
            "stop (state)": describe(*real["stop"]), "half (state)": describe(*real["half"]),
            "stop (median rule)": describe(*med["stop"]),
            "half (median rule)": describe(*med["half"]),
            "stop (80th rule)": describe(*tail["stop"]),
            "half (80th rule)": describe(*tail["half"]),
        }
        if name in RAW_LEGS:
            raw = s["unit"]
            one = pd.Series(1.0, index=idx)
            rows["raw 1x alone"] = describe(raw, one)
            stop_raw = crisis.couple(one, s["state"], "stop")
            rows["raw 1x stop (state)"] = describe(stop_raw * raw, stop_raw)
        table = pd.DataFrame(rows).T
        print(f"   {'arm':<20}" + "".join(f"{c:>9}" for c in table.columns))
        for arm, r in table.iterrows():
            print(f"   {arm:<20}" + "".join(f"{fmt[c].format(r[c]):>9}" for c in table.columns))

        alone = real["alone"][0]
        stress = s["state"].eq(crisis.STRESS)
        print("\n   alone, by lagged state (descriptive):")
        for label, mask in (("stress", stress), ("calm", ~stress)):
            part = alone[mask]
            print(f"     {label:<7} {mask.sum():>6,} sessions   mean {part.mean() * 252:>+7.1%}/yr"
                  f"   vol {part.std() * np.sqrt(252):>6.1%}   Sharpe {sharpe(part):>+6.2f}")

        base = sharpe(alone)
        shifts = rng.integers(MIN_SHIFT, len(idx) - MIN_SHIFT, size=ROTATIONS)
        print("\n   decision:")
        for mode in MODES:
            coupled = real[mode][0]
            delta = sharpe(coupled) - base
            t = paired_hac_t(coupled, alone)
            null = rotation_null(s, mode, shifts, base)
            pct = float((null < delta).mean()) if np.isfinite(null).all() else float("nan")
            d_med = sharpe(med[mode][0]) - sharpe(med["alone"][0])
            d_tail = sharpe(tail[mode][0]) - sharpe(tail["alone"][0])
            ddd = max_drawdown(coupled) - max_drawdown(alone)
            v = verdict(delta, mdes[(name, mode)], t, pct, d_med, d_tail)
            print(f"   {mode.upper():<5} delta {delta:+.3f}  MDE {mdes[(name, mode)]:.3f}  "
                  f"t {t:+.2f}  placebo pct {pct:.1%} (median {np.median(null):+.3f}, "
                  f"p5 {np.quantile(null, 0.05):+.3f}, p95 {np.quantile(null, 0.95):+.3f})")
            print(f"         vol rules: median {d_med:+.3f}, 80th {d_tail:+.3f}   "
                  f"change in max DD {ddd:+.1%}")
            print(f"         => {v}")
            trials.log(
                FAMILY,
                {"test": f"{name}_{mode}", "strategy": name, "coupling": mode,
                 "state": "A' sparse jump filtered, lag 1", "cost": "zero",
                 "sizing": "min(0.10/sigma63, 3)", "alpha": f"0.05/{N_TESTS}",
                 "sample": [str(idx.min().date()), str(idx.max().date())]},
                {"sharpe": sharpe(coupled), "sharpe_alone": base, "delta": delta,
                 "threshold": mdes[(name, mode)], "t_hac": t, "placebo_pct": pct,
                 "delta_median_rule": d_med, "delta_tail_rule": d_tail, "delta_max_dd": ddd,
                 "sessions": int(len(idx)), "verdict": v},
            )

    read_p(p)
    print(f"\n   logged to data/trials.parquet ({trials.summary()['n_distinct']} distinct)")
    print(RULE)


def read_p(p: dict) -> None:
    frame = p_frame(p)
    z = float(stats.norm.ppf(1.0 - ALPHA / 2.0))
    real = p_regression(frame, frame["state"].to_numpy(), HORIZON)
    rng = np.random.default_rng(20260924)
    shifts = rng.integers(MIN_SHIFT, len(frame) - MIN_SHIFT, size=ROTATIONS)
    null = np.array([p_regression(frame, np.roll(frame["state"].to_numpy(), k), HORIZON)
                     ["incremental"] for k in shifts])
    pct = float((null < real["incremental"]).mean())
    monthly = frame.iloc[::HORIZON]
    sens = p_regression(monthly, monthly["state"].to_numpy(), 6)
    beyond_rank = p_regression(frame, frame["state"].to_numpy(), HORIZON, ("vol_rank",))
    print(f"\n{RULE}\nTEST P — does the state predict the variance premium beyond the VIX?\n")
    print(f"   n {real['n']:,}   R2 base (log K, vol rank) {real['r2_base']:.1%}")
    print(f"   state (1 = calm): coef {real['coef']:+.3f}   t HAC21 {real['t']:+.2f}   "
          f"incremental R2 {real['incremental']:.3%}   placebo pct {pct:.1%} "
          f"(p95 {np.quantile(null, 0.95):.3%})")
    print(f"   sensitivity, every 21st session, HAC 6: coef {sens['coef']:+.3f}  "
          f"t {sens['t']:+.2f}  incremental {sens['incremental']:.3%}  n {sens['n']}")
    print(f"   descriptive, without log K (vol rank only): coef {beyond_rank['coef']:+.3f}  "
          f"t {beyond_rank['t']:+.2f}  incremental {beyond_rank['incremental']:.3%}")
    if not all(np.isfinite([real["t"], real["incremental"], pct])):
        v = "NOT FINITE — no verdict"
    elif abs(real["t"]) >= z and pct >= 0.95:
        v = "PREDICTS — beyond the VIX and the volatility quantile"
    else:
        v = "DOES NOT PREDICT BEYOND THE VIX"
    print(f"   threshold |t| >= {z:.2f}   => {v}")
    trials.log(
        FAMILY,
        {"test": "P_variance_premium", "target": "log forward 21-session realised variance",
         "controls": ["log implied monthly variance", "vol rank 21"], "hac": HORIZON,
         "alpha": f"0.05/{N_TESTS}",
         "sample": [str(frame.index.min().date()), str(frame.index.max().date())]},
        {"coef": real["coef"], "t_hac": real["t"], "incremental_r2": real["incremental"],
         "placebo_pct": pct, "t_monthly": sens["t"], "sessions": real["n"], "verdict": v},
    )


def main() -> None:
    p = build()
    mdes = instrument(p)
    if not mdes:
        return
    if "--read" not in sys.argv:
        print(f"\n{RULE}\nNOT READ — re-run with --read, once.\n{RULE}")
        return
    read(p, mdes)


if __name__ == "__main__":
    main()

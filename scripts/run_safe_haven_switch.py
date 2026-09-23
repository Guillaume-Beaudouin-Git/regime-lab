"""Equities in calm, a crisis asset in stress: does the sparse jump state earn its keep?

EVERYTHING BELOW IS FIXED, AND COMMITTED, BEFORE ANY RETURN CONDITIONED ON THE STATE
IS READ. A plain run prints the instrument only; `--read` performs the trial, once.

The question (Guillaume, 2026-09-23). The crisis study (`docs/RESULTS_CRISE.md`)
coupled the state to objects that crash in stress and found no coupling useful. The
other way to use a crisis detector is to hold, during stress, something that gains in
crises. Three candidates, each a SWITCH: the equity leg in calm, the crisis leg in
stress, never both.

The legs, each an excess return of cash, each sized as everywhere in the programme,
weight min(0.10 / sigma_63, 3) on its own unit return, known at T-1
    EQ    US equity market, Ken French daily Mkt-RF (dividends included, already in
          excess). `data/raw/panels/factors_5.parquet`
    TLT   20+ year US Treasuries, iShares TLT total-return price (Yahoo, adjusted),
          minus the 3-month cash rate stamped at availability.
          `data/cache/trend_universe_m1.parquet`, from 2002-07-30
    GOLD  gold, Yahoo continuous front future GC=F, unadjusted, so its daily change
          approximates the spot return; minus the cash rate. Same file.
    LVOL  long volatility: the synthetic one-month S&P 500 variance swap of the
          crisis study, bought instead of sold (`crisis.variance_swap_ladder`, sign
          flipped). A delta-hedged straddle is the traded approximation.
          `data/raw/prices/cross_asset.parquet`

The family (three tests; nothing is added after reading)
    switch_TLT, switch_GOLD, switch_LVOL: weight on EQ when the state known at T-1 is
    calm, weight on the crisis leg when it is stress. Compared with EQ alone.
    Descriptive reference, not a test: STOP (cash in stress).

Controls, each read on every test
    - the same switch driven by the one-line volatility rules on ^GSPC: the median
      rule of the programme and the 80th-percentile rule of the crisis study, whose
      stress share is close to the state's;
    - a placebo: the state rotated by 400 random shifts of at least 252 sessions.

Sample: from 2002-04-01, or the first session on which every leg, the state and both
rules are known, to the last session of the shortest input (Ken French: 2026-07-31).
Zero cost (Guillaume, 2026-09-23); turnover is reported.

Decision rule, per test, identical to `scripts/run_crisis_coupling.py::verdict`
    delta = Sharpe(switch) - Sharpe(EQ alone), excess, annualised, zero cost
    MDE   = largest of the blinded paired bootstrap MDEs at blocks 21, 63, 126,
            demeaned legs, alpha 0.05/3
    USEFUL        delta >= MDE, paired HAC t of the same sign, placebo percentile
                  >= 95 %, and delta above the same switch's delta under BOTH
                  volatility rules
    NOT SHOWN     delta >= MDE but one of those fails
    UNDERPOWERED  0 < delta < MDE — never a success
    NOT USEFUL    delta <= 0 inside the MDE, or beyond it unconfirmed
    HARMFUL       delta <= -MDE, t of the same sign, placebo percentile <= 5 %
Also reported: max drawdown, worst month, and the return over the crisis windows of
the crisis study (GFC, 2009 rebound, February 2018, COVID, 2020 rebound, 2022).
A non-finite input or MDE stops the script before any verdict.

Usage
    .venv/bin/python scripts/run_safe_haven_switch.py          # instrument only
    .venv/bin/python scripts/run_safe_haven_switch.py --read   # the trial, once
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_crisis_coupling import (  # noqa: E402
    MIN_SHIFT,
    ROTATIONS,
    WINDOWS,
    market,
    max_drawdown,
    series,
    sha256,
    sharpe,
    verdict,
)
from run_m3_evaluation import cash_rate_daily  # noqa: E402

from regime_lab.analysis import trials  # noqa: E402
from regime_lab.config import CACHE, RAW  # noqa: E402
from regime_lab.evaluation.predictive import volatility_quantile_placebo  # noqa: E402
from regime_lab.extensions import crisis  # noqa: E402
from regime_lab.selection.protocol import blinded_mde, mde_at, paired_hac_t  # noqa: E402

RULE = "=" * 78
FAMILY = "safe_haven_switch"
STATE_COLUMN = "A' sparse jump"
OOS_START = pd.Timestamp("2002-04-01")
CRISIS_LEGS = ("TLT", "GOLD", "LVOL")
N_TESTS = len(CRISIS_LEGS)
ALPHA = 0.05 / N_TESTS
BLOCKS = (21, 63, 126)
DRAWS = 2000
INPUTS = (
    RAW / "panels" / "factors_5.parquet",
    CACHE / "trend_universe_m1.parquet",
    RAW / "prices" / "cross_asset.parquet",
    RAW / "macro" / "rate_cash_3m.parquet",
    CACHE / "states.parquet",
)


def units() -> dict[str, pd.Series]:
    eq = series(RAW / "panels" / "factors_5.parquet", "ff_mkt-rf") / 100.0
    prices = pd.read_parquet(CACHE / "trend_universe_m1.parquet")
    out = {"EQ": eq}
    for leg, ticker in (("TLT", "TLT"), ("GOLD", "GC=F")):
        px = prices[ticker].dropna()
        out[leg] = px.pct_change() - cash_rate_daily(px.index)
    m = market()
    out["LVOL"] = -crisis.variance_swap_ladder(m["log"], m["vix"])
    return {k: v.dropna().rename(k) for k, v in out.items()}


def build() -> dict:
    u = units()
    index = u["EQ"].index
    sized = {}
    for leg, unit in u.items():
        weight = crisis.vol_target_weight(unit)
        sized[leg] = (unit.reindex(index).fillna(0.0), weight.reindex(index))
    m = market()
    states = pd.read_parquet(CACHE / "states.parquet")[STATE_COLUMN]
    labels = {
        "state": crisis.lagged_state(states, index),
        "median rule": crisis.lagged_state(volatility_quantile_placebo(m["simple"]), index),
        "80th rule": crisis.lagged_state(crisis.volatility_tail_rule(m["simple"]), index),
    }
    ready = pd.Series(True, index=index)
    for _, w in sized.values():
        ready &= w.notna()
    for lab in labels.values():
        ready &= lab.notna()
    end = min(states.dropna().index.max(), index.max())
    start = max(OOS_START, ready[ready].index.min())
    window = index[(index >= start) & (index <= end)]
    return {
        "index": window,
        "legs": {k: (r.reindex(window), w.reindex(window)) for k, (r, w) in sized.items()},
        "labels": {k: v.reindex(window) for k, v in labels.items()},
        "not_ready_inside": int((~ready.reindex(window)).sum()),
    }


def switch(p: dict, leg: str, label: pd.Series) -> tuple[pd.Series, float]:
    """EQ in calm, ``leg`` in stress; returns and annual turnover of both weights."""
    stress = label.eq(crisis.STRESS)
    r_eq, w_eq = p["legs"]["EQ"]
    r_x, w_x = p["legs"][leg]
    held_eq = w_eq.where(~stress, 0.0)
    held_x = w_x.where(stress, 0.0)
    turnover = float((held_eq.diff().abs() + held_x.diff().abs()).mean() * 252)
    return held_eq * r_eq + held_x * r_x, turnover


def alone(p: dict) -> pd.Series:
    r_eq, w_eq = p["legs"]["EQ"]
    return w_eq * r_eq


def describe(x: pd.Series, turnover: float) -> dict:
    monthly = (1.0 + x).groupby(x.index.to_period("M")).prod() - 1.0
    row = {"sharpe": sharpe(x), "ann_return": float(x.mean() * 252),
           "ann_vol": float(x.std(ddof=1) * np.sqrt(252)), "max_dd": max_drawdown(x),
           "worst_month": float(monthly.min()), "turnover": turnover}
    for name, (a, b) in WINDOWS.items():
        part = x.loc[a:b]
        row[name] = float((1.0 + part).prod() - 1.0) if len(part) else float("nan")
    return row


def instrument(p: dict) -> dict[str, float]:
    print(RULE)
    print("EQUITIES IN CALM, A CRISIS ASSET IN STRESS — INSTRUMENT")
    print(RULE)
    print("\n0.  INPUTS (SHA-256)\n")
    for path in INPUTS:
        print(f"   {sha256(path)[:16]}  {path.relative_to(RAW.parent)}")
    idx = p["index"]
    print("\n1.  SAMPLE AND COVERAGE — no mean, no Sharpe\n")
    print(f"   {idx.min():%Y-%m-%d} -> {idx.max():%Y-%m-%d}, {len(idx):,} sessions, "
          f"not ready inside the window: {p['not_ready_inside']}")
    for name, lab in p["labels"].items():
        print(f"   {name:<12} stress share {lab.eq(crisis.STRESS).mean():>6.1%}")
    bad = p["not_ready_inside"] > 0
    for leg, (r, w) in p["legs"].items():
        nan = int(r.isna().sum() + w.isna().sum())
        bad |= nan > 0
        print(f"   {leg:<5} NaN {nan}   weight p50 {w.median():.2f}  p95 {w.quantile(0.95):.2f}"
              f"   at cap {(w >= crisis.MAX_LEVERAGE).mean():.1%}")

    print(f"\n2.  MINIMUM DETECTABLE EFFECT — demeaned legs, alpha 0.05/{N_TESTS} = "
          f"{ALPHA:.4f}\n")
    base = alone(p).to_numpy()
    mdes = {}
    for leg in CRISIS_LEGS:
        coupled, _ = switch(p, leg, p["labels"]["state"])
        values = [mde_at(blinded_mde(coupled.to_numpy(), base, mean_block=b, draws=DRAWS,
                                     seed=0), ALPHA) for b in BLOCKS]
        mdes[leg] = max(values)
        print(f"   switch_{leg:<5} MDE " + " / ".join(f"{v:.3f}" for v in values)
              + f"  ->  threshold {mdes[leg]:.3f}")
    if bad or not all(np.isfinite(v) for v in mdes.values()):
        print("\n   A READING IS NOT FINITE. No verdict.")
        return {}
    return mdes


def read(p: dict, mdes: dict[str, float]) -> None:
    fmt = {"sharpe": "{:+.2f}", "ann_return": "{:+.1%}", "ann_vol": "{:.1%}",
           "max_dd": "{:.1%}", "worst_month": "{:+.1%}", "turnover": "{:.1f}",
           **{c: "{:+.1%}" for c in WINDOWS}}
    idx = p["index"]
    eq = alone(p)
    _, w_eq = p["legs"]["EQ"]
    rows = {"EQ alone": describe(eq, float(w_eq.diff().abs().mean() * 252))}
    stop = w_eq.where(~p["labels"]["state"].eq(crisis.STRESS), 0.0)
    rows["stop, cash (ref.)"] = describe(stop * p["legs"]["EQ"][0],
                                         float(stop.diff().abs().mean() * 252))
    for leg in CRISIS_LEGS:
        for name, lab in p["labels"].items():
            x, turn = switch(p, leg, lab)
            rows[f"{leg} ({name})"] = describe(x, turn)
    table = pd.DataFrame(rows).T
    print(f"\n{RULE}\nREADING  {idx.min():%Y-%m-%d} -> {idx.max():%Y-%m-%d}, zero cost, "
          "excess of cash\n")
    print(f"   {'arm':<20}" + "".join(f"{c:>9}" for c in table.columns))
    for arm, r in table.iterrows():
        print(f"   {arm:<20}" + "".join(f"{fmt[c].format(r[c]):>9}" for c in table.columns))

    stress = p["labels"]["state"].eq(crisis.STRESS)
    print("\n   each leg alone, by lagged state (descriptive):")
    for leg, (r, w) in p["legs"].items():
        x = w * r
        print(f"     {leg:<5} stress Sharpe {sharpe(x[stress]):+.2f} "
              f"({x[stress].mean() * 252:+.1%}/yr)   calm Sharpe {sharpe(x[~stress]):+.2f} "
              f"({x[~stress].mean() * 252:+.1%}/yr)")

    rng = np.random.default_rng(20260923)
    shifts = rng.integers(MIN_SHIFT, len(idx) - MIN_SHIFT, size=ROTATIONS)
    base = sharpe(eq)
    values = p["labels"]["state"].to_numpy()
    print("\n   decision:")
    for leg in CRISIS_LEGS:
        coupled, _ = switch(p, leg, p["labels"]["state"])
        delta = sharpe(coupled) - base
        t = paired_hac_t(coupled, eq)
        null = np.array([sharpe(switch(p, leg, pd.Series(np.roll(values, k), index=idx))[0])
                         - base for k in shifts])
        pct = float((null < delta).mean()) if np.isfinite(null).all() else float("nan")
        d_med = sharpe(switch(p, leg, p["labels"]["median rule"])[0]) - base
        d_tail = sharpe(switch(p, leg, p["labels"]["80th rule"])[0]) - base
        ddd = max_drawdown(coupled) - max_drawdown(eq)
        v = verdict(delta, mdes[leg], t, pct, d_med, d_tail)
        print(f"   switch_{leg:<5} delta {delta:+.3f}  MDE {mdes[leg]:.3f}  t {t:+.2f}  "
              f"placebo pct {pct:.1%} (p5 {np.quantile(null, 0.05):+.3f}, "
              f"p95 {np.quantile(null, 0.95):+.3f})")
        print(f"         vol rules: median {d_med:+.3f}, 80th {d_tail:+.3f}   "
              f"change in max DD {ddd:+.1%}")
        print(f"         => {v}")
        trials.log(
            FAMILY,
            {"test": f"switch_{leg}", "calm_leg": "EQ (Ken French Mkt-RF)", "stress_leg": leg,
             "state": "A' sparse jump filtered, lag 1", "cost": "zero",
             "sizing": "min(0.10/sigma63, 3) per leg", "alpha": f"0.05/{N_TESTS}",
             "sample": [str(idx.min().date()), str(idx.max().date())]},
            {"sharpe": sharpe(coupled), "sharpe_alone": base, "delta": delta,
             "threshold": mdes[leg], "t_hac": t, "placebo_pct": pct,
             "delta_median_rule": d_med, "delta_tail_rule": d_tail, "delta_max_dd": ddd,
             "sessions": int(len(idx)), "verdict": v},
        )
    print(f"\n   logged to data/trials.parquet ({trials.summary()['n_distinct']} distinct)")
    print(RULE)


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

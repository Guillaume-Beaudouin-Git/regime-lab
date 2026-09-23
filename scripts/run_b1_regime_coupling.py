"""Month-end Treasury rally, alone and coupled with the sparse jump model's state.

A question for the course presentation: does the study's classifier change what a
plain, documented strategy delivers, once it is coupled to it? The strategy is the
month-end Treasury rally, a flow effect around index rebalancing: long a long-duration
Treasury over the last three trading days of each calendar month.

EVERYTHING BELOW IS FIXED, AND COMMITTED, BEFORE ANY RETURN IS READ.

The strategy (B1)
    instrument   TLT, total-return series of `data/cache/trend_universe_m1.parquet`,
                 funded at the 3-month cash rate like every bond ETF of the programme
                 (`FUNDED_ON_CASH`), so its returns are excess returns
    window       entry at the close of the 4th-to-last US trading day of the month, exit
                 at the close of the last one: the position earns the returns of the
                 last three US trading days (dend 3, 2, 1). US trading days are the
                 sessions of ^GSPC in `data/raw/prices/cross_asset.parquet`
    sizing       daily volatility targeting, as everywhere in the programme:
                 min(0.10 / sigma_63 * sqrt(7), 3), where sigma_63 is TLT's 63-session
                 realised volatility lagged one session. sqrt(7) because the leg is
                 invested about 3 days in 21, so that its annualised volatility is
                 near the 10% the trend book targets and neither leg of the switch
                 dominates the risk by construction
    costs        the programme's schedule: TLT is priced as a Treasury future, 1 bp
                 round trip (headline, decides), 2 bp (conservative), 5 bp (stress),
                 charged on |dw| per instrument (`extensions/vehicle.py`)

The state
    A' sparse jump, filtered (out of sample), `data/cache/states.parquet`: 0 = stress,
    1 = calm (states ordered by training volatility, `models/mapping.py`). The state
    used on session d is the last one stamped at or before d-1: signal at T-1.

The two couplings
    (b) REDUCE   weights of B1 multiplied by 0.5 on sessions whose lagged state is stress
    (c) SWITCH   on stress sessions, hold the programme's trend book instead of B1: the
                 M3 book of `extensions/vehicle.py` (46 instruments, 12-1 trend, 10%
                 portfolio volatility target), the committed headline construction. On
                 calm sessions, hold B1. Chosen before any return is read because trend
                 following is the documented "crisis alpha" family (Hurst, Ooi and
                 Pedersen 2017), because this book is built and audited here, and
                 because its data covers every stress episode of the sample. Switching
                 costs are charged exactly: the arm's weight matrix is costed as a whole

The controls, each run through the same code
    - the one-line volatility rule the programme sets against every regime device:
      stress when the 21-session realised volatility of ^GSPC is above its own
      expanding median (`evaluation.predictive.volatility_quantile_placebo`),
      coupled in the same two ways;
    - a matched placebo: the state series rotated circularly against the dates, 400
      rotations of at least 252 sessions, each coupling rebuilt on each rotation. It
      keeps the state's clock and occupancy exactly and destroys only its timing.

Sample: the sessions on which every arm is defined, starting 63 sessions after the M3
book goes live, so that its volatility estimate is not the burn-in artefact the A4
audit found (43.8x gross on 2004-07-07). This drops the 2002-2003 stress episode.

THE DECISION, WRITTEN BEFORE THE NUMBER
    For each coupling X in {REDUCE, SWITCH}, against B1 alone, net of headline costs, in
    excess of cash, over the common sample:
      delta = Sharpe(X) - Sharpe(B1)
      MDE   = the paired, demeaned-leg bootstrap MDE (`selection.protocol.blinded_mde`),
              blocks 21 / 63 / 126, the largest, at alpha 0.05 / 2 (two couplings)
    The filter is called USEFUL for B1 under X only if all four hold:
      1. delta >= MDE;
      2. the HAC lag-6 t of the daily net difference has the sign of delta;
      3. delta is above the 95th percentile of the rotation placebo;
      4. delta beats the same coupling built on the one-line volatility rule.
    0 < delta < MDE is UNDERPOWERED, never useful. delta <= 0 is NOT USEFUL. A delta
    whose sign changes between 1 bp and 2 bp is UNDECIDED.
    Reported beside the decision, never deciding: annual return and volatility, maximum
    drawdown, worst calendar month, and the return of each arm inside three crises
    dated by the S&P 500's own peaks and troughs, not by the state:
      GFC   2007-10-09 -> 2009-03-09
      COVID 2020-02-19 -> 2020-03-23
      2022  2022-01-03 -> 2022-10-12
    The placebo percentile of the change in maximum drawdown is reported as well.

Declared priors, before the reading. B1 is a calendar flow effect; nothing makes it
depend on the volatility regime, so REDUCE is expected to change little except risk in
the stress months. SWITCH changes the strategy's nature on about 15% of sessions and is
expected to change the crisis rows most; its Sharpe effect is not predicted.

Discipline, as in `scripts/run_ahl_level_b.py`: run plain, the script measures the
instrument and prints no return. `--read` prints the results and logs two trial rows
(family `b1_regime_coupling`) to `data/trials.parquet`; run it once.

Usage:
    .venv/bin/python scripts/run_b1_regime_coupling.py           # instrument only
    .venv/bin/python scripts/run_b1_regime_coupling.py --read    # the reading, a trial
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from regime_lab.analysis import trials
from regime_lab.config import CACHE, RAW
from regime_lab.evaluation.predictive import volatility_quantile_placebo
from regime_lab.extensions.vehicle import vol_targeted_book
from regime_lab.selection.protocol import blinded_mde, mde_at, paired_hac_t

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_m3_evaluation import SAMPLE, active_window, cash_rate_daily, net_excess  # noqa: E402

RULE = "=" * 78
INSTRUMENT = "TLT"
WINDOW_DAYS = 3
VOL_WINDOW = 63
SCALE = np.sqrt(7.0)
CAP = 3.0
REDUCE = 0.5
PURGE = 63
ALPHA = 0.05 / 2
BLOCKS = (21, 63, 126)
ROTATIONS = 400
MIN_SHIFT = 252
CRISES = {
    "GFC": ("2007-10-09", "2009-03-09"),
    "COVID": ("2020-02-19", "2020-03-23"),
    "2022": ("2022-01-03", "2022-10-12"),
}
STATE_COLUMN = "A' sparse jump"
STRESS = 0.0


def us_sessions() -> pd.DatetimeIndex:
    frame = pd.read_parquet(RAW / "prices" / "cross_asset.parquet")
    spx = frame[frame["series_id"] == "eq_us_large"]
    return pd.DatetimeIndex(pd.to_datetime(spx["period"])).sort_values().unique()


def us_prices(series_id: str) -> pd.Series:
    frame = pd.read_parquet(RAW / "prices" / "cross_asset.parquet")
    one = frame[frame["series_id"] == series_id].set_index("period")["value"]
    return one.sort_index().astype(float)


def month_end_window(index: pd.DatetimeIndex, us: pd.DatetimeIndex) -> pd.Series:
    """1 on every session of `index` inside (entry, exit] of a month-end window.

    Entry is the close of the (WINDOW_DAYS + 1)-th-to-last US trading day of the
    month, exit the close of the last one. Holding by date rather than by US session
    keeps the position through a session on which only other markets trade. A month
    that ends the data before its last business day is incomplete and carries no
    window; a US holiday on a month's last business day inside the data does not.
    """
    us = us[(us >= index.min()) & (us <= index.max())]
    months = us.to_period("M")
    rank = pd.Series(0, index=us).groupby(months).cumcount(ascending=False) + 1
    flag = pd.Series(0.0, index=index)
    for month, dates in pd.Series(us, index=us).groupby(months):
        last = dates.max()
        if month == months.max() and last < last + pd.offsets.BMonthEnd(0):
            continue
        ranks = rank.loc[dates.index]
        entry = ranks.index[ranks == WINDOW_DAYS + 1]
        exit_ = ranks.index[ranks == 1]
        if len(entry) and len(exit_):
            flag[(index > entry[0]) & (index <= exit_[0])] = 1.0
    return flag


def lagged_state(labels: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """The last label stamped at or before d-1, for every session d of `index`."""
    known = labels.dropna().sort_index()
    asof = known.reindex(known.index.union(index)).ffill().reindex(index)
    return asof.shift(1)


def b1_weights(prices: pd.DataFrame, us: pd.DatetimeIndex) -> pd.DataFrame:
    returns = prices[INSTRUMENT].pct_change()
    us_ret = returns.reindex(us).dropna()
    sigma = (us_ret.rolling(VOL_WINDOW).std() * np.sqrt(252)).reindex(prices.index).ffill()
    size = (0.10 / sigma.replace(0.0, np.nan) * SCALE).clip(upper=CAP).shift(1)
    weight = month_end_window(prices.index, us) * size
    out = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    out[INSTRUMENT] = weight.fillna(0.0)
    return out


def coupled(base: pd.DataFrame, stress: pd.Series, mode: str, trend: pd.DataFrame) -> pd.DataFrame:
    is_stress = stress.eq(STRESS).reindex(base.index).fillna(False).to_numpy()[:, None]
    if mode == "reduce":
        return base * np.where(is_stress, REDUCE, 1.0)
    if mode == "switch":
        return pd.DataFrame(
            np.where(is_stress, trend.to_numpy(), base.to_numpy()),
            index=base.index, columns=base.columns,
        )
    raise ValueError(mode)


def arm_returns(weights: pd.DataFrame, returns: pd.DataFrame, rate: pd.Series,
                column: str = "headline") -> pd.Series:
    gross = (weights * returns).sum(axis=1)
    return net_excess(gross, weights, rate, column=column)


def sharpe(x: pd.Series) -> float:
    v = x.dropna().to_numpy(float)
    return float(v.mean() / v.std(ddof=1) * np.sqrt(252)) if v.std(ddof=1) > 0 else np.nan


def max_drawdown(x: pd.Series) -> float:
    curve = (1.0 + x.fillna(0.0)).cumprod()
    return float((curve / curve.cummax() - 1.0).min())


def describe(x: pd.Series) -> dict:
    monthly = (1.0 + x).groupby(x.index.to_period("M")).prod() - 1.0
    row = {
        "sharpe": sharpe(x),
        "ann_return": float(x.mean() * 252),
        "ann_vol": float(x.std(ddof=1) * np.sqrt(252)),
        "max_dd": max_drawdown(x),
        "worst_month": float(monthly.min()),
    }
    for name, (a, b) in CRISES.items():
        row[name] = float((1.0 + x.loc[a:b]).prod() - 1.0)
    return row


def build() -> dict:
    prices = pd.read_parquet(CACHE / "trend_universe_m1.parquet").loc[SAMPLE]
    returns = prices.pct_change()
    us = us_sessions()
    m3 = vol_targeted_book(prices)
    live = active_window(m3["returns"]).index
    start = live[PURGE]
    states = pd.read_parquet(CACHE / "states.parquet")[STATE_COLUMN]
    state = lagged_state(states, prices.index)
    spx = us_prices("eq_us_large")
    vol_rule = volatility_quantile_placebo(spx.pct_change())
    vol_state = lagged_state(vol_rule, prices.index)
    end = min(states.dropna().index.max(), prices.index.max())
    index = prices.index[(prices.index >= start) & (prices.index <= end)]
    index = index[state.reindex(index).notna() & vol_state.reindex(index).notna()]
    rate = cash_rate_daily(prices.index)
    base = b1_weights(prices, us)
    return {
        "prices": prices, "returns": returns, "index": index, "rate": rate,
        "base": base, "trend": m3["weights"], "state": state, "vol_state": vol_state,
        "window": month_end_window(prices.index, us),
    }


def arms(p: dict, state: pd.Series, column: str = "headline") -> dict[str, pd.Series]:
    out = {"B1": p["base"]}
    for mode in ("reduce", "switch"):
        out[mode] = coupled(p["base"], state, mode, p["trend"])
    return {
        name: arm_returns(w, p["returns"], p["rate"], column).reindex(p["index"])
        for name, w in out.items()
    }


def main() -> None:
    read_the_answer = "--read" in sys.argv
    p = build()
    idx = p["index"]
    stress_share = float(p["state"].reindex(idx).eq(STRESS).mean())
    windows = p["window"].reindex(idx)
    n_windows = int((windows.diff() == 1).sum())
    print(RULE)
    print("MONTH-END TREASURY RALLY, ALONE AND COUPLED WITH THE SPARSE JUMP STATE")
    print(RULE)
    print(f"\n   sample        {idx.min():%Y-%m-%d} to {idx.max():%Y-%m-%d}, {len(idx):,} sessions")
    print(f"   month-ends    {n_windows} windows of {WINDOW_DAYS} US trading days")
    print(f"   state         stress on {stress_share:.1%} of sessions (A' sparse jump, lagged)")
    vol_share = float(p["vol_state"].reindex(idx).eq(STRESS).mean())
    print(f"   vol rule      stress on {vol_share:.1%} of sessions (21-session vol > median)")
    in_stress = int(((windows > 0) & p["state"].reindex(idx).eq(STRESS)).sum())
    print(f"   B1 sessions   {int((windows > 0).sum())} invested, {in_stress} of them in stress")

    legs = arms(p, p["state"])
    print(f"\n{RULE}\n1.  THE INSTRUMENT — MDE at alpha {ALPHA:.3f}, demeaned legs, "
          "no return read\n")
    mdes = {}
    for mode in ("reduce", "switch"):
        values = []
        for block in BLOCKS:
            res = blinded_mde(legs[mode].to_numpy(), legs["B1"].to_numpy(),
                              mean_block=block, draws=2000, seed=0)
            values.append(mde_at(res, ALPHA))
        mdes[mode] = max(values)
        joined = " / ".join(f"{v:.3f}" for v in values)
        print(f"   {mode:<7} MDE {joined} at blocks 21/63/126  ->  threshold {mdes[mode]:.3f}")
    if any(not np.isfinite(v) for v in mdes.values()):
        print("\n   A READING IS NOT FINITE. No verdict.")
        return

    if not read_the_answer:
        print(f"\n{RULE}\n2.  NOT READ — re-run with --read, once.\n{RULE}")
        return
    read(p, legs, mdes)


def read(p: dict, legs: dict[str, pd.Series], mdes: dict[str, float]) -> None:
    idx = p["index"]
    cons = arms(p, p["state"], "conservative")
    vol_legs = arms(p, p["vol_state"])
    trend_alone = arm_returns(p["trend"], p["returns"], p["rate"]).reindex(idx)

    print(f"\n{RULE}\n2.  THE READING — net of 1 bp, excess of cash\n")
    rows = {name: describe(x) for name, x in legs.items()}
    rows["vol rule reduce"] = describe(vol_legs["reduce"])
    rows["vol rule switch"] = describe(vol_legs["switch"])
    rows["trend book alone"] = describe(trend_alone)
    table = pd.DataFrame(rows).T
    fmt = {"sharpe": "{:+.2f}", "ann_return": "{:+.2%}", "ann_vol": "{:.2%}",
           "max_dd": "{:.1%}", "worst_month": "{:+.1%}",
           **{c: "{:+.1%}" for c in CRISES}}
    header = f"   {'arm':<18}" + "".join(f"{c:>12}" for c in table.columns)
    print(header)
    for name, r in table.iterrows():
        print(f"   {name:<18}" + "".join(f"{fmt[c].format(r[c]):>12}" for c in table.columns))

    rng = np.random.default_rng(20260923)
    shifts = rng.integers(MIN_SHIFT, len(idx) - MIN_SHIFT, size=ROTATIONS)
    state_values = p["state"].reindex(idx)
    print(f"\n{RULE}\n3.  THE DECISION — per coupling, against B1 alone\n")
    for mode in ("reduce", "switch"):
        delta = sharpe(legs[mode]) - sharpe(legs["B1"])
        delta_cons = sharpe(cons[mode]) - sharpe(cons["B1"])
        t = paired_hac_t(legs[mode], legs["B1"])
        delta_vol = sharpe(vol_legs[mode]) - sharpe(vol_legs["B1"])
        null, null_dd = [], []
        for s in shifts:
            rotated = pd.Series(np.roll(state_values.to_numpy(), s), index=idx)
            rleg = arm_returns(coupled(p["base"], rotated, mode, p["trend"]),
                               p["returns"], p["rate"]).reindex(idx)
            null.append(sharpe(rleg) - sharpe(legs["B1"]))
            null_dd.append(max_drawdown(rleg) - max_drawdown(legs["B1"]))
        null, null_dd = np.array(null), np.array(null_dd)
        pct = float((null < delta).mean())
        ddelta = max_drawdown(legs[mode]) - max_drawdown(legs["B1"])
        pct_dd = float((null_dd < ddelta).mean())
        if not all(np.isfinite([delta, t, delta_vol, pct])) or not np.isfinite(null).all():
            verdict = "NOT FINITE — no verdict"
        elif np.sign(delta) != np.sign(delta_cons):
            verdict = "UNDECIDED — the sign changes between 1 and 2 bp"
        elif delta <= 0:
            verdict = "NOT USEFUL — the coupling lowers the Sharpe"
        elif delta < mdes[mode]:
            verdict = "UNDERPOWERED — positive but below the MDE, never useful"
        elif np.sign(t) != np.sign(delta) or pct < 0.95 or delta <= delta_vol:
            verdict = "NOT SHOWN — above the MDE but fails the t, the placebo or the vol rule"
        else:
            verdict = "USEFUL — all four conditions hold"
        print(f"   {mode.upper()}")
        print(f"     delta Sharpe      {delta:+.3f}   (2 bp: {delta_cons:+.3f})   "
              f"MDE {mdes[mode]:.3f}")
        print(f"     HAC-6 t           {t:+.2f}")
        print(f"     placebo pct       {pct:.1%}  (rotation null: median {np.median(null):+.3f},"
              f" p95 {np.quantile(null, 0.95):+.3f})")
        print(f"     vol rule delta    {delta_vol:+.3f}")
        print(f"     change in max DD  {ddelta:+.1%}  (placebo pct {pct_dd:.1%})")
        print(f"     => {verdict}\n")
        trials.log(
            "b1_regime_coupling",
            {"test": mode, "strategy": "month-end Treasury, TLT, last 3 US days",
             "state": "A' sparse jump filtered, lag 1", "reduce": REDUCE,
             "switch_to": "M3 trend book headline", "cost_bps": "schedule headline (1 bp)",
             "sample": [str(idx.min().date()), str(idx.max().date())]},
            {"sharpe": sharpe(legs[mode]), "delta": delta, "threshold": mdes[mode],
             "t_hac": t, "placebo_pct": pct, "delta_vol_rule": delta_vol,
             "delta_max_dd": ddelta, "sessions": int(len(idx)), "verdict": verdict},
        )
    print(f"   logged to data/trials.parquet ({trials.summary()['n_distinct']} distinct)")
    print(RULE)


if __name__ == "__main__":
    main()

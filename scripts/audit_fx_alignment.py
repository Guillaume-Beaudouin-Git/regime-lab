"""Which spot FX series in the trend universe are stamped a session late.

`docs/PRESPEC_TREND_VEHICLE.md` §1 D3 records that the spot FX series lag the
market by one session, on evidence from three of them — EURUSD, GBPUSD and AUDUSD
against the matching CME futures — and infers the rest from a shared data
endpoint. Only 6A, 6B and 6E are held locally, so the other seven cannot be
checked the same way, and an inference is not a measurement. The pre-registration
locks the repair of *ten* series. This script asks which ones actually need it.

Two independent tests, neither requiring data this repository does not already
hold. They are run together because either alone is arguable.

Test 1 — dating against shocks with a settled calendar date. Each event below
moved its currency by many standard deviations in one session, so the session a
correctly stamped series records it on is not in doubt. Rule fixed before any
output was read: take the largest absolute log return in the eleven sessions
centred on the event and report its offset in trading sessions. A move smaller
than four trailing standard deviations is not decisive and is not counted.

Test 2 — lead-lag against `DX-Y.NYB`, the ICE dollar index, which quotes on a
different venue and is built on six of these currencies. If a spot series is
stamped late, its correlation with *yesterday's* dollar index must exceed its
correlation with today's. `^GSPC` is carried as a control: an equity index is
aligned, so it must show the opposite pattern.

Usage:
    .venv/bin/python scripts/audit_fx_alignment.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from regime_lab.config import CACHE

RULE = "=" * 92
SIGMA_GATE = 4.0
REFERENCE = "DX-Y.NYB"
CONTROL = "^GSPC"

#: (label, settled date, series the event is decisive for)
EVENTS: list[tuple[str, str, list[str]]] = [
    ("SNB removes the EUR/CHF floor", "2015-01-15", ["CHF=X"]),
    (
        "Brexit referendum result",
        "2016-06-24",
        [
            "GBPUSD=X", "EURUSD=X", "AUDUSD=X", "NZDUSD=X", "JPY=X",
            "CAD=X", "CHF=X", "SEK=X", "NOK=X",
        ],
    ),
    ("US presidential election", "2016-11-09", ["MXN=X", "JPY=X"]),
    ("BoJ yield-curve shock", "2022-12-20", ["JPY=X"]),
]


def log_returns(frame: pd.DataFrame) -> pd.DataFrame:
    """Log returns on strictly positive prices; anything else becomes NaN."""
    positive = frame.where(frame > 0)
    return np.log(positive).diff()


def event_offsets(returns: pd.DataFrame, sigma: pd.DataFrame, index: pd.DatetimeIndex) -> dict:
    tally: dict[str, list[int]] = {}
    for label, date, series in EVENTS:
        event = pd.Timestamp(date)
        print(f"\n{RULE}\n{label}  —  {date}\n{RULE}")
        print(
            f"   {'series':<12} {'biggest move':>13} {'on':>12} "
            f"{'offset':>8} {'n sigma':>9}   verdict"
        )
        anchor = index[min(index.searchsorted(event), len(index) - 1)]
        for col in series:
            if col not in returns.columns:
                continue
            pos = index.get_loc(anchor)
            window = index[max(0, pos - 5) : pos + 6]
            segment = returns.loc[window, col].dropna()
            if segment.empty:
                print(f"   {col:<12} {'no data':>13}")
                continue
            peak = segment.abs().idxmax()
            move = returns.loc[peak, col]
            scale = sigma.loc[peak, col]
            n_sigma = abs(move) / scale if scale and not np.isnan(scale) else np.nan
            offset = index.get_loc(peak) - pos
            decisive = n_sigma >= SIGMA_GATE
            if decisive:
                tally.setdefault(col, []).append(offset)
            verdict = "not decisive"
            if decisive:
                verdict = "ALIGNED" if offset == 0 else f"LATE by {offset}"
            print(
                f"   {col:<12} {move * 100:>12.2f}% {peak:%Y-%m-%d} {offset:>8d} "
                f"{n_sigma:>9.1f}   {verdict}"
            )
    return tally


def lead_lag(returns: pd.DataFrame, spot: list[str]) -> dict[str, float]:
    reference = returns[REFERENCE]
    print(f"\n{RULE}\nTEST 2 — lead-lag against {REFERENCE}, the ICE dollar index\n{RULE}")
    print(
        f"\n   {'series':<12} {'corr(ref_t, x_t)':>18} "
        f"{'corr(ref_t, x_t+1)':>20} {'ratio':>8}   reading"
    )
    ratios: dict[str, float] = {}
    for col in spot + [CONTROL]:
        pair = pd.concat([reference.rename("r"), returns[col].rename("x")], axis=1).dropna()
        same = pair["r"].corr(pair["x"])
        lagged = pair["r"].corr(pair["x"].shift(-1))
        ratio = abs(lagged) / abs(same) if same else np.nan
        if col != CONTROL:
            ratios[col] = ratio
        reading = "control, must read aligned" if col == CONTROL else (
            "LATE" if abs(lagged) > abs(same) else "aligned"
        )
        print(f"   {col:<12} {same:>18.3f} {lagged:>20.3f} {ratio:>8.2f}   {reading}")
    return ratios


def main() -> None:
    universe = pd.read_parquet(CACHE / "trend_universe.parquet")
    spot = [c for c in universe.columns if c.endswith("=X")]
    returns = log_returns(universe)
    sigma = returns.rolling(60).std()

    print(RULE)
    print("IS EACH SPOT FX SERIES STAMPED A SESSION LATE?")
    print(RULE)
    print(f"\n   universe {universe.index.min():%Y-%m-%d} to {universe.index.max():%Y-%m-%d}")
    print(f"   spot FX series: {len(spot)} — {', '.join(spot)}")
    print(f"\n{RULE}\nTEST 1 — dating against shocks with a settled calendar date\n{RULE}")

    tally = event_offsets(returns, sigma, universe.index)
    ratios = lead_lag(returns, spot)

    print(f"\n{RULE}\nVERDICT — the two tests together\n{RULE}")
    print(f"\n   {'series':<12} {'event test':>22} {'lead-lag':>12}   conclusion")
    late, aligned, unresolved = [], [], []
    for col in spot:
        offsets = tally.get(col, [])
        if not offsets:
            event_read = "no decisive event"
        elif sorted(set(offsets)) == [1]:
            event_read = "late by one"
        elif sorted(set(offsets)) == [0]:
            event_read = "aligned"
        else:
            event_read = f"mixed {sorted(set(offsets))}"
        ratio = ratios[col]
        lag_read = "late" if ratio > 1.0 else "aligned"
        agree = (event_read == "late by one" and lag_read == "late") or (
            event_read == "aligned" and lag_read == "aligned"
        )
        if event_read == "no decisive event":
            conclusion = f"{lag_read.upper()}, on the lead-lag test alone"
        elif agree:
            conclusion = f"{lag_read.upper()}, both tests agree"
        else:
            conclusion = "UNRESOLVED, the two tests disagree"
        (late if lag_read == "late" else aligned).append(col)
        if conclusion.startswith("UNRESOLVED"):
            unresolved.append(col)
        print(f"   {col:<12} {event_read:>22} {ratio:>11.2f}   {conclusion}")

    print(f"\n   repair ({len(late)}): {', '.join(late)}")
    print(f"   leave alone ({len(aligned)}): {', '.join(aligned)}")
    if unresolved:
        print(f"   unresolved: {', '.join(unresolved)}")
    print(
        "\n   The pre-registration locks a repair of ten series. It is wrong on "
        f"{len(aligned)}:\n   shifting those would introduce the defect it is meant to remove."
    )
    print("\n" + RULE)


if __name__ == "__main__":
    main()

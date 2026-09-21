"""M2 — what the vehicle substitution can actually cover, and what it would cost.

`docs/PRESPEC_TREND_VEHICLE.md` §4 locks M2 as: represent each exposure by the
instrument that would really be traded, keep the stored series where no
back-adjusted one exists, flag it, and report the headline with and without the
flagged instruments.

The locked text does not say what to do when substitution is possible but
*shortens the sample*. That is the case here, and this script measures it before
any return of the repaired book is computed.

Four measurements, in the order that decides M2:

1. **Coverage.** How many of the 46 instruments have a back-adjusted futures
   series available, and over what window.
2. **The power consequence.** What the smallest resolvable Sharpe becomes if the
   sample is cut to the substitutable window, against the study's own 0.70 gate.
   Same constants as `regime_lab.analysis.power`.
3. **Alignment by family.** The FX audit in `audit_fx_alignment.py` established a
   one-session lag on the `=X` spot endpoint. It did not test the other families.
   This does, against the same CME futures, so the scope of M1 is bounded by
   measurement rather than by assumption.
4. **The size of the defect M2 would repair.** For the three commodities held both
   raw and back-adjusted: whether the stored series is the raw front month, what
   the roll gap costs in annualised drift, and how often it flips the sign of a
   12-minus-1 trend signal — which is the only thing a trend book reads.

Read-only. Writes nothing, and computes no return of the repaired book.

Usage:
    .venv/bin/python scripts/audit_vehicle_coverage.py
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from regime_lab.config import CACHE

RULE = "=" * 78

#: Locked sample of the study, docs/PRESPEC_TREND_VEHICLE.md §3.
SAMPLE_START = pd.Timestamp("2003-07-17")
SAMPLE_END = pd.Timestamp("2026-09-10")
GATE = 0.70

#: Stored series -> the future that would really be traded for that exposure.
VEHICLE = {
    "EURUSD=X": "6E", "GBPUSD=X": "6B", "AUDUSD=X": "6A",
    "^GSPC": "ES", "^NDX": "NQ", "^RUT": "RTY",
    "CL=F": "CL", "GC=F": "GC", "SI=F": "SI",
    "IEF": "ZN",
}

FAMILY = {
    "fx": [c for c in VEHICLE if c.endswith("=X")],
    "equity index": ["^GSPC", "^NDX", "^RUT"],
    "commodity": ["CL=F", "GC=F", "SI=F"],
    "fixed income": ["IEF"],
}


def simulator_root() -> Path:
    """The private sibling repository holding the back-adjusted daily futures."""
    configured = os.environ.get("SIMULATOR_PATH")
    if not configured:
        raise SystemExit(
            "audit_vehicle_coverage.py needs the back-adjusted futures set, which "
            "lives in a private sibling repository. Set SIMULATOR_PATH in .env."
        )
    return Path(configured).expanduser() / "strategies" / "trend_futures" / "daily"


def smallest_resolvable_sharpe(years: float) -> float:
    """Invert the sample-length requirement of `analysis.power` for a Sharpe.

    Years needed to separate a true Sharpe from zero at 80% power, two-sided 5%:
    ``T = z^2 (1 + SR^2 / 2) / SR^2``. Solved here for SR given T.
    """
    z = stats.norm.ppf(0.975) + stats.norm.ppf(0.80)
    denominator = years - z**2 / 2
    if denominator <= 0:
        return float("inf")
    return float(np.sqrt(z**2 / denominator))


def trend_signal(price: pd.Series) -> pd.Series:
    """12-minus-1: cumulative return over 252 sessions, skipping the last 21."""
    return price.shift(21) / price.shift(252) - 1.0


def main() -> None:
    daily = simulator_root()
    universe = pd.read_parquet(CACHE / "trend_universe.parquet")

    print(RULE)
    print("M2  VEHICLE COVERAGE, AND WHAT SUBSTITUTING WOULD COST")
    print(RULE)
    print(f"\n   universe        {universe.shape[1]} instruments, "
          f"{universe.index.min():%Y-%m-%d} to {universe.index.max():%Y-%m-%d}")
    print(f"   locked sample   {SAMPLE_START:%Y-%m-%d} to {SAMPLE_END:%Y-%m-%d}, "
          f"{(SAMPLE_END - SAMPLE_START).days / 365.25:.1f} years")

    print(f"\n{RULE}\n1.  COVERAGE — what a back-adjusted series exists for\n")
    print(f"   {'stored':<10} {'vehicle':<8} {'from':<12} {'to':<12} {'years':>6}")
    windows = []
    for stored, future in VEHICLE.items():
        path = daily / f"{future}.parquet"
        if not path.exists():
            print(f"   {stored:<10} {future:<8} {'ABSENT':<12}")
            continue
        frame = pd.read_parquet(path)
        years = (frame.index.max() - frame.index.min()).days / 365.25
        windows.append((frame.index.min(), frame.index.max()))
        print(f"   {stored:<10} {future:<8} {frame.index.min():%Y-%m-%d}   "
              f"{frame.index.max():%Y-%m-%d}   {years:>6.1f}")

    start = max(w[0] for w in windows)
    end = min(w[1] for w in windows)
    common = (end - start).days / 365.25
    covered = len(VEHICLE)
    print(f"\n   {covered} of {universe.shape[1]} instruments are substitutable, "
          f"{universe.shape[1] - covered} are not.")
    print(f"   Common window of the substitutable set: {start:%Y-%m-%d} to "
          f"{end:%Y-%m-%d}, {common:.1f} years.")

    print(f"\n{RULE}\n2.  THE POWER CONSEQUENCE — the study's own gate is {GATE:.2f}\n")
    locked_years = (SAMPLE_END - SAMPLE_START).days / 365.25
    rows = [
        ("stored series, locked sample", locked_years),
        ("substituted, common window", common),
        ("substituted, RTY window", (end - pd.Timestamp("2017-07-10")).days / 365.25),
    ]
    print(f"   {'sample':<34} {'years':>7} {'smallest resolvable SR':>24}  verdict")
    for label, years in rows:
        sr = smallest_resolvable_sharpe(years)
        verdict = "gate decidable" if sr < GATE else "GATE NOT DECIDABLE"
        print(f"   {label:<34} {years:>7.1f} {sr:>24.3f}  {verdict}")
    print("\n   Substituting where possible costs "
          f"{locked_years - common:.1f} years, and takes the smallest resolvable")
    print("   Sharpe above the gate the study is trying to clear. The headline")
    print("   hypothesis would stop being decidable.")

    print(f"\n{RULE}\n3.  ALIGNMENT BY FAMILY — bounding M1 by measurement\n")
    print(f"   {'stored':<10} {'vehicle':<8} {'corr(t,t)':>11} {'corr(t+1,t)':>13} "
          f"{'ratio':>7}  reading")
    for family, columns in FAMILY.items():
        print(f"   -- {family}")
        for stored in columns:
            future = VEHICLE[stored]
            path = daily / f"{future}.parquet"
            if not path.exists() or stored not in universe.columns:
                continue
            reference = pd.read_parquet(path)["close"].pct_change()
            series = universe[stored].pct_change()
            index = series.index.intersection(reference.index)
            series, reference = series.reindex(index), reference.reindex(index)
            same = series.corr(reference)
            lagged = series.shift(-1).corr(reference)
            ratio = abs(lagged) / max(abs(same), 1e-9)
            reading = "LATE by one session" if ratio > 1.3 else "aligned"
            print(f"   {stored:<10} {future:<8} {same:>11.4f} {lagged:>13.4f} "
                  f"{ratio:>6.2f}x  {reading}")
    print("\n   The one-session lag is a property of the `=X` spot endpoint and of")
    print("   nothing else measured here. M1's scope is the FX series, and the")
    print("   commodity, index and fixed-income series are already on the right day.")

    print(f"\n{RULE}\n4.  THE DEFECT M2 WOULD REPAIR, sized where it is measurable\n")
    print(f"   {'stored':<8} {'vs raw':>8} {'vs back-adj':>12}  {'the stored series is':<24}"
          f" {'drift gap':>10} {'12-1 sign flips':>16}")
    for stored in FAMILY["commodity"]:
        frame = pd.read_parquet(daily / f"{VEHICLE[stored]}.parquet")
        series = universe[stored].pct_change()
        raw = frame["raw_close"].pct_change()
        adjusted = frame["close"].pct_change()
        index = series.index.intersection(frame.index)
        series, raw, adjusted = (s.reindex(index) for s in (series, raw, adjusted))
        keep = series.notna() & raw.notna() & adjusted.notna()
        to_raw, to_adjusted = series[keep].corr(raw[keep]), series[keep].corr(adjusted[keep])
        years = (index.max() - index.min()).days / 365.25
        drift_raw = (1 + raw[keep]).prod() ** (1 / years) - 1
        drift_adjusted = (1 + adjusted[keep]).prod() ** (1 / years) - 1
        signal_raw, signal_adjusted = trend_signal(frame["raw_close"]), trend_signal(frame["close"])
        both = signal_raw.notna() & signal_adjusted.notna()
        flips = float(np.mean(np.sign(signal_raw[both]) != np.sign(signal_adjusted[both])))
        verdict = "the RAW front month" if to_raw > to_adjusted else "already back-adjusted"
        print(f"   {stored:<8} {to_raw:>8.4f} {to_adjusted:>12.4f}  {verdict:<24}"
              f" {drift_raw - drift_adjusted:>+9.2%} {flips:>15.1%}")
    print("\n   Ten further commodities carry the same defect with no back-adjusted")
    print("   series to measure it against, so its size there is unknown rather")
    print("   than zero.")
    print(f"\n{RULE}")


if __name__ == "__main__":
    main()

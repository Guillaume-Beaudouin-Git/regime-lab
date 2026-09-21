"""M1 — apply the calendar realignment, and prove it worked.

Produces the two panels the rest of the study consumes, and checks the repair
against the same external instruments that diagnosed the defect. A repair that is
not verified against the measurement that motivated it is a guess.

Writes to `data/cache/`, which is gitignored and regenerable. The stored
`trend_universe.parquet` is never modified.

    trend_universe_m1.parquet           headline: the eight late series shifted
    trend_universe_m1_verified.parquet  sensitivity: only the three confirmed
                                        against a future are shifted

No book return is computed here. That is A3 and A4.

Usage:
    .venv/bin/python scripts/apply_m1_realignment.py
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from regime_lab.config import CACHE
from regime_lab.extensions.repair import (
    FX_ALIGNED,
    FX_LATE_INFERRED,
    FX_LATE_VERIFIED,
    realign_fx,
    realignment_report,
)

RULE = "=" * 78

#: Stored column -> the CME future that diagnosed it. Only these three exist locally.
CONFIRMED_AGAINST = {"EURUSD=X": "6E", "GBPUSD=X": "6B", "AUDUSD=X": "6A"}

#: Families measured as already aligned, carried here as controls.
CONTROLS = {"^GSPC": "ES", "CL=F": "CL", "IEF": "ZN"}


def futures_dir() -> Path | None:
    configured = os.environ.get("SIMULATOR_PATH")
    if not configured:
        return None
    path = Path(configured).expanduser() / "strategies" / "trend_futures" / "daily"
    return path if path.exists() else None


def same_session_correlation(series: pd.Series, reference: pd.Series) -> float:
    index = series.index.intersection(reference.index)
    return float(series.reindex(index).corr(reference.reindex(index)))


def main() -> None:
    stored = pd.read_parquet(CACHE / "trend_universe.parquet")
    headline = realign_fx(stored, scope="all")
    sensitivity = realign_fx(stored, scope="verified")

    print(RULE)
    print("M1  CALENDAR REALIGNMENT OF THE LATE FX SERIES")
    print(RULE)
    print(f"\n   panel  {stored.shape[1]} instruments, "
          f"{stored.index.min():%Y-%m-%d} to {stored.index.max():%Y-%m-%d}\n")
    print(realignment_report(stored).to_string())

    print(f"\n{RULE}\n1.  DOES THE REPAIR ACTUALLY REPAIR?\n")
    daily = futures_dir()
    if daily is None:
        print("   SIMULATOR_PATH is unset, so the CME futures are unavailable and")
        print("   this check is SKIPPED. The panels below are still written, but the")
        print("   repair is then asserted rather than verified.")
    else:
        print(f"   {'instrument':<11} {'stored':>9} {'repaired':>10}   reading")
        for column, future in CONFIRMED_AGAINST.items():
            reference = pd.read_parquet(daily / f"{future}.parquet")["close"].pct_change()
            before = same_session_correlation(stored[column].pct_change(), reference)
            after = same_session_correlation(headline[column].pct_change(), reference)
            verdict = "repaired" if after > 0.5 and after > before else "NOT REPAIRED"
            print(f"   {column:<11} {before:>9.4f} {after:>10.4f}   {verdict}")

        print("\n   controls, which M1 must not touch:\n")
        print(f"   {'instrument':<11} {'stored':>9} {'repaired':>10}   reading")
        for column, future in CONTROLS.items():
            reference = pd.read_parquet(daily / f"{future}.parquet")["close"].pct_change()
            before = same_session_correlation(stored[column].pct_change(), reference)
            after = same_session_correlation(headline[column].pct_change(), reference)
            verdict = "unchanged" if abs(after - before) < 1e-12 else "CHANGED — BUG"
            print(f"   {column:<11} {before:>9.4f} {after:>10.4f}   {verdict}")

    print(f"\n{RULE}\n2.  WHAT THE REPAIR COSTS\n")
    untouched = [c for c in stored.columns if c not in (*FX_LATE_VERIFIED, *FX_LATE_INFERRED)]
    changed = [c for c in stored.columns if not stored[c].equals(headline[c])]
    print(f"   columns changed          {len(changed)} of {stored.shape[1]}")
    print(f"   columns left alone       {len(untouched)}, including {', '.join(FX_ALIGNED)}")
    lost = int(headline[list(changed)].iloc[-1].isna().sum()) if changed else 0
    print(f"   observations lost        {lost}, the last session of each shifted column")
    print(f"   sensitivity shifts       {len(FX_LATE_VERIFIED)} columns instead of {len(changed)}")

    anchor = pd.Timestamp("2016-06-24")
    if anchor in stored.index and "GBPUSD=X" in stored.columns:
        before = stored["GBPUSD=X"].pct_change().loc[anchor]
        after = headline["GBPUSD=X"].pct_change().loc[anchor]
        print(f"\n   anchor, GBP on {anchor:%Y-%m-%d}: stored {before:+.2%}, "
              f"repaired {after:+.2%}")

    for name, panel in (("trend_universe_m1", headline),
                        ("trend_universe_m1_verified", sensitivity)):
        panel.to_parquet(CACHE / f"{name}.parquet")
        print(f"\n   written  data/cache/{name}.parquet")
    print(f"\n{RULE}")


if __name__ == "__main__":
    main()

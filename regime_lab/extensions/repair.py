"""M1 — put each FX return on the session in which it happened.

`docs/PRESPEC_TREND_VEHICLE.md` §4 locks M1 as a calendar correction, not a
fitted parameter. `scripts/audit_fx_alignment.py` and
`scripts/audit_vehicle_coverage.py` establish what it applies to.

**Direction.** The `=X` spot endpoint stamps a session's move on the *following*
session: the universe's price at *t* is the true price at *t-1*. Against the
matching CME futures, same-session correlation is 0.057 to 0.107 while
one-session-lagged correlation is 0.877 to 0.895. The repair therefore moves each
affected column one session earlier, `shift(-1)`, so the level recorded on the
session after a move returns to the session of the move.

**This is not look-ahead.** The information existed on the earlier session — the
pound really did fall on 24 June 2016 — and the vendor recorded it late. The
repair restores the true observation date rather than revealing anything that was
unknown at the time. It does, however, hand the book information one session
fresher than it had before, which is the direction that flatters a backtest. That
is why M1 carries a mandatory sensitivity, `scope="verified"` below.

**Scope, bounded by measurement rather than by assumption.** Eight of the ten spot
series are late. CAD and MXN are not, and shifting them would push them into the
very defect M1 exists to remove: MXN records its 8.30% move on the correct session
of the 2016 US election at 5.8 sigma, and CAD reads aligned on the lead-lag test.
Equity index, commodity and fixed-income series are aligned too — the lag is a
property of the `=X` endpoint and of nothing else.
"""

from __future__ import annotations

import pandas as pd

#: Late by one session and verified against a matching CME future.
FX_LATE_VERIFIED = ("EURUSD=X", "GBPUSD=X", "AUDUSD=X")

#: Late by one session on the lead-lag test against the ICE dollar index only.
#: NOK and SEK clear that test by ratios of 1.04 and 1.07, which is thin.
FX_LATE_INFERRED = ("CHF=X", "JPY=X", "NOK=X", "NZDUSD=X", "SEK=X")

#: Measured as already aligned. Shifting these would *create* the M1 defect.
FX_ALIGNED = ("CAD=X", "MXN=X")

FX_LATE = FX_LATE_VERIFIED + FX_LATE_INFERRED


def realign_fx(prices: pd.DataFrame, *, scope: str = "all") -> pd.DataFrame:
    """Return a copy of ``prices`` with the late FX columns moved one session earlier.

    Parameters
    ----------
    prices:
        The stored price panel. Not modified.
    scope:
        ``"all"`` shifts the eight series measured as late — the headline.
        ``"verified"`` shifts only the three confirmed against a future and leaves
        the five that rest on the lead-lag test alone. This is the sensitivity M1
        requires, and it exists because treating the five as settled would repeat
        the inference error the FX audit was written to correct.
        ``"none"`` returns the panel untouched, for a like-for-like baseline.

    The last session of every shifted column becomes NaN: the true level for that
    session would have been stamped on the following one, which does not exist yet.
    """
    if scope not in {"all", "verified", "none"}:
        raise ValueError(f"scope must be 'all', 'verified' or 'none', not {scope!r}")

    repaired = prices.copy()
    if scope == "none":
        return repaired

    columns = FX_LATE if scope == "all" else FX_LATE_VERIFIED
    for column in columns:
        if column in repaired.columns:
            repaired[column] = repaired[column].shift(-1)
    return repaired


def realignment_report(prices: pd.DataFrame) -> pd.DataFrame:
    """One row per FX column: what M1 does to it, and on what evidence."""
    rows = []
    for column in FX_LATE_VERIFIED + FX_LATE_INFERRED + FX_ALIGNED:
        if column not in prices.columns:
            continue
        if column in FX_LATE_VERIFIED:
            action, evidence = "shifted", "CME future, same-session vs lagged correlation"
        elif column in FX_LATE_INFERRED:
            action, evidence = "shifted", "lead-lag against the ICE dollar index only"
        else:
            action, evidence = "left alone", "measured aligned; shifting would create the defect"
        rows.append({"instrument": column, "action": action, "evidence": evidence})
    return pd.DataFrame(rows).set_index("instrument")

"""Macro surprises built from first releases, because consensus is not free.

The market trades the gap between what printed and what economists expected.
That gap is sold, not published: Trading Economics discontinued its guest
account, Bloomberg's consensus is a terminal product, and the only free calendar
carrying a forecast field — ForexFactory — serves the current week and nothing
before it.

So the surprise is modelled. For each release, the value **as first published**
minus a forecast built from an autoregression on that series' own prior **first
releases**. Both sides use only what had been published; nothing revised and
nothing later enters either.

This is a proxy and the difference matters. A time-series model does not know
what a hurricane did to payrolls, and economists do. The proxy is therefore
scored against the Philadelphia Fed's Survey of Professional Forecasters, which
is free and carries real consensus, before any return is looked at.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def first_releases(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse a point-in-time frame to one row per period: its first print.

    Returns columns ``period``, ``released_at`` and ``value``, sorted by release
    date, which is the order a real observer saw them in.
    """
    frame = frame.sort_values(["period", "available_at"])
    first = frame.groupby("period", as_index=False).first()
    return (
        first.rename(columns={"available_at": "released_at"})
        .loc[:, ["period", "released_at", "value"]]
        .sort_values("released_at", ignore_index=True)
    )


def transform(values: pd.Series, kind: str) -> pd.Series:
    """Put a level series on a scale where an autoregression is meaningful.

    Levels of an index are not comparable across time — rebasing alone moves
    industrial production by ten percent — so anything that behaves like an
    index becomes a growth rate. Rates and percentages are already stationary
    enough to difference.
    """
    if kind == "growth":
        return values.pct_change()
    if kind == "diff":
        return values.diff()
    if kind == "level":
        return values
    raise ValueError(f"unknown transform {kind!r}")


def model_surprise(
    releases: pd.DataFrame, *, kind: str = "growth", lags: int = 6, min_history: int = 36
) -> pd.DataFrame:
    """Surprise of each release against a forecast that saw only earlier ones.

    The forecast is refitted at every release on the history available then, so
    a surprise in 2003 is measured against a model that knows nothing after
    2003. Slower than fitting once, and the only version that can be called
    point-in-time.

    Returns the release table with ``expected``, ``surprise`` and ``z``, the
    surprise standardised by the expanding dispersion of its own past — which is
    what makes a payroll surprise of forty thousand comparable with a CPI
    surprise of two tenths.
    """
    out = releases.copy()
    series = transform(out["value"], kind)
    n = len(out)
    expected = np.full(n, np.nan)

    for i in range(min_history, n):
        history = series.iloc[:i].dropna()
        if len(history) < min_history:
            continue
        y = history.to_numpy()
        k = min(lags, len(y) // 4)
        if k < 1:
            continue
        # Ordinary autoregression on the history known at this release.
        design = np.column_stack([y[k - j - 1 : len(y) - j - 1] for j in range(k)])
        target = y[k:]
        design = np.column_stack([np.ones(len(target)), design])
        try:
            beta, *_ = np.linalg.lstsq(design, target, rcond=None)
        except np.linalg.LinAlgError:
            continue
        expected[i] = float(beta[0] + beta[1:] @ y[-k:][::-1])

    out["actual"] = series.to_numpy()
    out["expected"] = expected
    out["surprise"] = out["actual"] - out["expected"]
    scale = out["surprise"].expanding(min_periods=24).std().shift(1)
    out["z"] = (out["surprise"] / scale).clip(-4.0, 4.0)
    return out


def decayed_score(
    events: pd.DataFrame, dates: pd.DatetimeIndex, *, half_life: float = 10.0
) -> pd.Series:
    """Aggregate standardised surprises with an exponential kernel.

    ``S(t) = Σ z_k exp(−λ (t − t_k))`` over every release already public at
    ``t``. The kernel is the source model's central mechanism: an announcement
    moves prices at once and keeps being absorbed for days, and older figures
    fade rather than being cut off.

    It is also what makes the model hard to test. Between releases the score
    changes deterministically and carries no new information, so daily
    observations of it are far from independent — which the accompanying
    analysis measures rather than assumes.
    """
    lam = np.log(2.0) / half_life
    clean = events.dropna(subset=["z"])
    times = clean["released_at"].to_numpy().astype("datetime64[D]").astype(np.int64)
    z = clean["z"].to_numpy(dtype=float)

    grid = np.asarray(dates.to_numpy().astype("datetime64[D]").astype(np.int64))
    out = np.zeros(len(grid))
    for t, weight in zip(times, z, strict=True):
        elapsed = grid - t
        live = elapsed >= 0
        out[live] += weight * np.exp(-lam * elapsed[live])
    return pd.Series(out, index=dates, name="score")


def raw_score(events: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.Series:
    """The same surprises with no decay: each lands on its release day only.

    The comparison the kernel has to win. If a score that forgets everything
    the next morning does as well, the kernel is decoration.
    """
    clean = events.dropna(subset=["z"])
    out = pd.Series(0.0, index=dates, name="raw")
    for stamp, weight in zip(clean["released_at"], clean["z"], strict=True):
        idx = dates[dates >= stamp]
        if len(idx):
            out.loc[idx[0]] += weight
    return out


#: Sign of each indicator as a reading of economic strength, ``+1`` when a
#: higher-than-expected print means a stronger economy.
#:
#: Every sign here follows from the definition of the series and needs no
#: backtest, which is the standard hypothesis 1 failed: its map was arguable and
#: turned out to sit at the 7th percentile of random maps. Unemployment and
#: jobless claims are inverted because they count people out of work.
STRENGTH_SIGN = {
    "payems": +1, "unrate": -1, "indpro": +1, "houst": +1, "permit": +1,
    "dgorder": +1, "retail": +1, "sentiment": +1, "capacity": +1, "hours": +1,
    "claims": -1,
    # Inflation is the one contested pair. A hot print is not economic strength,
    # but it tightens expected policy, which is what moves the currency. Kept at
    # +1 and flagged rather than dropped, so the choice is visible.
    "cpi": +1, "core_cpi": +1,
}

#: The one the pre-specification names as arguable.
CONTESTED = {"cpi", "core_cpi"}

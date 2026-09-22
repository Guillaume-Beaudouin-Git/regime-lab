"""The walk-forward of the Two Sigma plan — `PRESPEC_TWOSIGMA.md` §4.

What the draft fixes: **5 test folds of 4.0 years, expanding training, first training
window 11.6 years**, on the inferential sample **1995-01-04 to 2026-07-31, 7,946
sessions** — where every signal of the library exists at once.

**The arithmetic does not close, by eleven days.** The sample spans 11,531 days =
31.570 years of 365.25 days; 11.6 + 5 x 4.0 = 31.6. One of the three figures has to
give by 0.030 years, about 8 sessions, and the draft does not say which. Two readings,
both kept, both consistent with the draft's rounded text:

``anchor="end"``   (default) the five test folds are exactly 4.0 years each and end
                   on the last session; the first training window takes what is left,
                   **11.57 years** (1995-01-04 to 2006-07-31). Every fold carries the
                   same calendar length and so roughly the same power.
``anchor="start"`` the first training window is exactly 11.6 years; the test folds
                   follow at 4.0 years each and the **last one is cut short** at the
                   end of the sample, at 3.97 years.

The lock must choose. The difference moves each fold boundary by eleven days.

Years are calendar years of 365.25 days, so 4.0 years is exactly 1,461 days. A session
belongs to test fold *k* when its date lies in ``(b[k-1], b[k]]``; the training window
of fold *k* is every session from the sample start through ``b[k-1]``. Sessions before
the first test fold are training-only and carry fold number 0.

A side finding for the library, not for this module: the draft says the binding
warm-up is the 1,260-session long-term reversal. It is not — `IND_LTREV` is complete
from 1994-12-23 (7,952 sessions); the 1995-01-04 start is set by `IND_SEASON`, whose
same-month mean needs five prior years. The sample itself is right.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from regime_lab.config import RAW

INFERENTIAL_START = pd.Timestamp("1995-01-04")
INFERENTIAL_END = pd.Timestamp("2026-07-31")
N_FOLDS = 5
TEST_YEARS = 4.0
FIRST_TRAIN_YEARS = 11.6
DAYS_PER_YEAR = 365.25


@dataclass(frozen=True)
class Fold:
    """One walk-forward fold: an expanding training window and the test window after it."""

    number: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    #: The calendar cut-offs ``b[k-1]`` and ``b[k]``: training runs through the first,
    #: the test window through the second. They need not fall on a session.
    train_cutoff: pd.Timestamp
    test_cutoff: pd.Timestamp

    def train(self, index: pd.DatetimeIndex) -> pd.DatetimeIndex:
        """The dates of ``index`` inside the training window."""
        return index[(index >= self.train_start) & (index <= self.train_end)]

    def test(self, index: pd.DatetimeIndex) -> pd.DatetimeIndex:
        """The dates of ``index`` inside the test window."""
        return index[(index >= self.test_start) & (index <= self.test_end)]

    @property
    def train_years(self) -> float:
        """Calendar length of the training window, first session to its cut-off."""
        return (self.train_cutoff - self.train_start) / pd.Timedelta(days=DAYS_PER_YEAR)

    @property
    def test_years(self) -> float:
        """Calendar length of the test window, cut-off to cut-off.

        Consecutive folds tile the calendar: five folds of 4.0 years sum to 20.0.
        """
        return (self.test_cutoff - self.train_cutoff) / pd.Timedelta(days=DAYS_PER_YEAR)


def inferential_sessions(
    path: Path | None = None,
    *,
    start: pd.Timestamp = INFERENTIAL_START,
    end: pd.Timestamp = INFERENTIAL_END,
) -> pd.DatetimeIndex:
    """The sessions of the 49-industry panel inside the inferential sample.

    Reads the ``period`` column only: no return enters this function.
    """
    periods = pd.read_parquet(path or RAW / "panels" / "industry_49.parquet", columns=["period"])
    sessions = pd.DatetimeIndex(np.sort(periods["period"].unique()))
    return sessions[(sessions >= start) & (sessions <= end)]


def walk_forward_folds(
    sessions: pd.DatetimeIndex,
    *,
    n_folds: int = N_FOLDS,
    test_years: float = TEST_YEARS,
    first_train_years: float = FIRST_TRAIN_YEARS,
    anchor: Literal["end", "start"] = "end",
) -> tuple[Fold, ...]:
    """Expanding-window walk-forward folds on a session calendar.

    ``anchor="end"`` makes every test fold exactly ``test_years`` long and lets the
    first training window absorb the remainder; ``first_train_years`` is then
    ignored. ``anchor="start"`` makes the first training window exactly
    ``first_train_years`` long and truncates the last test fold at the sample end.
    """
    sessions = pd.DatetimeIndex(sessions).sort_values()
    if len(sessions) < 2:
        raise ValueError("need at least two sessions")
    if sessions.has_duplicates:
        raise ValueError("sessions must be unique")
    first, last = sessions[0], sessions[-1]
    fold_days = pd.Timedelta(days=test_years * DAYS_PER_YEAR)

    if anchor == "end":
        bounds = [last - (n_folds - k) * fold_days for k in range(n_folds + 1)]
    elif anchor == "start":
        # 11.6 x 365.25 days is not a whole day: the cut falls at 21:36 on its date,
        # after that session's close, so the session belongs to training. Rounding to
        # the second only removes the float residue (21:35:59.999999936).
        origin = first + pd.Timedelta(days=first_train_years * DAYS_PER_YEAR).round("s")
        bounds = [min(origin + k * fold_days, last) for k in range(n_folds + 1)]
        bounds[-1] = last
    else:
        raise ValueError(f"anchor must be 'end' or 'start', not {anchor!r}")

    if bounds[0] <= first:
        raise ValueError(
            f"the sample ({(last - first).days / DAYS_PER_YEAR:.2f} years) leaves no training "
            f"window before {n_folds} test folds of {test_years} years"
        )

    folds = []
    for k in range(1, n_folds + 1):
        train = sessions[sessions <= bounds[k - 1]]
        test = sessions[(sessions > bounds[k - 1]) & (sessions <= bounds[k])]
        if len(test) == 0:
            raise ValueError(f"test fold {k} holds no session")
        folds.append(Fold(k, train[0], train[-1], test[0], test[-1], bounds[k - 1], bounds[k]))
    return tuple(folds)


def fold_of(sessions: pd.DatetimeIndex, folds: tuple[Fold, ...]) -> pd.Series:
    """The test fold each session belongs to; 0 for training-only sessions."""
    sessions = pd.DatetimeIndex(sessions)
    out = pd.Series(0, index=sessions, name="fold", dtype=np.int64)
    for fold in folds:
        out.loc[fold.test(sessions)] = fold.number
    return out


def describe_folds(folds: tuple[Fold, ...], sessions: pd.DatetimeIndex) -> pd.DataFrame:
    """One row per fold: windows, calendar lengths and session counts."""
    rows = [
        {
            "fold": f.number,
            "train_start": f.train_start.date(),
            "train_end": f.train_end.date(),
            "train_years": f.train_years,
            "train_sessions": len(f.train(sessions)),
            "test_start": f.test_start.date(),
            "test_end": f.test_end.date(),
            "test_years": f.test_years,
            "test_sessions": len(f.test(sessions)),
        }
        for f in folds
    ]
    return pd.DataFrame(rows).set_index("fold")

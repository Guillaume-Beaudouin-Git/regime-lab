"""The evaluation protocol every model family is held to.

Three rules, fixed here rather than in each model, because a comparison is only
as good as what the families share.

**Expanding refits on a common schedule.** Every family is refitted on the same
dates, using only data available then. This matters more than it sounds: the
published claim that jump models are more stable than hidden Markov models was
measured with the jump model refitted twice a year and the Markov model
refitted daily, so part of the advantage is a difference in protocol rather than
in method. Pairing the schedule is what makes the stability test mean anything.

**Online prediction between refits.** States are produced by filtering forward,
never by smoothing over a window that contains the future.

**One feature subset, chosen on training data.** The sparse jump model selects
features on each training window; every family then receives that same subset.
Letting each family pick its own on the full sample would compare selections,
not models.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Fold:
    """One expanding-window evaluation block."""

    index: int
    train_end: pd.Timestamp
    eval_start: pd.Timestamp
    eval_end: pd.Timestamp

    def __str__(self) -> str:
        return (
            f"fold {self.index}: train to {self.train_end:%Y-%m}, "
            f"evaluate {self.eval_start:%Y-%m} to {self.eval_end:%Y-%m}"
        )


def walk_forward(
    dates: pd.DatetimeIndex, *, min_train_years: int = 10, folds: int = 5
) -> list[Fold]:
    """Split ``dates`` into expanding-window folds.

    The first fold trains on ``min_train_years`` and evaluates the block that
    follows; each later fold trains on everything up to its own start. A single
    train/test split is not admissible under the charter.
    """
    if folds < 5:
        raise ValueError("the charter requires at least five folds")

    start = dates.min()
    first_eval = start + pd.DateOffset(years=min_train_years)
    tail = dates[dates >= first_eval]
    if len(tail) < folds * 252:
        raise ValueError("not enough evaluation data for the requested folds")

    edges = pd.date_range(tail.min(), dates.max(), periods=folds + 1)
    return [
        Fold(
            index=i,
            train_end=edges[i],
            eval_start=edges[i],
            eval_end=edges[i + 1],
        )
        for i in range(folds)
    ]


def refit_dates(
    dates: pd.DatetimeIndex, *, first: pd.Timestamp, months: int = 6
) -> pd.DatetimeIndex:
    """Dates on which every family refits, from ``first`` to the end of sample."""
    schedule = pd.date_range(first, dates.max(), freq=f"{months}MS")
    return pd.DatetimeIndex([dates[dates <= d].max() for d in schedule]).dropna().unique()

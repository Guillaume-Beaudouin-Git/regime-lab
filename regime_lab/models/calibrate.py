"""Choosing the jump penalty, on training data and nowhere else.

The charter is explicit that lambda is an *optimised* parameter, not a free
structural constant: the published implementation picks it by maximising a
validation Sharpe. Pretending otherwise would understate the study's degrees of
freedom, so the grid is declared here, the search runs inside each training
window, and every candidate is written to the trials log.

The objective is the Sharpe of the base book gated by the resulting states,
measured on the training window. It is not the clustering loss: a partition can
fit the feature space beautifully and be useless for the decision the model
exists to make.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

#: Declared before any fit. Widened only by an amendment to the protocol.
LAMBDA_GRID = (1.0, 3.0, 10.0, 30.0, 100.0, 300.0)

#: A partition that never switches, or switches constantly, is not a regime
#: model. Candidates outside this band are rejected before scoring.
MIN_SWITCHES_PER_YEAR = 0.5
MAX_SWITCHES_PER_YEAR = 12.0


@dataclass(frozen=True)
class Calibration:
    """The chosen penalty and what it was chosen against."""

    jump_penalty: float
    train_sharpe: float
    switches_per_year: int
    candidates: list[tuple[float, float, float]]


def choose_jump_penalty(
    factory,
    features: pd.DataFrame,
    returns: pd.Series,
    *,
    grid: tuple[float, ...] = LAMBDA_GRID,
) -> Calibration:
    """Pick the penalty maximising the in-training Sharpe of the gated book."""
    years = max(len(features) / 252.0, 1.0)
    scored: list[tuple[float, float, float]] = []

    for penalty in grid:
        model = factory(penalty)
        try:
            model.fit(features, returns)
            states = model.predict_online(features)
        except (ValueError, np.linalg.LinAlgError) as exc:
            # A penalty that collapses the partition is a legitimate rejection;
            # anything else is a bug, and swallowing it would show up later as
            # an unexplained gap in the grid rather than as a failure.
            scored.append((penalty, np.nan, np.nan))
            print(f"      lambda {penalty}: rejected ({type(exc).__name__}: {exc})")
            continue

        clean = states.dropna()
        if clean.empty:
            continue
        rate = float((clean.diff().abs() > 0).sum()) / years
        if not MIN_SWITCHES_PER_YEAR <= rate <= MAX_SWITCHES_PER_YEAR:
            scored.append((penalty, np.nan, rate))
            continue

        position = (clean == clean.max()).astype(float).shift(1).fillna(0.0)
        gated = (returns.reindex(clean.index) * position).dropna()
        sharpe = (
            float(gated.mean() / gated.std(ddof=1) * np.sqrt(252))
            if gated.std(ddof=1) > 0
            else np.nan
        )
        scored.append((penalty, sharpe, rate))

    valid = [row for row in scored if np.isfinite(row[1])]
    if not valid:
        # Nothing in the grid produced a usable partition; fall back to the
        # middle of the grid rather than silently returning a degenerate model.
        return Calibration(float(np.median(grid)), float("nan"), 0, scored)

    best = max(valid, key=lambda row: row[1])
    return Calibration(best[0], best[1], int(round(best[2])), scored)

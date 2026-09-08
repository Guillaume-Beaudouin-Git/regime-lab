"""Shared machinery: expanding refits, online prediction, one state series."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd

from regime_lab.models.protocol import refit_dates


class RegimeModel(Protocol):
    """What a family must provide to be run under the common protocol."""

    def fit(self, features: pd.DataFrame, returns: pd.Series) -> None:
        """Estimate on a training window."""

    def predict_online(self, features: pd.DataFrame) -> pd.Series:
        """Filter forward over ``features``, using no information after each row."""


def run_expanding(
    model_factory,
    features: pd.DataFrame,
    returns: pd.Series,
    *,
    first_refit: pd.Timestamp,
    months: int = 6,
) -> tuple[pd.Series, dict[pd.Timestamp, object], pd.DataFrame]:
    """Refit on a schedule, predict online in between, return one state series.

    At each refit the model sees data strictly before that date. Between refits
    the fitted model filters forward over new observations only. The returned
    series is therefore out of sample everywhere after ``first_refit``.
    """
    schedule = refit_dates(features.index, first=first_refit, months=months)
    states = pd.Series(index=features.index, dtype="float64")
    offline = pd.Series(index=features.index, dtype="float64")
    fitted: dict[pd.Timestamp, object] = {}
    diagnostics: list[dict[str, object]] = []

    for i, refit in enumerate(schedule):
        train = features.loc[features.index < refit]
        if len(train) < 504:
            continue

        # The factory receives the training window so a family may calibrate
        # its own hyper-parameters there, never on the evaluation sample.
        model = model_factory(train, returns.loc[train.index], i)
        model.fit(train, returns.loc[train.index])
        fitted[refit] = model

        # In-training behaviour of the overlay, needed to set the volatility
        # target of the benchmark on training data alone. Computed here because
        # this is the only point where the model and its training window meet.
        in_sample = model.predict_online(train)
        held = (in_sample == in_sample.max()).astype(float).shift(1).fillna(0.0)
        gated = (returns.reindex(train.index) * held).dropna()
        diagnostics.append(
            {
                "refit": refit,
                "train_days": len(train),
                "train_exposure": float(held.mean()),
                "train_overlay_vol": float(gated.std(ddof=1) * np.sqrt(252)),
            }
        )

        stop = schedule[i + 1] if i + 1 < len(schedule) else features.index.max()
        block = features.loc[(features.index >= refit) & (features.index <= stop)]
        if block.empty:
            continue

        # Filtering needs history to condition on, so the model sees the training
        # window followed by the block, and only the block is kept.
        context = pd.concat([train.tail(252), block])
        predicted = model.predict_online(context)
        states.loc[block.index] = np.asarray(predicted)[-len(block) :]

        # The same block assigned with hindsight, used only to price latency.
        seen_whole = model.predict_offline(context)
        offline.loc[block.index] = np.asarray(seen_whole)[-len(block) :]

    return (
        states.rename("state"),
        fitted,
        pd.DataFrame(diagnostics).set_index("refit"),
        offline.rename("state_offline"),
    )


def label_by_training_returns(
    states: pd.Series, returns: pd.Series, train_index: pd.Index
) -> dict[int, float]:
    """Rank states by their mean return **on training data only**.

    Deciding which state is the good one from the evaluation sample would select
    on the very quantity the study tests. This is the guard against that.
    """
    frame = pd.concat([states, returns], axis=1).loc[train_index].dropna()
    frame.columns = ["state", "ret"]
    return frame.groupby("state")["ret"].mean().to_dict()

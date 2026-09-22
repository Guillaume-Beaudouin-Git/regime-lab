"""The Two Sigma walk-forward: five test folds that tile the calendar, expanding training.

The draft's arithmetic does not close by eleven days (11.6 + 5 x 4.0 = 31.6 against a
31.57-year sample), so the module keeps two anchors. Both must tile: no session in
two test folds, none skipped after the first cut-off, and every training window ending
exactly where its test window begins.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regime_lab.config import RAW
from regime_lab.selection.folds import (
    DAYS_PER_YEAR,
    INFERENTIAL_END,
    INFERENTIAL_START,
    describe_folds,
    fold_of,
    inferential_sessions,
    walk_forward_folds,
)


def _sessions(seed: int = 0) -> pd.DatetimeIndex:
    """Weekdays of the inferential sample, less a few random holidays."""
    days = pd.bdate_range(INFERENTIAL_START, INFERENTIAL_END)
    rng = np.random.default_rng(seed)
    holidays = rng.choice(np.arange(1, len(days) - 1), 250, replace=False)
    return days.delete(holidays)


@pytest.mark.parametrize("anchor", ["end", "start"])
def test_the_test_folds_tile_the_sessions_after_the_first_cutoff(anchor):
    sessions = _sessions()
    folds = walk_forward_folds(sessions, anchor=anchor)
    assert [f.number for f in folds] == [1, 2, 3, 4, 5]
    tested = np.concatenate([f.test(sessions) for f in folds])
    assert len(tested) == len(np.unique(tested))
    after = sessions[sessions > folds[0].train_cutoff]
    np.testing.assert_array_equal(np.sort(tested), after.to_numpy())
    for previous, fold in zip(folds[:-1], folds[1:], strict=True):
        assert fold.train_cutoff == previous.test_cutoff
        assert fold.train_end == previous.test_end


@pytest.mark.parametrize("anchor", ["end", "start"])
def test_training_expands_from_the_first_session_and_stops_before_the_test(anchor):
    sessions = _sessions()
    for fold in walk_forward_folds(sessions, anchor=anchor):
        train, test = fold.train(sessions), fold.test(sessions)
        assert fold.train_start == sessions[0] == train[0]
        assert train[-1] == fold.train_end <= fold.train_cutoff < fold.test_start
        assert test[0] == fold.test_start and test[-1] == fold.test_end <= fold.test_cutoff
        assert len(train) + len(test) == int((sessions <= fold.test_cutoff).sum())


def test_anchor_end_gives_five_equal_folds_ending_on_the_last_session():
    sessions = _sessions()
    folds = walk_forward_folds(sessions, anchor="end")
    assert all(f.test_years == pytest.approx(4.0, abs=1e-12) for f in folds)
    assert folds[-1].test_end == sessions[-1]
    assert sum(f.test_years for f in folds) == pytest.approx(20.0)
    span = (sessions[-1] - sessions[0]).days / DAYS_PER_YEAR
    assert folds[0].train_years == pytest.approx(span - 20.0)
    assert folds[0].train_cutoff == pd.Timestamp("2006-07-31")


def test_anchor_start_gives_the_first_window_exactly_and_truncates_the_last_fold():
    sessions = _sessions()
    folds = walk_forward_folds(sessions, anchor="start")
    assert folds[0].train_years == pytest.approx(11.6, abs=1e-6)
    assert all(f.test_years == pytest.approx(4.0, abs=1e-6) for f in folds[:-1])
    assert folds[-1].test_years < 4.0
    assert folds[-1].test_cutoff == sessions[-1]
    assert folds[0].train_cutoff == folds[0].train_cutoff.round("s")


def test_fold_of_numbers_test_sessions_and_zeroes_training_only_ones():
    sessions = _sessions()
    folds = walk_forward_folds(sessions)
    fold = fold_of(sessions, folds)
    assert (fold[sessions <= folds[0].train_cutoff] == 0).all()
    for f in folds:
        assert (fold.loc[f.test(sessions)] == f.number).all()
    assert set(fold.unique()) == {0, 1, 2, 3, 4, 5}


def test_describe_folds_counts_sessions():
    sessions = _sessions()
    folds = walk_forward_folds(sessions)
    table = describe_folds(folds, sessions)
    assert list(table.index) == [1, 2, 3, 4, 5]
    assert table["test_sessions"].sum() == int((sessions > folds[0].train_cutoff).sum())
    assert (table["train_sessions"].diff().dropna() > 0).all()


def test_impossible_or_ambiguous_requests_are_refused():
    short = pd.bdate_range("2000-01-03", periods=2_000)
    with pytest.raises(ValueError, match="leaves no training"):
        walk_forward_folds(short)
    with pytest.raises(ValueError, match="anchor"):
        walk_forward_folds(_sessions(), anchor="middle")
    doubled = _sessions().append(_sessions()[:3])
    with pytest.raises(ValueError, match="unique"):
        walk_forward_folds(doubled)


@pytest.mark.skipif(
    not (RAW / "panels" / "industry_49.parquet").exists(), reason="industry panel absent"
)
def test_the_declared_folds_on_the_industry_calendar():
    sessions = inferential_sessions()
    assert len(sessions) == 7_946
    assert sessions[0] == INFERENTIAL_START and sessions[-1] == INFERENTIAL_END
    table = describe_folds(walk_forward_folds(sessions, anchor="end"), sessions)
    assert table["test_sessions"].tolist() == [1_007, 1_007, 1_007, 1_006, 1_004]
    assert table.loc[1, "train_sessions"] == 2_915
    assert table.loc[1, "train_years"] == pytest.approx(11.570, abs=5e-4)

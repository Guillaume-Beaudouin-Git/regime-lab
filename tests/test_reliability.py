"""Layer-1 metrics must refuse inputs on which they are not defined.

`external_validation` scored two degenerate cases instead of declining them. A
state that never moves satisfies `state == state.min()` on every session, so the
classifier was graded as if it had called a recession every day; and a reference
with one class has no balanced accuracy, only the recall of the class present.
Both returned a float, and the float was read as a verdict: three per-fold cells
read 0.0% balanced accuracy and three others read 0.93 to 0.98 at kappa exactly
0.000, all six from the same defect with opposite signs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from regime_lab.evaluation.reliability import external_validation

INDEX = pd.date_range("2010-01-01", periods=500, freq="B")


def _reference(positives: int) -> pd.Series:
    ref = pd.Series(0.0, index=INDEX)
    ref.iloc[:positives] = 1.0
    return ref


def test_constant_state_is_not_scored_as_a_permanent_alarm():
    out = external_validation(pd.Series(1.0, index=INDEX), _reference(40))
    assert np.isnan(out["balanced_accuracy"])
    assert np.isnan(out["kappa"])


def test_single_class_reference_has_no_balanced_accuracy():
    state = pd.Series(np.resize([0.0, 1.0], len(INDEX)), index=INDEX)
    out = external_validation(state, _reference(0))
    assert np.isnan(out["balanced_accuracy"])
    assert np.isnan(out["kappa"])


def test_a_state_that_moves_against_a_reference_that_moves_is_still_scored():
    state = pd.Series(np.resize([0.0, 1.0], len(INDEX)), index=INDEX)
    out = external_validation(state, _reference(40))
    assert 0.0 < out["balanced_accuracy"] <= 1.0
    assert out["n"] == len(INDEX)


def test_every_branch_returns_the_same_keys():
    short = external_validation(pd.Series(1.0, index=INDEX[:50]), _reference(40).iloc[:50])
    state = pd.Series(np.resize([0.0, 1.0], len(INDEX)), index=INDEX)
    full = external_validation(state, _reference(40))
    assert set(short) == set(full)

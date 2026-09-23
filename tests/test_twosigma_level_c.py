"""Level C of the Two Sigma tree must be the lock's §12.10, and nothing else.

Every test runs on synthetic data (`tree.synthetic_tree_data`, or a planted target and a
planted partition built here); nothing reads `data/`. The properties tested are the ones
the level-C reading would silently depend on: a cadence book follows the rebalancing rule
to the session, the fast engine is the literal book, the pooled-budget rule is the lock's
λ procedure step for step, the training check reverts to the twin, the arm reads each
fold's path lagged one session and carries its counter across folds, nothing after a date
changes anything before it, a planted cadence effect is found and an unrelated partition
is not, the null does not depend on the worker count, the instrument never builds the
real arm, a NaN never reads as a verdict, and the reading refuses to run unless its
thresholds are committed and recomputed bitwise.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from regime_lab.config import ROOT
from regime_lab.selection import level_c as C
from regime_lab.selection import protocol
from regime_lab.selection import tree as T

N_SMALL = 20


# --------------------------------------------------------------------------- fixtures


def _markov(n: int, k: int, stay: float, rng: np.random.Generator) -> np.ndarray:
    labels = np.zeros(n, dtype=np.int64)
    for t in range(1, n):
        stays = rng.random() < stay
        labels[t] = labels[t - 1] if stays else (labels[t - 1] + rng.integers(1, k)) % k
    return labels


def _paths(labels: np.ndarray, data: T.TreeData) -> pd.Series:
    """One label series as every fold's stamped path, in ``session_paths`` format."""
    series = pd.Series(labels, index=data.sessions)
    pieces = []
    for fold in data.folds:
        for segment, dates in (("train", fold.train(data.sessions)),
                               ("test", fold.test(data.sessions))):
            index = pd.MultiIndex.from_arrays(
                [np.full(len(dates), fold.number), np.full(len(dates), segment), dates],
                names=["fold", "segment", "session"])
            pieces.append(pd.Series(series.loc[dates].to_numpy(), index=index))
    return pd.concat(pieces).rename("state").astype(np.int64)


def _planted_target(labels: np.ndarray, data: T.TreeData, rng: np.random.Generator,
                    *, oscillation: float = 0.02, trend: float = 0.01) -> pd.DataFrame:
    """A target whose dynamics depend on the state: states 0-1 oscillate around a fixed
    base (a long interval saves turnover at almost no tracking cost), states 2-3 trend
    (tracking grows with the interval). Undefined over a 63-session warm-up, as w* is."""
    n, m = len(labels), data.excess.shape[1]
    base = rng.normal(0.0, 0.1, m)
    out = np.empty((n, m))
    for t in range(n):
        if labels[t] >= 2:
            base = base + rng.normal(0.0, trend, m)
            out[t] = base
        else:
            out[t] = base + rng.normal(0.0, oscillation, m)
    out[:63] = np.nan
    return pd.DataFrame(out, index=data.sessions, columns=data.excess.columns)


@pytest.fixture(scope="module")
def data() -> T.TreeData:
    return T.synthetic_tree_data(seed=0, n_industries=49)


@pytest.fixture(scope="module")
def primary(data: T.TreeData) -> T.Partition:
    return T.partition_paths(data, T.PRIMARY)


@pytest.fixture(scope="module")
def planted(data: T.TreeData):
    rng = np.random.default_rng(2)
    labels = _markov(len(data.sessions), 4, 0.97, rng)
    target = _planted_target(labels, data, rng)
    unrelated = _markov(len(data.sessions), 4, 0.97, np.random.default_rng(102))
    return {
        "target": target,
        "setup": C._setup(data, target),
        "paths": _paths(labels, data),
        "unrelated": _paths(unrelated, data),
    }


def _arm(space: C.CadenceSpace, paths: pd.Series, *, detail: bool = True) -> C.ArmReading:
    layout = C.path_layout(paths, space)
    return C.conditional_arm(space, layout, C.state_codes(paths),
                             C.train_states(T.cells(paths), space), detail=detail)


# ---------------------------------------------------------------- the literal book


def test_a_cadence_book_follows_the_rule_session_by_session():
    dates = pd.bdate_range("2020-01-01", periods=12)
    w = pd.DataFrame({"a": 0.1 * np.arange(12), "b": -0.05 * np.arange(12)}, index=dates)
    w.iloc[:2] = np.nan
    h = np.array([9, 9, 7, 3, 3, 3, 21, 21, 21, 2, 1, 5])
    excess = pd.DataFrame({"a": np.linspace(0.01, -0.01, 12), "b": 0.002}, index=dates)
    book = C.cadence_book(w, h, excess=excess)

    # start on the first defined session with counter 0; rebalance when count >= h_t;
    # the 2 after three sessions at 21 rebalances at once (count 4 >= 2)
    assert book.start == dates[2]
    assert list(np.flatnonzero(book.rebalance.to_numpy())) == [2, 5, 9, 10]
    source = [None, None, 2, 2, 2, 5, 5, 5, 5, 9, 10, 10]
    for t, s in enumerate(source):
        expected = np.full(2, np.nan) if s is None else w.iloc[s].to_numpy()
        np.testing.assert_array_equal(book.held.iloc[t].to_numpy(), expected)

    def gap(a: int, b: int) -> float:  # Σ_n |w*_a − w*_b|
        return 0.15 * abs(a - b)

    turnover = [np.nan, np.nan, np.nan, 0, 0, gap(5, 2), 0, 0, 0, gap(9, 5), gap(10, 9), 0]
    np.testing.assert_allclose(book.turnover.to_numpy(), turnover, rtol=0, atol=1e-15)
    delta = [np.nan, np.nan, 0, gap(2, 3), gap(2, 4), 0, gap(5, 6), gap(5, 7), gap(5, 8), 0, 0,
             gap(10, 11)]
    np.testing.assert_allclose(book.delta.to_numpy(), delta, rtol=0, atol=1e-15)

    held = book.held.to_numpy()
    gross = (held * excess.to_numpy()).sum(axis=1)
    charged = np.nan_to_num(np.array(turnover, dtype=float)) * 5.0 / 10_000.0
    expected_net = gross - charged
    expected_net[:2] = np.nan
    np.testing.assert_allclose(book.net.to_numpy(), expected_net, rtol=0, atol=1e-15)


def test_the_book_refuses_an_interval_below_one_and_holds_an_undefined_target_as_nan():
    dates = pd.bdate_range("2020-01-01", periods=6)
    w = pd.DataFrame({"a": np.arange(6.0)}, index=dates)
    with pytest.raises(ValueError, match="integer >= 1"):
        C.cadence_book(w, np.array([1, 1, 0, 1, 1, 1]))
    with pytest.raises(ValueError, match="integer >= 1"):
        C.cadence_book(w, np.array([1, 1, 1.5, 1, 1, 1]))
    w.iloc[3] = np.nan
    book = C.cadence_book(w, 1)
    assert np.isnan(book.held.iloc[3, 0]) and np.isnan(book.turnover.iloc[3])
    assert np.isnan(book.turnover.iloc[4]) and book.held.iloc[4, 0] == 4.0


# ---------------------------------------------------------------- the fast engine


def test_the_state_blind_menu_is_the_literal_book_and_h1_is_the_control(data):
    setup = C._setup(data)
    space, target = setup.space, C.target_weights(data)
    for i, h in enumerate(C.H_MENU):
        literal = C.cadence_book(target, h)
        np.testing.assert_array_equal(space.held(space.menu_source[i]).to_numpy(),
                                      literal.held.to_numpy())
        np.testing.assert_allclose(space.menu_tau[i], literal.turnover.to_numpy(),
                                   rtol=0, atol=1e-13)
        np.testing.assert_allclose(space.menu_delta[i], literal.delta.to_numpy(),
                                   rtol=0, atol=1e-13)
        held_test = literal.held.loc[data.test_sessions]
        assert space.turnover(space.menu_tau[i]) == pytest.approx(
            protocol.annual_turnover(held_test), rel=1e-12)
    control = setup.control.book.weights
    np.testing.assert_array_equal(space.held(space.menu_source[0]).to_numpy(),
                                  control.to_numpy())
    assert np.nanmax(space.menu_delta[0]) == 0.0
    assert space.twin_turnover == pytest.approx(setup.twin.turnover, rel=1e-12)
    assert space.start == int(np.argmax(np.isfinite(control.to_numpy()).all(axis=1)))


def test_the_engine_schedule_is_the_literal_rule_on_any_intervals(data):
    target = C.target_weights(data)
    space = C.cadence_space(target, data.sessions, data.folds)
    rng = np.random.default_rng(7)
    n = len(data.sessions)
    for case in range(4):
        h = rng.choice(C.H_MENU, size=n).astype(np.int64)
        if case == 1:  # a long constant prefix, then variation
            h[: n // 2] = 5
        if case == 2:  # a prefix that changes on the first session after the start
            h[: space.start + 1] = 13
        literal = C.cadence_book(target, h)
        source = C._schedule(h, space.start)
        np.testing.assert_array_equal(space.held(source).to_numpy(), literal.held.to_numpy())
        np.testing.assert_allclose(C._book_tau(space.dist, source, space.start),
                                   literal.turnover.to_numpy(), rtol=0, atol=1e-13)
        np.testing.assert_allclose(C._book_delta(space.dist, source, space.start),
                                   literal.delta.to_numpy(), rtol=0, atol=1e-13)


# ---------------------------------------------------------------- the λ procedure


def _reference_intervals(pi, cost, track, menu=C.H_MENU, twin=C.TWIN_H, steps=100):
    """§12.10's λ procedure transcribed literally, loop by loop."""
    k = len(pi)

    def h_of(lam):
        chosen = []
        for s in range(k):
            best = None
            for i in range(len(menu)):
                value = cost[s][i] + lam * track[s][i]
                if best is None or value <= best[0]:  # ties to the larger h
                    best = (value, i)
            chosen.append(best[1])
        return chosen

    def spent(idx):
        return float(np.sum(np.asarray(pi) * np.array([track[s][i] for s, i in enumerate(idx)])))

    budget = spent([menu.index(twin)] * k)
    if spent(h_of(0.0)) <= budget:
        return [menu[i] for i in h_of(0.0)], 0.0
    lo, hi = 0.0, 1.0
    while spent(h_of(hi)) > budget:
        lo, hi = hi, 2 * hi
    for _ in range(steps):
        mid = (lo + hi) / 2
        if spent(h_of(mid)) <= budget:
            hi = mid
        else:
            lo = mid
    return [menu[i] for i in h_of(hi)], hi


def test_lambda_is_zero_when_zero_meets_the_budget_and_ties_go_to_the_larger_h():
    cost = [[10, 8, 6, 5, 4, 3, 2]] * 2
    flat = [[0, 1, 1, 1, 1, 1, 1]] * 2
    out = C.pooled_budget_intervals([0.5, 0.5], cost, flat)
    assert out.h == (21, 21) and out.lam == 0.0 and out.spend == out.budget == 1.0
    tied = C.pooled_budget_intervals([1.0], [[10, 8, 6, 5, 4, 3, 3]], [[0, 1, 1, 1, 1, 1, 1]])
    assert tied.h == (21,)
    assert C.pooled_budget_intervals([], np.empty((0, 7)), np.empty((0, 7))).h == ()
    assert C.pooled_budget_intervals([1.0], [[np.nan] * 7], [[0.0] * 7]) is None


def test_the_lambda_procedure_is_the_lock_s_step_for_step():
    rng = np.random.default_rng(3)
    menu = np.array(C.H_MENU, dtype=float)
    for _ in range(60):
        k = int(rng.integers(1, 6))
        scale = rng.uniform(0.2, 3.0, (k, 1))
        shape = rng.uniform(0.2, 1.5, (k, 1))
        cost = scale * menu[None, :] ** -shape + rng.uniform(0, 0.05, (k, 7))
        track = rng.uniform(0.1, 2.0, (k, 1)) * (menu[None, :] - 1) ** rng.uniform(0.3, 1.2, (k, 1))
        if rng.random() < 0.3:  # a large turnover scale needs λ beyond the first doubling
            cost = cost * 1e4
        pi = rng.dirichlet(np.ones(k)) * rng.uniform(0.5, 1.0)
        out = C.pooled_budget_intervals(pi, cost, track)
        h, lam = _reference_intervals(pi, cost.tolist(), track.tolist())
        assert list(out.h) == h and out.lam == lam
        assert out.spend <= out.budget
        if lam > 0:  # λ* is the smallest λ meeting the budget, to bisection precision
            assert C.pooled_budget_intervals(pi, cost, track, steps=0).lam >= lam
            lower = 6 - np.argmin((cost + lam * (1 - 1e-9) * track)[:, ::-1], axis=1)
            assert np.sum(pi * track[np.arange(k), lower]) > out.budget


def test_the_pooled_budget_on_a_hand_computed_case():
    # Both states: T(h) = 10, 6, 4, 3, 2.5, 2.2, 2.0 over H. δ(h) = i in state 0 and
    # 0.45 i in state 1 (i the menu index). The argmin of T + λδ leaves h for the next
    # shorter one at λ = ΔT/Δδ: state 0 at 0.2, 0.3, 0.5, 1, 2, 4; state 1 at 0.444,
    # 0.667, 1.111, 2.222, 4.444, 8.889. π = (0.6, 0.4): B_f = 0.6 x 3 + 0.4 x 1.35 = 2.34.
    # Spend: 4.68 below λ 0.2, then 4.08, 3.48, 3.30, 2.70, 2.52 on (0.667, 1] (state 0
    # at h 5, tied with h 3 at λ = 1 and so at the larger), then 1.92 on (1, 1.111]:
    # λ* = 1 from above, h = (3, 8). λ = 1 fails and λ_hi = 2 holds, so the bisection
    # runs on (1, 2] and ends within a few ulp of 1.
    t = np.array([10.0, 6.0, 4.0, 3.0, 2.5, 2.2, 2.0])
    i = np.arange(7.0)
    out = C.pooled_budget_intervals([0.6, 0.4], np.array([t, t]), np.array([i, 0.45 * i]))
    assert out.h == (3, 8)
    assert 1.0 < out.lam <= 1.0 + 1e-12
    assert out.budget == pytest.approx(2.34, rel=1e-15)
    assert out.spend == pytest.approx(1.92, rel=1e-15) and out.spend <= out.budget


def test_at_a_degenerate_breakpoint_rounding_picks_a_tied_h_that_meets_the_budget():
    # Collinear T and δ tie every h of state 0 at λ = 1 exactly. The bisection ends a few
    # ulp above 1, where rounding, not arithmetic, separates the tied values; the lock's
    # procedure is kept literal, and what it returns still meets the budget, because
    # h_k(λ_hi) is the very evaluation that passed it.
    steps = np.arange(7.0)
    cost = np.array([10.0 - steps, 10.0 - steps])
    track = np.array([steps, 0.5 * steps])
    out = C.pooled_budget_intervals([0.5, 0.5], cost, track)
    assert out.h[1] == 21 and out.h[0] < 21
    assert 1.0 < out.lam <= 1.0 + 1e-12 and out.spend <= out.budget == 2.25


def test_a_large_turnover_scale_doubles_lambda_past_one():
    menu = np.array(C.H_MENU, dtype=float)
    cost = 1e4 / np.sqrt(menu)[None, :].repeat(2, axis=0)
    track = np.sqrt(menu - 1)[None, :].repeat(2, axis=0) * np.array([[1.0], [3.0]])
    out = C.pooled_budget_intervals([0.5, 0.5], cost, track)
    assert out.lam > 1.0 and out.spend <= out.budget


# ---------------------------------------------------------------- the rule on a fold


def test_the_rule_s_inputs_are_the_lock_s_means_over_lagged_training_cells(data, planted):
    space, paths, target = planted["setup"].space, planted["paths"], planted["target"]
    arm = _arm(space, paths)
    fold = data.folds[1]
    rule = arm.rules[1]
    train = fold.train(data.sessions)
    lagged = T.lagged_path(paths, fold.number).loc[train]
    labelled = lagged.notna()
    for i, h in enumerate(C.H_MENU):
        literal = C.cadence_book(target, h)
        frame = pd.DataFrame({"s": lagged, "tau": literal.turnover.loc[train],
                              "delta": literal.delta.loc[train]})[labelled]
        means = frame.groupby("s")[["tau", "delta"]].mean()
        for j, state in enumerate(rule.states):
            assert rule.turnover[j, i] == pytest.approx(means.loc[state, "tau"], rel=1e-12)
            assert rule.tracking[j, i] == pytest.approx(means.loc[state, "delta"], rel=1e-12)
    shares = lagged[labelled].value_counts() / labelled.sum()
    np.testing.assert_allclose(rule.pi, shares.loc[list(rule.states)].to_numpy(), rtol=1e-12)


def test_the_training_check_is_a_cadence_book_along_the_lagged_training_path(data, planted):
    space, paths, target = planted["setup"].space, planted["paths"], planted["target"]
    arm = _arm(space, paths)
    ran = [r for r in arm.rules if not r.holds_twin]
    assert ran, "the planted partition should run its rule in some fold"
    for rule in ran:
        fold = next(f for f in data.folds if f.number == rule.fold)
        train = fold.train(data.sessions)
        lagged = T.lagged_path(paths, fold.number).loc[train]
        h = pd.Series(C.TWIN_H, index=data.sessions)
        h.loc[train] = lagged.map(rule.mapping()).fillna(C.TWIN_H).astype(int).to_numpy()
        literal = C.cadence_book(target, h)
        twin = C.cadence_book(target, C.TWIN_H)
        read_at = train[lagged.notna().to_numpy()]
        assert rule.check[0] == pytest.approx(literal.turnover.loc[read_at].mean(), rel=1e-12)
        assert rule.check[1] == pytest.approx(literal.delta.loc[read_at].mean(), rel=1e-12)
        assert rule.check[2] == pytest.approx(twin.turnover.loc[read_at].mean(), rel=1e-12)
        assert rule.check[3] == pytest.approx(twin.delta.loc[read_at].mean(), rel=1e-12)
        # the rule is kept only if it trades strictly less and tracks no worse
        assert rule.check[0] < rule.check[2] and rule.check[1] <= rule.check[3]


def test_a_rule_that_fails_its_training_check_holds_the_twin(data, primary):
    setup = C._setup(data)
    space = setup.space
    draws = T.placebo_draws(primary.paths, n=6)
    layout = C.path_layout(primary.paths, space)
    states = C.train_states(T.cells(primary.paths), space)
    failed = []
    for j in range(draws.n):
        arm = C.conditional_arm(space, layout, draws.codes[:, j].astype(np.int64), states)
        failed += [r for r in arm.rules if r.reason.startswith("training check")]
        for rule in arm.rules:
            if rule.holds_twin:
                assert set(rule.mapping().values()) <= {C.TWIN_H}
    assert failed
    for rule in failed:
        assert rule.holds_twin and not all(h == C.TWIN_H for h in rule.intervals.h)
        assert not rule.check[0] < rule.check[2] or rule.check[1] > rule.check[3]


# ---------------------------------------------------------------- the conditional arm


def test_the_arm_is_the_twin_before_the_test_and_reads_each_fold_lagged(data, planted):
    space, paths, target = planted["setup"].space, planted["paths"], planted["target"]
    arm = _arm(space, paths)
    first_test = int(space.test[0])
    np.testing.assert_array_equal(arm.h[:first_test], C.TWIN_H)
    np.testing.assert_array_equal(arm.source[:first_test],
                                  space.menu_source[C._TWIN][:first_test])
    for fold, rule in zip(data.folds, arm.rules, strict=True):
        test = fold.test(data.sessions)
        lagged = T.lagged_path(paths, fold.number).loc[test]
        stamped = paths.xs(fold.number, level="fold").droplevel("segment")
        assert lagged.iloc[0] == stamped.loc[fold.train(data.sessions)[-1]]
        expected = lagged.map(rule.mapping()).fillna(C.TWIN_H).astype(int).to_numpy()
        np.testing.assert_array_equal(arm.h[data.sessions.get_indexer(test)], expected)
    # one book over the full path: the counter carries across fold boundaries
    literal = C.cadence_book(target, arm.h)
    np.testing.assert_array_equal(space.held(arm.source).to_numpy(), literal.held.to_numpy())


def test_the_saving_and_the_gate_are_the_literal_books_read_on_the_test(data, planted):
    space, paths, target = planted["setup"].space, planted["paths"], planted["target"]
    arm = _arm(space, paths)
    test = data.test_sessions
    literal = C.cadence_book(target, arm.h)
    twin = C.cadence_book(target, C.TWIN_H)
    saving = (protocol.annual_turnover(twin.held.loc[test])
              - protocol.annual_turnover(literal.held.loc[test]))
    assert arm.saving == pytest.approx(saving, rel=1e-12, abs=1e-12)
    assert arm.mean_delta == pytest.approx(literal.delta.loc[test].mean(), rel=1e-12)
    assert space.twin_delta == pytest.approx(twin.delta.loc[test].mean(), rel=1e-12)
    assert C._gate(arm, space) == bool(arm.mean_delta <= space.twin_delta)


def test_a_planted_cadence_effect_is_found_and_an_unrelated_partition_is_not(planted):
    setup = planted["setup"]
    readings = {}
    for name in ("paths", "unrelated"):
        paths = planted[name]
        draws = T.placebo_draws(paths, n=99)
        inst = C._instrument(setup, T.PRIMARY, T.Partition(paths, None), draws, 1)
        arm = C.conditional_arm(setup.space, inst.layout, C.state_codes(paths), inst.states)
        readings[name] = (arm, protocol.placebo_p_value(arm.saving, inst.null))
    found, p_found = readings["paths"]
    assert p_found == pytest.approx(1 / 100) and found.saving > 0
    assert C._gate(found, setup.space)
    assert all(r.holds_twin is False for r in found.rules)
    _, p_unrelated = readings["unrelated"]
    assert p_unrelated > 0.05


# ---------------------------------------------------------------- causality


def test_nothing_after_a_date_changes_the_book_before_it():
    dates = pd.bdate_range("2020-01-01", periods=300)
    rng = np.random.default_rng(0)
    w = pd.DataFrame(rng.normal(size=(300, 5)), index=dates)
    w.iloc[:10] = np.nan
    h = rng.choice(C.H_MENU, size=300)
    cut = 150
    w2, h2 = w.copy(), h.copy()
    w2.iloc[cut + 1:] = rng.normal(size=(300 - cut - 1, 5))
    h2[cut + 1:] = rng.choice(C.H_MENU, size=300 - cut - 1)
    a, b = C.cadence_book(w, h), C.cadence_book(w2, h2)
    np.testing.assert_array_equal(a.held.iloc[:cut + 1].to_numpy(),
                                  b.held.iloc[:cut + 1].to_numpy())
    np.testing.assert_array_equal(a.delta.iloc[:cut + 1].to_numpy(),
                                  b.delta.iloc[:cut + 1].to_numpy())


def test_a_fold_s_rule_reads_nothing_after_its_training_window(data, planted):
    space, paths, target = planted["setup"].space, planted["paths"], planted["target"]
    fold = data.folds[1]
    after = data.sessions > fold.train_end
    changed = target.copy()
    changed.loc[after] = changed.loc[after].to_numpy() * 1.7 + 0.01
    relabelled = paths.copy()
    rows = relabelled.index.get_level_values("session") > fold.train_end
    relabelled[rows] = (relabelled[rows] + 1) % 4
    other = C.cadence_space(changed, data.sessions, data.folds)
    a = _arm(space, paths).rules[:2]
    b = _arm(other, relabelled).rules[:2]
    for x, y in zip(a, b, strict=True):
        assert x.mapping() == y.mapping() and x.reason == y.reason
        assert x.check == y.check or np.isnan(x.check).all() and np.isnan(y.check).all()
        np.testing.assert_array_equal(x.turnover, y.turnover)
        np.testing.assert_array_equal(x.tracking, y.tracking)


def test_relabelling_the_test_after_a_date_changes_nothing_before_it(data, planted):
    space, paths = planted["setup"].space, planted["paths"]
    cut = data.test_sessions[400]
    changed = paths.copy()
    rows = ((changed.index.get_level_values("segment") == "test")
            & (changed.index.get_level_values("session") >= cut))
    changed[rows] = (changed[rows] + 2) % 4
    a, b = _arm(space, paths), _arm(space, changed)
    upto = data.sessions.get_loc(cut)  # the state stamped on `cut` steers the next session
    np.testing.assert_array_equal(a.source[:upto + 1], b.source[:upto + 1])
    assert not np.array_equal(a.source, b.source)
    assert [r.mapping() for r in a.rules] == [r.mapping() for r in b.rules]


# ---------------------------------------------------------------- determinism


def test_the_null_is_the_same_on_any_worker_count_and_moves_with_the_seed(planted):
    setup, paths = planted["setup"], planted["paths"]
    layout = C.path_layout(paths, setup.space)
    states = C.train_states(T.cells(paths), setup.space)
    draws = T.placebo_draws(paths, n=8, seed=0)
    serial = C.null_savings(setup.space, layout, draws, states, workers=1)
    pooled = C.null_savings(setup.space, layout, draws, states, workers=2)
    np.testing.assert_array_equal(serial, pooled)
    again = C.null_savings(setup.space, layout, T.placebo_draws(paths, n=8, seed=0), states)
    np.testing.assert_array_equal(serial, again)
    other = C.null_savings(setup.space, layout, T.placebo_draws(paths, n=8, seed=1), states)
    assert not np.array_equal(serial, other)


# ---------------------------------------------------------------- the instrument


INSTRUMENT_KEYS = {"variant", "K", "placebo", "cells", "twin", "menu_turnover",
                   "control_cap_share", "null", "MDE_C", "S_star", "powered", "undecidable"}


def test_the_instrument_prints_only_what_13_1_allows_and_never_builds_the_real_arm(
        data, primary, monkeypatch):
    calls = []
    real_arm = C.conditional_arm

    def counting(*args, **kwargs):
        calls.append(kwargs.get("detail", True))
        return real_arm(*args, **kwargs)

    def refuse(paths):
        raise AssertionError("the instrument must not read the real labels into an arm")

    monkeypatch.setattr(C, "conditional_arm", counting)
    monkeypatch.setattr(C, "state_codes", refuse)
    out = C.instrument(data, T.PRIMARY, N_SMALL, partition=primary)
    assert calls == [False] * N_SMALL  # the placebo arms only, saving only
    assert set(out) == INSTRUMENT_KEYS
    assert set(out["twin"]) == {"h", "turnover", "mean_delta", "sigma", "missing"}
    assert set(out["null"]) == {"q20", "q50", "q95", "q99", "q99-q50", "q99-q20", "non_finite",
                                "below_zero", "at_zero", "above_zero", "atom_share"}
    signs = (out["null"]["below_zero"], out["null"]["at_zero"], out["null"]["above_zero"])
    assert sum(signs) == N_SMALL
    assert set(out["menu_turnover"]) == {str(h) for h in C.H_MENU}
    assert out["placebo"] == {"method": "uniform", "draws": N_SMALL, "seed": 0, "exact": True}
    assert out["cells"]["test_sessions"] == len(data.test_sessions)
    assert out["undecidable"] == [] and out["null"]["non_finite"] == 0
    json.dumps(T.encode_thresholds(out))
    text = json.dumps(out).lower()
    for forbidden in ("s_c", "gate", "h_f", "lambda", "sharpe", "rules"):
        assert f'"{forbidden}' not in text


def test_the_instrument_s_thresholds_are_the_lock_s_formulas(planted):
    setup, paths = planted["setup"], planted["paths"]
    draws = T.placebo_draws(paths, n=49)
    inst = C._instrument(setup, T.PRIMARY, T.Partition(paths, None), draws, 1)
    q20, q50, q95, q99 = np.percentile(inst.null, [20, 50, 95, 99])
    out = inst.printout
    assert (out["null"]["q20"], out["null"]["q50"], out["null"]["q95"], out["null"]["q99"]) == (
        q20, q50, q95, q99)
    assert out["MDE_C"] == q99 - q20
    sigma = T.realised_sd(setup.twin.net5)
    assert out["twin"]["sigma"] == sigma
    assert out["S_star"] == 0.05 * sigma * 10_000 / 5
    assert out["powered"] == (0 < out["MDE_C"] <= out["S_star"]
                              and out["null"]["atom_share"] < C.ATOM_SHARE)
    assert (out["null"]["below_zero"], out["null"]["at_zero"], out["null"]["above_zero"]) == (
        int((inst.null < 0).sum()), int((inst.null == 0).sum()), int((inst.null > 0).sum()))
    assert out["twin"]["turnover"] == setup.space.twin_turnover
    assert out["twin"]["turnover"] == pytest.approx(
        protocol.annual_turnover(C.cadence_book(planted["target"], 5).held.loc[
            setup.twin.test]), rel=1e-12)
    again = C._instrument(setup, T.PRIMARY, T.Partition(paths, None), draws, 1)
    assert C._encoded(again.printout) == C._encoded(out)


@pytest.mark.parametrize("variant", [T.SMOOTH21, T.K3])
def test_the_instrument_runs_on_the_sensitivities(data, variant):
    partition = T.partition_paths(data, variant)
    if variant.smoothing:
        assert partition.paths.isna().any()  # the first 20 sessions of each fold's path
    out = C.instrument(data, variant, 5, partition=partition)
    assert out["K"] == variant.K and out["variant"] == variant.name
    assert out["placebo"]["exact"] and out["undecidable"] == []
    assert out["null"]["non_finite"] == 0
    assert C.section_name(variant) == f"C:{variant.name}"


def test_s_star_and_the_c2_bound():
    assert C.s_star(0.0788) == pytest.approx(7.88)
    assert C.c2_bound(10.0, 0.08, 5.0) == pytest.approx(10 * 5 / 10_000 / 0.08)
    assert C.c2_bound(10.0, 0.08, 20.0) == pytest.approx(4 * C.c2_bound(10.0, 0.08, 5.0))
    assert np.isnan(C.c2_bound(np.nan, 0.08, 5.0)) and np.isnan(C.c2_bound(1.0, 0.0, 5.0))


def test_variants_sections_and_the_pit_rebuild_variant(data):
    assert C.section_name(T.PRIMARY) == "C" and C.section_name(T.K3) == "C:K=3"
    assert C.pit_variant(T.PRIMARY) is T.PIT18
    rebuilt = C.pit_variant(T.K5)
    assert rebuilt.K == 5 and rebuilt.features == T.PIT_FEATURES and rebuilt.role == "control"
    assert rebuilt.name == "pit18[K=5]"
    smoothed = C.pit_variant(T.SMOOTH21)
    assert smoothed.smoothing == 21 and smoothed.features == T.PIT_FEATURES
    with pytest.raises(ValueError, match="not evaluated at level C"):
        C.instrument(data, T.D025, 2)
    with pytest.raises(ValueError, match="never instrumented or read"):
        C.instrument(data, T.PIT18, 2)
    with pytest.raises(ValueError, match="never instrumented or read"):
        C.read(data, T.PIT18, 2, {})
    with pytest.raises(ValueError, match="not committed"):
        C.section_name(T.PIT18)
    with pytest.raises(ValueError, match="already the PIT rebuild"):
        C.pit_variant(T.PIT18)


def test_the_layout_refuses_a_path_that_is_not_a_fold_s_training_then_test(data, planted):
    space, paths = planted["setup"].space, planted["paths"]
    with pytest.raises(ValueError, match="training then test"):
        C.path_layout(paths.drop(paths.index[5]), space)
    with pytest.raises(ValueError, match="indexed by"):
        C.path_layout(paths.droplevel("segment"), space)


# ---------------------------------------------------------------- verdicts


BASE = {"s_c": 8.0, "p": 0.001, "exact": True, "gate": True, "s_w": 1.0, "mde_c": 2.0,
        "s_star": 7.0, "atom": 0.0, "leg_missing": False}


@pytest.mark.parametrize("change, verdict", [
    ({}, T.PASS),
    ({"s_c": np.nan}, T.UNDECIDABLE),
    ({"p": np.nan}, T.UNDECIDABLE),
    ({"exact": False}, T.UNDECIDABLE),
    ({"leg_missing": True}, T.UNDECIDABLE),
    ({"mde_c": np.nan}, T.UNDECIDABLE),
    ({"s_star": np.nan}, T.UNDECIDABLE),
    ({"gate": None}, T.UNDECIDABLE),
    ({"s_w": np.nan}, T.UNDECIDABLE),
    ({"gate": False, "p": np.nan}, T.UNDECIDABLE),
    ({"gate": False}, T.FAIL),
    ({"gate": False, "mde_c": 9.0}, T.FAIL),
    ({"p": 0.02}, T.FAIL),
    ({"p": 0.02, "mde_c": 9.0}, T.NOT_SHOWN),
    ({"p": 0.009}, T.FAIL),
    ({"p": 0.009, "holm_rejected": True}, T.PASS),
    ({"p": 0.011, "holm_rejected": True}, T.FAIL),
    ({"p": 0.001, "holm_rejected": False, "mde_c": 9.0}, T.NOT_SHOWN),
    ({"s_w": 8.0}, T.DOMINATED),
    ({"s_w": 6.0, "p": 0.02, "mde_c": 9.0}, T.NOT_SHOWN),
    ({"s_c": -1.0, "p": 1.0}, T.FAIL),
    ({"s_c": -1.0, "p": 1.0, "mde_c": 0.0}, T.NOT_SHOWN),
    ({"s_c": 0.0, "p": 1.0, "mde_c": 0.0}, T.NOT_SHOWN),
    ({"gate": False, "mde_c": 0.0}, T.FAIL),
    # amendment of 2026-09-23: the lock holds only if S_C >= S*, and power is unmeasured
    # when the null has an atom of 20% or more
    ({"s_c": 7.0}, T.PASS),
    ({"s_c": 5.0}, T.FAIL),
    ({"s_c": 5.0, "atom": 0.5}, T.NOT_SHOWN),
    ({"s_c": 1.5, "p": 0.001, "mde_c": 0.0}, T.NOT_SHOWN),
    ({"s_c": 1.5, "p": 0.001, "mde_c": 0.0, "atom": 1.0}, T.NOT_SHOWN),
    ({"p": 0.02, "atom": 0.2}, T.NOT_SHOWN),
    ({"p": 0.02, "atom": 0.19}, T.FAIL),
    ({"atom": np.nan}, T.UNDECIDABLE),
    ({"s_c": 20.0, "s_w": 20.0}, T.DOMINATED),
])
def test_the_c1_lines_apply_in_the_lock_s_order(change, verdict):
    assert C.c1_verdict(**{**BASE, **change}) == verdict


def test_power_is_unmeasured_when_the_null_has_an_atom():
    """Amendment of 2026-09-23: ``MDE_C = 0`` (q20 = q99) measures no power, and neither
    does ``q99 − q20`` when one value holds 20% or more of the draws (§13.2)."""
    assert C.powered(2.0, 7.0, 0.0) and C.powered(7.0, 7.0, 0.19)
    assert not C.powered(0.0, 7.0, 0.0) and not C.powered(9.0, 7.0, 0.0)
    assert not C.powered(2.0, 7.0, 0.2) and not C.powered(2.0, 7.0, 1.0)
    assert not C.powered(np.nan, 7.0, 0.0) and not C.powered(2.0, np.nan, 0.0)
    assert not C.powered(2.0, 7.0, np.nan)
    atom = np.zeros(1000)
    summary = T.null_summary(atom)
    assert summary.q20 == summary.q99 == 0.0 and summary.mde_c == 0.0
    assert C.atom_share(atom) == 1.0
    assert not C.powered(summary.mde_c, 7.84, C.atom_share(atom))
    mixed = np.r_[np.zeros(300), np.linspace(-5.0, 5.0, 700)]
    assert C.atom_share(mixed) == pytest.approx(0.3)
    assert C.atom_share(np.array([1.0, np.nan])) != C.atom_share(np.array([1.0, np.nan]))
    assert C.atom_share(np.arange(10.0)) == pytest.approx(0.1)
    assert C.null_signs(np.array([-1.0, 0.0, 0.0, 2.0, np.nan])) == {
        "below_zero": 1, "at_zero": 2, "above_zero": 1}


def test_content_free_labels_hold_the_twin_so_the_null_is_an_atom_at_zero(data, primary):
    """The finding behind candidate amendment (a), on synthetic data: when the target's
    drift does not depend on the state, every placebo draw's rule holds the twin in every
    fold, ``S^(j) = 0`` exactly, ``MDE_C = 0``, and the instrument reports power as
    unmeasured (it read ``MDE_C ≤ S*``, powered, before the candidate)."""
    out = C.instrument(data, T.PRIMARY, N_SMALL, partition=primary)
    assert out["null"]["q20"] == out["null"]["q99"] == 0.0 and out["MDE_C"] == 0.0
    assert out["null"]["at_zero"] == N_SMALL and out["null"]["atom_share"] == 1.0
    assert out["S_star"] > 0 and out["powered"] is False


def test_pit_downgrades_only_a_pass_and_the_level_is_c1_s():
    assert T.apply_pit(C.c1_verdict(**BASE), T.PASS) == T.PASS
    assert T.apply_pit(C.c1_verdict(**BASE), T.NOT_SHOWN) == T.DOWNGRADED_PIT
    assert T.apply_pit(C.c1_verdict(**{**BASE, "p": 0.02}), None) == T.FAIL
    assert T.level_verdict([T.DOWNGRADED_PIT]) == T.DOWNGRADED_PIT


def test_c2_is_a_bound_and_never_a_pass():
    assert C.c2_verdict(np.nan) == T.UNDECIDABLE
    assert C.c2_verdict(0.0) == T.BOUND and C.c2_verdict(-0.01) == T.BOUND
    assert C.c2_verdict(0.02) == T.BOUND


# ---------------------------------------------------------------- non-finite readings


def test_a_nan_placebo_draw_is_undecidable_and_never_dropped():
    null = np.array([0.0, 1.0, np.nan])
    assert np.isnan(protocol.placebo_p_value(5.0, null))
    assert np.isnan(T.null_summary(null).mde_c)
    assert C.c1_verdict(**{**BASE, "p": protocol.placebo_p_value(5.0, null)}) == T.UNDECIDABLE


def test_an_undefined_training_target_leaves_the_rule_undefined(data, planted):
    target, paths = planted["target"].copy(), planted["paths"]
    lagged = T.lagged_path(paths, data.folds[0].number)
    rows = lagged.index[(lagged == 3).to_numpy()]
    target.loc[rows] = np.nan
    space = C.cadence_space(target, data.sessions, data.folds)
    arm = _arm(space, paths)
    assert not arm.rules[0].defined and np.isnan(arm.saving) and arm.source is None
    verdict = C.c1_verdict(**{**BASE, "s_c": arm.saving,
                              "p": protocol.placebo_p_value(arm.saving, np.zeros(9))})
    assert verdict == T.UNDECIDABLE


def test_a_missing_twin_leg_is_undecidable_before_the_reading_and_spends_no_row(
        data, planted, monkeypatch):
    target = planted["target"].copy()
    target.loc[data.test_sessions[300:306]] = np.nan  # six sessions hold a twin rebalance
    setup = C._setup(data, target)
    assert setup.twin.missing > 0
    paths = planted["paths"]
    partition = T.Partition(paths, None)
    draws = T.placebo_draws(paths, n=9)
    inst = C._instrument(setup, T.PRIMARY, partition, draws, 1)
    out = inst.printout
    assert out["twin"]["missing"] == setup.twin.missing
    assert out["twin"]["sigma"] == C.NON_FINITE and out["S_star"] == C.NON_FINITE
    assert any("twin has no net return" in r for r in out["undecidable"])
    T.encode_thresholds(out)  # a non-finite reading is recorded, never refused as a number

    def refuse(*args, **kwargs):
        raise AssertionError("a lock UNDECIDABLE before its reading is not read")

    monkeypatch.setattr(C, "_c1_reading", refuse)
    monkeypatch.setattr(C, "_witness_saving", refuse)
    reading = C._reading(data, setup, T.PRIMARY, partition, inst, pit=None, workers=1)
    assert reading["verdicts"] == {"C-1": T.UNDECIDABLE, "C-2": T.UNDECIDABLE,
                                   "C": T.UNDECIDABLE}
    assert reading["trial_rows"] == {} and not reading["C-1"]["read"]
    assert reading["C-1"]["undecidable"] == out["undecidable"]


def test_a_non_finite_placebo_draw_is_recorded_and_makes_c1_undecidable(data, planted,
                                                                        monkeypatch):
    setup, paths = planted["setup"], planted["paths"]
    real = C.null_savings

    def one_broken(*args, **kwargs):
        null = real(*args, **kwargs)
        null[3] = np.nan
        return null

    monkeypatch.setattr(C, "null_savings", one_broken)
    partition = T.Partition(paths, None)
    inst = C._instrument(setup, T.PRIMARY, partition, T.placebo_draws(paths, n=9), 1)
    out = inst.printout
    assert out["null"]["non_finite"] == 1 and out["MDE_C"] == C.NON_FINITE
    signs = ("below_zero", "at_zero", "above_zero")
    assert {v for k, v in out["null"].items() if k not in signs} - {1} == {C.NON_FINITE}
    assert sum(out["null"][k] for k in signs) == 8  # the finite draws, counted by sign
    assert out["powered"] is False
    assert any("1 of 9 placebo draws" in r for r in out["undecidable"])
    json.dumps(T.encode_thresholds(out))
    reading = C._reading(data, setup, T.PRIMARY, partition, inst, pit=None, workers=1)
    assert reading["verdicts"]["C-1"] == T.UNDECIDABLE and reading["trial_rows"] == {}


# ---------------------------------------------------------------- the reading


#: The planted saving (about 10x/yr) sits below S* at the lock's 0.05 Sharpe on this
#: synthetic twin; the tests that walk the PASS path lower S* to reach it, and
#: test_a_planted_saving_below_s_star_is_not_shown checks the size rule itself.
SMALL_S_STAR = 0.01


def test_a_planted_saving_below_s_star_is_not_shown(data, planted):
    """Amendment of 2026-09-23: p ≤ 0.01, the gate and the witness all hold, yet a saving
    below S* does not make the lock hold; the power is unmeasured on an atom null, so the
    verdict is NOT SHOWN, never FAIL."""
    setup, paths = planted["setup"], planted["paths"]
    partition = T.Partition(paths, None)
    draws = T.placebo_draws(paths, n=149)
    inst = C._instrument(setup, T.PRIMARY, partition, draws, 1)
    reading = C._reading(data, setup, T.PRIMARY, partition, inst, pit=None, workers=1)
    c1 = reading["C-1"]
    assert c1["p"] == pytest.approx(1 / 150) and c1["gate"] is True
    assert 0 < c1["S_C"] < inst.s_star and c1["S_W"] < c1["S_C"]
    assert inst.printout["powered"] is False
    assert reading["verdicts"]["C-1"] == T.NOT_SHOWN and reading["pit"] is None


def test_a_planted_effect_reads_pass_through_the_pit_rebuild(data, planted, monkeypatch):
    monkeypatch.setattr(C, "S_STAR_SHARPE", SMALL_S_STAR)
    setup, paths, unrelated = planted["setup"], planted["paths"], planted["unrelated"]
    partition = T.Partition(paths, None)
    draws = T.placebo_draws(paths, n=149)
    inst = C._instrument(setup, T.PRIMARY, partition, draws, 1)
    kept = C._reading(data, setup, T.PRIMARY, partition, inst, pit=(partition, draws),
                      workers=1)
    c1 = kept["C-1"]
    assert c1["p"] == pytest.approx(1 / 150) and c1["gate"] is True
    assert c1["S_W"] < c1["S_C"] and c1["verdict_before_pit"] == T.PASS
    assert kept["verdicts"] == {"C-1": T.PASS, "C-2": T.BOUND, "C": T.PASS}
    assert kept["pit"]["verdict"] == T.PASS
    assert kept["holm_p"] == {"C-1": c1["p"], "C-2": 1.0}
    sigma = setup.twin.sigma
    assert kept["C-2"]["bound"]["5.0"] == pytest.approx(c1["S_C"] * 5 / 10_000 / sigma)
    assert kept["C-2"]["bound"]["20.0"] == pytest.approx(4 * kept["C-2"]["bound"]["5.0"])
    control = setup.control
    assert kept["C-2"]["control_cost"]["5.0"] == pytest.approx(
        control.turnover * 5 / 10_000 / control.sigma)
    rows = kept["trial_rows"]
    assert list(rows) == ["C-1", "C-2"]
    assert rows["C-1"]["delta"] == c1["S_C"] and rows["C-1"]["threshold"] == inst.summary.q99
    literal = C.cadence_book(planted["target"], _arm(setup.space, paths).h, excess=data.excess)
    assert rows["C-1"]["sharpe"] == pytest.approx(
        T.sharpe(literal.net.loc[data.test_sessions]), rel=1e-12)
    assert rows["C-2"]["p"] == 1.0 and rows["C-2"]["verdict"] == T.BOUND

    downgraded = C._reading(
        data, setup, T.PRIMARY, partition, inst,
        pit=(T.Partition(unrelated, None), T.placebo_draws(unrelated, n=149)), workers=1)
    assert downgraded["verdicts"]["C-1"] == T.DOWNGRADED_PIT
    assert downgraded["verdicts"]["C"] == T.DOWNGRADED_PIT
    assert downgraded["pit"]["verdict"] != T.PASS

    null_only = C._reading(data, setup, T.PRIMARY, T.Partition(unrelated, None),
                           C._instrument(setup, T.PRIMARY, T.Partition(unrelated, None),
                                         T.placebo_draws(unrelated, n=149), 1),
                           pit=None, workers=1)
    assert null_only["pit"] is None  # computed only for a lock that could pass
    assert null_only["verdicts"]["C-1"] in (T.FAIL, T.NOT_SHOWN)


def test_the_final_holm_step_can_pass_a_lock_between_the_bars_through_its_rebuild(
        data, planted, monkeypatch):
    monkeypatch.setattr(C, "S_STAR_SHARPE", SMALL_S_STAR)
    setup, paths = planted["setup"], planted["paths"]
    partition = T.Partition(paths, None)
    draws = T.placebo_draws(paths, n=110)  # no draw at or above: p = 1/111, in (0.05/6, 0.01]
    inst = C._instrument(setup, T.PRIMARY, partition, draws, 1)
    reading = C._reading(data, setup, T.PRIMARY, partition, inst,
                         pit=(partition, T.placebo_draws(paths, n=149)), workers=1)
    assert reading["C-1"]["p"] == pytest.approx(1 / 111)
    assert T.BONFERRONI < reading["C-1"]["p"] <= T.PLACEBO_P_BAR
    powered = inst.printout["powered"]
    assert reading["verdicts"]["C-1"] == (T.FAIL if powered else T.NOT_SHOWN)
    assert reading["pit"] is not None and reading["pit"]["verdict"] == T.PASS
    assert C.lock_verdicts(reading, {"C-1", "A-1"})["C-1"] == T.PASS
    assert C.lock_verdicts(reading, {"A-1"})["C-1"] == reading["verdicts"]["C-1"]
    assert reading["trial_rows"]["C-1"]["verdict"] == reading["verdicts"]["C-1"]  # provisional
    sensitivity = {**reading, "role": "sensitivity"}
    with pytest.raises(ValueError, match="six primaries"):
        C.lock_verdicts(sensitivity, {"C-1"})


def test_a_sensitivity_logs_one_level_row(data, planted):
    setup, paths = planted["setup"], planted["paths"]
    partition = T.Partition(paths, None)
    draws = T.placebo_draws(paths, n=9)
    inst = C._instrument(setup, T.K3, partition, draws, 1)
    out = C._reading(data, setup, T.K3, partition, inst, pit=None, workers=1)
    assert list(out["trial_rows"]) == ["C"]
    row = out["trial_rows"]["C"]
    assert np.isnan(row["p2"]) and row["delta"] == out["C-1"]["S_C"]
    assert row["verdict"] == out["verdicts"]["C"] == out["verdicts"]["C-1"]
    assert "not robust" not in row["note"]


def test_a_sensitivity_row_with_delta_at_most_zero_is_noted_not_robust_under_a_pass():
    reading = {
        "test_sessions": 100,
        "verdicts": {"C-1": T.NOT_SHOWN, "C-2": T.FAIL, "C": T.NOT_SHOWN},
        "C-1": {"read": True, "sharpe": 0.1, "S_C": 0.0, "q99": 2.0, "p": 0.5,
                "placebo_pct": 0.3, "undecidable": []},
        "C-2": {"read": True, "bound": {"5.0": 0.0, "10.0": 0.0, "20.0": 0.0}},
    }
    row = C.trial_rows(reading, T.K5, primary_level=T.PASS)["C"]
    assert "PASS (not robust)" in row["note"] and row["verdict"] == T.NOT_SHOWN
    assert "not robust" not in C.trial_rows(reading, T.K5)["C"]["note"]
    positive = {**reading, "C-1": {**reading["C-1"], "S_C": 0.5}}
    assert "not robust" not in C.trial_rows(positive, T.K5, primary_level=T.PASS)["C"]["note"]
    assert C.trial_rows(reading, T.PIT18) == {}
    undecidable = {**reading, "verdicts": {**reading["verdicts"], "C": T.UNDECIDABLE},
                   "C-1": {**reading["C-1"], "undecidable": ["S_C is not finite (fold 1: x)"]}}
    assert "S_C is not finite" in C.trial_rows(undecidable, T.K5)["C"]["note"]


INPUTS = ("data/raw/a.parquet", "data/cache/b.parquet")


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t",
                    *args], check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    for relative in INPUTS:
        (tmp_path / relative).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / relative).write_bytes(relative.encode() * 10)
    shutil.copy(ROOT / "uv.lock", tmp_path / "uv.lock")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", "uv.lock")
    _git(tmp_path, "commit", "-q", "-m", "init")
    return tmp_path


def _commit_thresholds(repo: Path, values: dict) -> Path:
    path = repo / T.THRESHOLDS_FILE
    T.write_thresholds(values, section="C", path=path, root=repo, inputs=INPUTS)
    _git(repo, "add", T.THRESHOLDS_FILE)
    _git(repo, "commit", "-q", "-m", "thresholds")
    return path


def test_the_reading_runs_only_on_committed_bitwise_thresholds(data, primary, repo, tmp_path):
    draws = T.placebo_draws(primary.paths, n=N_SMALL)
    values = C.instrument(data, T.PRIMARY, draws, partition=primary)
    path = repo / T.THRESHOLDS_FILE
    kwargs = {"partition": primary, "root": repo, "path": path, "inputs": INPUTS}

    T.write_thresholds(values, section="C", path=path, root=repo, inputs=INPUTS)
    with pytest.raises(T.ReadingRefused, match="not tracked"):
        C.read(data, T.PRIMARY, draws, values, **kwargs)
    with pytest.raises(T.ReadingRefused, match="not tracked"):
        C.read(data, T.PRIMARY, draws, None, **kwargs)
    _git(repo, "add", T.THRESHOLDS_FILE)
    _git(repo, "commit", "-q", "-m", "thresholds")
    stored = T.read_thresholds(path)["thresholds"]["C"]
    out = C.read(data, T.PRIMARY, draws, stored, **kwargs)
    assert out["verdicts"]["C-1"] in T.LOCK_VERDICTS and out["verdicts"]["C"] in T.LOCK_VERDICTS
    assert out["instrument"] == values
    again = C.read(data, T.PRIMARY, draws, None, **kwargs)
    assert C._encoded(again["verdicts"]) == C._encoded(out["verdicts"])
    trials = tmp_path / "trials.parquet"
    for test, row in out["trial_rows"].items():
        T.log_trial(test, T.PRIMARY, path=trials, **row)
    assert len(pd.read_parquet(trials)) == 2
    with pytest.raises(ValueError, match="already logged"):
        T.log_trial("C-1", T.PRIMARY, path=trials, **out["trial_rows"]["C-1"])

    path.write_text(path.read_text() + " ")
    with pytest.raises(T.ReadingRefused, match="differs from its committed version"):
        C.read(data, T.PRIMARY, draws, stored, **kwargs)
    _git(repo, "checkout", "--", T.THRESHOLDS_FILE)
    (repo / INPUTS[0]).write_bytes(b"republished")
    with pytest.raises(T.ReadingRefused, match="SHA-256"):
        C.read(data, T.PRIMARY, draws, stored, **kwargs)


def test_the_reading_refuses_before_computing_anything(data, primary, repo, monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("nothing may be computed before the verification")

    monkeypatch.setattr(C, "_instrument", refuse)
    monkeypatch.setattr(C, "_setup", refuse)
    with pytest.raises(T.ReadingRefused, match="no threshold file"):
        C.read(data, T.PRIMARY, 2, {"any": 1.0}, partition=primary, root=repo,
               path=repo / T.THRESHOLDS_FILE, inputs=INPUTS)


def test_the_reading_refuses_a_committed_instrument_it_does_not_recompute(data, primary, repo):
    draws = T.placebo_draws(primary.paths, n=N_SMALL)
    values = C.instrument(data, T.PRIMARY, draws, partition=primary)
    forged = {**values, "S_star": float(np.nextafter(values["S_star"], np.inf))}
    path = _commit_thresholds(repo, forged)
    with pytest.raises(T.ReadingRefused, match="recomputed thresholds differ bitwise"):
        C.read(data, T.PRIMARY, draws, forged, partition=primary, root=repo,
               path=path, inputs=INPUTS)
    with pytest.raises(T.ReadingRefused, match="bitwise"):
        C.read(data, T.PRIMARY, draws, values, partition=primary, root=repo,
               path=path, inputs=INPUTS)

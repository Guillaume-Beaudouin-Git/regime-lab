"""The matched placebo: exact clock, exact occupancy, exact episodes, and nothing else.

A placebo that is only approximately matched is the failure it exists to prevent — a
gain travelling through the dimension left unmatched — so every test here asserts
equality, never closeness. The two constructions are tested on the same footing:
``swap`` (the draft's construction 3, a port of the pre-lock script) and ``uniform``
(the proposed construction 4). All data are synthetic.
"""

from __future__ import annotations

import itertools
from collections import Counter

import numpy as np
import pandas as pd
import pytest
from scipy.stats import chisquare

from regime_lab.analysis.placebo import (
    MAX_TRIES,
    JoinFreeSequences,
    PlaceboError,
    _arrange,
    check_placebos,
    episodes,
    matched_placebos,
    occupancy,
    placebo_feasibility,
    run_lengths,
    span_years,
    transitions,
)

METHODS = ("swap", "uniform")


def _markov(n: int = 2_000, k: int = 4, stay: float = 0.93, seed: int = 0) -> pd.Series:
    """A persistent chain that always leaves to a uniformly drawn *other* state."""
    rng = np.random.default_rng(seed)
    moves = rng.random(n) > stay
    steps = rng.integers(1, k, n)
    states = np.cumsum(np.where(moves, steps, 0)) % k
    return pd.Series(states, index=pd.bdate_range("2001-01-01", periods=n), name="state")


def _from_episodes(states: list[int], seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    lengths = rng.integers(1, 12, len(states))
    values = np.repeat(np.asarray(states), lengths)
    return pd.Series(values, index=pd.bdate_range("2001-01-01", periods=len(values)))


def _dominant(n_episodes: int = 41) -> pd.Series:
    """State 0 on every other episode: the only join-free orders keep it there."""
    others = itertools.cycle([1, 2, 3])
    return _from_episodes([0 if i % 2 == 0 else next(others) for i in range(n_episodes)])


def _pre_lock(lens: np.ndarray, states: np.ndarray, seed: int, tries: int = 60):
    """The pre-lock script's construction, restated as it was written."""
    r = np.random.default_rng(seed)
    for _ in range(tries):
        o = r.permutation(len(lens))
        length, st = lens[o].copy(), states[o].copy()
        for i in range(1, len(st)):
            if st[i] == st[i - 1]:
                nxt = st[i + 1] if i + 1 < len(st) else -1
                j = next((k for k in range(i + 1, len(st))
                          if st[k] != st[i - 1] and st[k] != nxt
                          and (k + 1 >= len(st) or st[k + 1] != st[i])
                          and st[k - 1] != st[i]), None)
                if j is None:
                    break
                length[[i, j]] = length[[j, i]]
                st[[i, j]] = st[[j, i]]
        if not (st[1:] == st[:-1]).any():
            return np.concatenate([np.full(v, s) for v, s in zip(length, st, strict=True)])
    return None


# -------------------------------------------------------------------- primitives


def test_run_lengths_transitions_and_episodes():
    labels = pd.Series([2, 2, 0, 0, 0, 2, 1], index=pd.bdate_range("2020-01-01", periods=7))
    lengths, states = run_lengths(labels)
    assert lengths.tolist() == [2, 3, 1, 1]
    assert states.tolist() == [2, 0, 2, 1]
    assert transitions(labels) == 3
    frame = episodes(labels)
    assert frame["length"].tolist() == [2, 3, 1, 1]
    assert frame["start"].iloc[1] == labels.index[2]
    assert frame["end"].iloc[1] == labels.index[4]
    assert occupancy(labels).to_dict() == pytest.approx({0: 3 / 7, 1: 1 / 7, 2: 3 / 7})
    assert transitions(np.array([])) == 0


def test_span_years_is_first_to_last_session_in_years_of_365_25_days():
    index = pd.DatetimeIndex(["2000-01-01", "2000-06-30", "2004-01-01"])
    assert span_years(index) == pytest.approx(1_461 / 365.25)
    assert span_years(pd.DatetimeIndex(["2000-01-03"])) == 0.0
    assert np.isnan(span_years(pd.DatetimeIndex([])))


# ------------------------------------------------------------------- exactness


@pytest.mark.parametrize("method", METHODS)
def test_every_draw_is_exact_on_four_balanced_states(method):
    labels = _markov()
    placebo = matched_placebos(labels, 40, seed=3, method=method)
    check = check_placebos(labels, placebo.draws)
    assert check.exact
    assert (check.transitions == transitions(labels)).all()
    assert check.transitions_sd == 0.0
    assert check.max_occupancy_deviation == 0.0
    assert check.episodes_preserved
    assert list(placebo.draws.index) == list(labels.index)


@pytest.mark.parametrize("method", METHODS)
def test_draws_move_the_episodes(method):
    labels = _markov()
    draws = matched_placebos(labels, 5, seed=1, method=method).draws
    assert all((draws[c] != labels).mean() > 0.3 for c in draws.columns)


@pytest.mark.parametrize("method", METHODS)
def test_grouped_draws_are_exact_group_by_group(method):
    labels = _markov(n=2_400, seed=5)
    groups = pd.Series(np.repeat(["a", "b", "c"], 800), index=labels.index)
    placebo = matched_placebos(labels, 20, seed=0, groups=groups, method=method)
    check = check_placebos(labels, placebo.draws, groups=groups)
    assert check.exact
    for name, block in labels.groupby(groups):
        for column in placebo.draws.columns:
            drawn = placebo.draws.loc[block.index, column]
            assert transitions(drawn) == transitions(block)
            assert sorted(zip(*run_lengths(drawn)[::-1], strict=True)) == sorted(
                zip(*run_lengths(block)[::-1], strict=True)
            ), name


@pytest.mark.parametrize("method", METHODS)
def test_a_draw_depends_only_on_the_seed_and_its_number(method):
    labels = _markov(n=1_200, seed=2)
    few = matched_placebos(labels, 3, seed=11, method=method).draws
    many = matched_placebos(labels, 8, seed=11, method=method).draws
    pd.testing.assert_frame_equal(few, many.iloc[:, :3])
    other = matched_placebos(labels, 3, seed=12, method=method).draws
    assert not few.equals(other)


def test_labels_keep_their_values_and_type():
    labels = _markov(n=600).map({0: "calm", 1: "stress", 2: "boom", 3: "bust"})
    draws = matched_placebos(labels, 3, method="uniform").draws
    assert set(np.unique(draws.to_numpy())) == set(labels.unique())


# ---------------------------------------------------------------- the swap repair


def test_the_swap_repair_is_a_line_for_line_port_of_the_pre_lock_script():
    lengths, states = run_lengths(_markov(n=1_500, seed=4).to_numpy())
    for seed in range(15):
        ported = _arrange(lengths, states, np.random.default_rng(seed), MAX_TRIES)
        reference = _pre_lock(lengths, states, seed)
        assert (ported is None) == (reference is None)
        if ported is not None:
            np.testing.assert_array_equal(np.repeat(ported[1], ported[0]), reference)


def test_the_swap_repair_raises_where_it_cannot_reach_a_join_free_order():
    two_states = _from_episodes([i % 2 for i in range(30)])
    with pytest.raises(PlaceboError):
        matched_placebos(two_states, 3, method="swap")
    with pytest.raises(PlaceboError):
        matched_placebos(_dominant(), 3, method="swap")


def test_feasibility_walks_the_same_streams_as_the_draws():
    labels = _markov(n=1_800, seed=7)
    groups = pd.Series(np.repeat([1, 2], 900), index=labels.index)
    used = placebo_feasibility(labels, 12, seed=4, groups=groups)
    assert used.shape == (12, 2)
    assert (used > 0).all()
    placebo = matched_placebos(labels, 12, seed=4, groups=groups, method="swap")
    np.testing.assert_array_equal(placebo.tries, used.max(axis=1))


def test_feasibility_reports_failures_instead_of_raising():
    used = placebo_feasibility(_dominant(), 5, max_tries=5)
    assert used.shape == (5, 1)
    assert (used == 0).all()


# -------------------------------------------------------- the uniform construction


def _brute_force(counts: tuple[int, ...]) -> set[tuple[int, ...]]:
    letters = [i for i, c in enumerate(counts) for _ in range(c)]
    return {
        w for w in set(itertools.permutations(letters))
        if all(a != b for a, b in zip(w, w[1:], strict=False))
    }


@pytest.mark.parametrize(
    "counts",
    [(1,), (2,), (1, 1), (3, 2), (2, 2), (3, 1, 1), (3, 2, 2), (2, 2, 2), (4, 1, 1, 2),
     (0, 3, 0, 2), (5, 2, 2), (3, 3, 1, 2)],
)
def test_the_count_of_join_free_sequences_is_exact(counts):
    expected = len(_brute_force(counts))
    counted = np.exp(JoinFreeSequences(counts).log_count)
    assert round(float(counted)) == expected


def test_the_uniform_sampler_is_uniform():
    counts = (3, 2, 2)
    words = sorted(_brute_force(counts))
    sampler = JoinFreeSequences(counts)
    rng = np.random.default_rng(1)
    seen = Counter(tuple(sampler.sample(rng).tolist()) for _ in range(150 * 38))
    assert set(seen) <= set(words)
    assert set(seen) == set(words)
    assert chisquare([seen[w] for w in words]).pvalue > 0.001


def test_the_uniform_sampler_refuses_an_impossible_count():
    with pytest.raises(PlaceboError):
        JoinFreeSequences((3, 1)).sample(np.random.default_rng(0))


def test_the_uniform_construction_succeeds_where_the_swap_repair_fails():
    for labels in (_from_episodes([i % 2 for i in range(30)]), _dominant()):
        placebo = matched_placebos(labels, 30, seed=2, method="uniform")
        assert check_placebos(labels, placebo.draws).exact
        assert (placebo.tries == 1).all()
    draws = matched_placebos(_dominant(), 10, method="uniform").draws
    for column in draws.columns:
        _, states = run_lengths(draws[column].to_numpy())
        assert (states[::2] == 0).all()


def test_the_uniform_construction_permutes_lengths_within_each_state():
    labels = _markov(n=3_000, seed=9)
    draws = matched_placebos(labels, 200, seed=0, method="uniform").draws
    first = np.array([run_lengths(draws[c].to_numpy())[0][0] for c in draws.columns])
    assert len(np.unique(first)) > 5


# ---------------------------------------------------------------------- the check


def test_the_check_catches_a_broken_draw():
    labels = _markov(n=800, seed=3)
    draws = matched_placebos(labels, 4, method="uniform").draws
    joined = draws.copy()
    joined.iloc[:, 0] = np.sort(joined.iloc[:, 0].to_numpy())
    assert not check_placebos(labels, joined).exact
    shifted = draws.copy()
    shifted.iloc[0, 1] = (shifted.iloc[0, 1] + 1) % 4
    assert not check_placebos(labels, shifted).exact


# ----------------------------------------------------------------------- refusals


def test_bad_inputs_are_refused():
    labels = _markov(n=300)
    with pytest.raises(ValueError):
        matched_placebos(labels, 0)
    with pytest.raises(ValueError):
        matched_placebos(labels, 2, method="shuffle")
    with pytest.raises(ValueError):
        matched_placebos(labels.astype(float).where(labels.index != labels.index[5]), 2)
    interleaved = pd.Series(np.tile([1, 2], 150), index=labels.index)
    with pytest.raises(ValueError, match="contiguous"):
        matched_placebos(labels, 2, groups=interleaved)

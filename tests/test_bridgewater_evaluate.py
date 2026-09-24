"""The Bridgewater evaluation machinery must compute the draft's statistics, and nothing else.

Every test runs on synthetic data built here. Nothing reads `data/`, so no test needs
to skip for its absence. The properties tested:
- the statistic D and both balanced readings on cases solved by hand;
- a planted variance effect is detected against the P1 null, and a content-free one
  is not;
- P1 is exact on the quarterly stamps, and its null is neither an atom nor free of skew;
- every fitted quantity is causal, and the same seed gives the same bits;
- P2's rescalings do what they claim;
- the witness uses training cut-offs and a lag;
- the P3 map spaces have the sizes and ties the module docstring states;
- level C's null collapses to an atom when the target does not move, the Two Sigma
  C-1 trap.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from regime_lab.analysis.placebo import check_placebos, transitions
from regime_lab.construction import evaluate as E
from regime_lab.selection.folds import walk_forward_folds

#: The draft's quarterly occupancy (§6 P1), normalised.
OCCUPANCY = np.array([0.143, 0.308, 0.275, 0.275]) / 1.001
#: Annualised volatility by sleeve (rows, `E.SLEEVES`) and cell (columns, 2g + i).
PLANT = np.array([
    [0.40, 0.25, 0.14, 0.14],
    [0.05, 0.05, 0.08, 0.09],
    [0.04, 0.06, 0.04, 0.06],
    [0.20, 0.30, 0.18, 0.28],
    [0.22, 0.20, 0.14, 0.15],
])
DRAWS = 100


# ------------------------------------------------------------------------------ fixtures


def _world(seed: int, years: int = 22) -> tuple[pd.DatetimeIndex, pd.Series, pd.Series]:
    """Sessions, iid quarterly labels stamped on the 14th of Feb/May/Aug/Nov, and the path."""
    sessions = pd.bdate_range("2004-01-02", periods=years * 252)
    stamps = (pd.date_range(sessions[0] - pd.Timedelta(days=100), sessions[-1], freq="QS-FEB")
              + pd.Timedelta(days=13))
    rng = np.random.default_rng(seed)
    stamped = pd.Series(rng.choice(4, len(stamps), p=OCCUPANCY).astype(float), index=stamps,
                        name="quadrant")
    return sessions, stamped, E.expand_to_sessions(stamped, sessions, lag=1)


def _returns(sessions: pd.DatetimeIndex, labels: pd.Series, *, planted: bool,
             seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    codes = np.nan_to_num(labels.reindex(sessions).to_numpy(float), nan=1.0).astype(int)
    sigma = PLANT[:, codes].T if planted else np.broadcast_to(PLANT.mean(axis=1),
                                                                (len(sessions), 5))
    x = rng.standard_normal((len(sessions), 5)) * sigma / np.sqrt(252)
    return pd.DataFrame(x, index=sessions, columns=list(E.SLEEVES))


@pytest.fixture(scope="module")
def world():
    sessions, stamped, labels = _world(0)
    folds = walk_forward_folds(sessions, n_folds=5, test_years=3.0, anchor="end")
    return sessions, stamped, labels, folds


@pytest.fixture(scope="module")
def planted(world):
    sessions, _, labels, _ = world
    return _returns(sessions, labels, planted=True, seed=1)


@pytest.fixture(scope="module")
def flat(world):
    sessions, _, labels, _ = world
    return _returns(sessions, labels, planted=False, seed=1)


# --------------------------------------------------------------------------------- labels


def test_a_label_is_held_from_the_session_after_its_stamp():
    sessions = pd.bdate_range("2020-01-01", periods=12)
    # Thursday 2020-01-02 is a session; Saturday 2020-01-11 is not.
    stamped = pd.Series([1.0, 3.0], index=pd.DatetimeIndex(["2020-01-02", "2020-01-11"]))
    path = E.expand_to_sessions(stamped, sessions, lag=1)
    assert path.loc[:"2020-01-02"].isna().all()
    assert path.loc["2020-01-03"] == 1.0
    assert path.loc["2020-01-13"] == 1.0  # Monday: the Saturday stamp is known, not traded
    assert path.loc["2020-01-14"] == 3.0
    same_day = E.expand_to_sessions(stamped, sessions, lag=0)
    assert same_day.loc["2020-01-02"] == 1.0 and same_day.loc["2020-01-13"] == 3.0
    assert list(E.stamps_to_sessions(stamped.index, sessions, lag=1)) == [
        pd.Timestamp("2020-01-03"), pd.Timestamp("2020-01-14")]


def test_change_points_are_the_transition_stamps_and_a_subset_of_the_calendar(world):
    sessions, stamped, labels, _ = world
    in_force = E.stamps_in_force(stamped, sessions)
    moves = E.transition_stamps(in_force)
    assert set(moves) <= set(in_force.index)  # OC-C1: transition dates ⊂ stamps
    changed = labels.index[1:][(labels.to_numpy()[1:] != labels.to_numpy()[:-1])
                               & labels.notna().to_numpy()[:-1]]
    assert list(changed) == list(E.stamps_to_sessions(moves, sessions))
    assert in_force.index[0] <= sessions[0] < in_force.index[1]


def test_labels_must_be_integers_in_range():
    returns = pd.Series([0.01, -0.02, 0.0], index=pd.bdate_range("2020-01-01", periods=3))
    with pytest.raises(ValueError, match="integers"):
        E.d_statistic(returns, pd.Series([0.0, 4.0, 1.0], index=returns.index))
    with pytest.raises(ValueError, match="integers"):
        E.d_statistic(returns, pd.Series([0.0, 0.5, 1.0], index=returns.index))


# -------------------------------------------------------------------------------------- D


def test_d_by_hand_and_its_readings():
    index = pd.bdate_range("2020-01-01", periods=8)
    x = pd.Series([2.0, -2.0, 2.0, -2.0, 1.0, -1.0, 0.5, -0.5], index=index)
    k = pd.Series([0, 0, 0, 0, 1, 1, 2, 2], index=index, dtype=float)
    about_zero = {0: 4.0, 1: 1.0, 2: 0.25}
    assert E.d_statistic(x, k, about_zero=True) == pytest.approx(np.log(4.0 / 0.25))
    centred = E.cell_variances(x, k)["variance"]
    assert centred[0] == pytest.approx(16.0 / 3.0) and centred[2] == pytest.approx(0.5)
    assert E.d_statistic(x, k) == pytest.approx(np.log(centred[0] / centred[2]))
    assert E.d_statistic(x, k, about_zero=True, cells=(1, 0)) == pytest.approx(
        np.log(about_zero[1] / about_zero[0]))
    share = {c: n / 8 * about_zero[c] for c, n in ((0, 4), (1, 2), (2, 2))}
    assert E.d_statistic(x, k, about_zero=True, occupancy_weighted=True) == pytest.approx(
        np.log(max(share.values()) / min(share.values())))
    assert np.isnan(E.d_statistic(x, pd.Series(0.0, index=index)))


def test_d_is_scale_free_so_matching_mean_gross_cannot_move_it(world, planted):
    _, _, labels, _ = world
    book = planted.mean(axis=1)
    assert E.d_statistic(book, labels) == pytest.approx(E.d_statistic(3.7 * book, labels))


# --------------------------------------------------------------------- balanced weights


def _diagonal_cells() -> np.ndarray:
    """Four diagonal cells over three sleeves, between which a long-only book can balance."""
    return np.stack([np.diag(v) for v in ([4.0, 1.0, 1.0], [1.0, 4.0, 1.0],
                                          [1.0, 1.0, 4.0], [2.0, 2.0, 2.0])])


def test_equal_variance_is_reached_when_feasible():
    cells = _diagonal_cells()
    start = np.array([0.6, 0.3, 0.1])
    solution = E.balanced_weights(cells, start)
    w = solution.weights
    v = np.einsum("i,kij,j->k", w, cells, w)
    assert (w >= 0).all() and w.sum() == pytest.approx(1.0)
    assert v.max() / v.min() - 1.0 < 1e-4
    assert solution.dispersion < 1e-4
    np.testing.assert_allclose(w, 1.0 / 3.0, atol=1e-3)  # the only equaliser here


def test_equal_share_equalises_occupancy_weighted_variance():
    cells = np.stack([np.diag(v) for v in ([4.0, 1.0], [1.0, 4.0])])
    pi = np.array([0.25, 0.75])
    solution = E.balanced_weights(cells, np.array([0.5, 0.5]), offsets=-np.log(pi))
    v = np.einsum("i,kij,j->k", solution.weights, cells, solution.weights)
    assert (pi * v)[0] == pytest.approx((pi * v)[1], rel=1e-4)
    assert v[0] / v[1] == pytest.approx(3.0, rel=1e-3)  # R2 sets v_k ∝ 1/π_k (OC-A4)


def test_the_solver_is_deterministic_and_stays_on_a_balanced_start():
    cells = _diagonal_cells()
    a = E.balanced_weights(cells, np.array([0.2, 0.5, 0.3]))
    b = E.balanced_weights(cells, np.array([0.2, 0.5, 0.3]))
    assert np.array_equal(a.weights, b.weights)
    flat = np.stack([np.eye(3)] * 4)
    start = np.array([0.5, 0.3, 0.2])
    np.testing.assert_allclose(E.balanced_weights(flat, start).weights, start, atol=1e-8)


def test_fit_balanced_qualifies_cells_and_falls_back_below_two():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((400, 3))
    codes = np.r_[np.zeros(200, int), np.ones(200, int)]
    fit = E.fit_balanced(x, codes, np.ones(3))
    assert fit.fallback and fit.cells == ()  # two cells of one episode each
    assert list(fit.sessions[:2]) == [200, 200] and list(fit.episodes[:2]) == [1, 1]
    np.testing.assert_allclose(fit.weights, 1.0 / 3.0)
    striped = np.tile(np.repeat([0, 1], 25), 8)
    fit = E.fit_balanced(x, striped, np.ones(3), min_cell_sessions=63)
    assert not fit.fallback and fit.cells == (0, 1)


# --------------------------------------------------------------------------- level A


@pytest.mark.parametrize("scaling", ["none", "vol_target"])
def test_a_planted_variance_effect_is_detected(world, planted, scaling):
    _, stamped, labels, folds = world
    engine = E.LevelA(planted, folds, scaling=scaling)
    result = engine.run(labels)
    assert result.missing_sessions == 0 and result.fallback_folds == ()
    assert result.reduction > 0.1
    null = E.null_reductions(engine, stamped, DRAWS, seed=0)
    assert null.percentile(result.reduction) >= 0.95


@pytest.mark.parametrize("seed", [2, 3, 4])
def test_a_content_free_partition_is_not_detected(seed):
    sessions, stamped, labels = _world(seed)
    folds = walk_forward_folds(sessions, n_folds=5, test_years=3.0, anchor="end")
    engine = E.LevelA(_returns(sessions, labels, planted=False, seed=seed), folds)
    null = E.null_reductions(engine, stamped, DRAWS, seed=seed)
    assert null.percentile(engine.run(labels).reduction) < 0.95


def test_the_null_is_no_atom_sits_below_zero_and_is_skewed(world, planted):
    _, stamped, _, folds = world
    null = E.null_reductions(E.LevelA(planted, folds, scaling="none"), stamped, DRAWS)
    assert null.finite and not null.is_atom and null.n_distinct == DRAWS
    # A balanced leg fitted on content-free cells widens D out of sample on average.
    assert null.describe()["mean"] < 0 and null.skewness < 0
    assert null.mde_shift(0.05) > 0 and null.mde_sd(0.05) > 0
    assert null.mde_shift(E.sidak_alpha()) >= null.mde_shift(0.05)


def test_equal_share_is_built_to_widen_d_on_training(world, planted):
    """OC-A4: R2 aims at v_k ∝ 1/π_k, so its training D exceeds R1's on every fold.

    On this panel neither reading can equalise exactly with long-only weights (R1's
    training D stays near 0.3), which the lock must also expect of the real one.
    """
    _, _, labels, folds = world
    fits = {reading: E.LevelA(planted, folds, reading=reading, about_zero=True).fit(labels)
            for reading in ("equal_variance", "equal_share")}
    for f, fold in enumerate(folds):
        train = fold.train(planted.index)
        d = {reading: E.d_statistic(planted.loc[train] @ fit[f].weights, labels.loc[train],
                                    about_zero=True)
             for reading, fit in fits.items()}
        assert fits["equal_variance"][f].dispersion == pytest.approx(d["equal_variance"])
        assert d["equal_share"] > d["equal_variance"] + 0.2


def test_volatility_targeting_compresses_d(world, planted):
    _, _, labels, folds = world
    unscaled = E.LevelA(planted, folds, scaling="none").run(labels)
    targeted = E.LevelA(planted, folds, scaling="vol_target").run(labels)
    assert targeted.d_blind < unscaled.d_blind


def test_the_blind_leg_reads_no_label_and_equal_weights_give_zero(world, planted):
    sessions, _, labels, folds = world
    engine = E.LevelA(planted, folds)
    other = labels.sample(frac=1.0, random_state=0).set_axis(labels.index)
    a, b = engine.run(labels), engine.run(other)
    pd.testing.assert_frame_equal(a.blind_weights, b.blind_weights)
    pd.testing.assert_series_equal(a.blind_leg.returns, b.blind_leg.returns)
    same = engine.run(labels, weights=engine.blind_weights)
    assert same.reduction == 0.0 and same.fits is None


def test_blind_extremes_read_both_legs_on_the_blind_cells(world, planted):
    _, _, labels, folds = world
    result = E.LevelA(planted, folds, extremes="blind").run(labels)
    worst, best = result.cells
    variances = E.cell_variances(result.blind, result.test_labels)["variance"]
    assert variances.idxmax() == worst and variances.idxmin() == best
    balanced = E.cell_variances(result.balanced, result.test_labels)["variance"]
    assert result.d_balanced == pytest.approx(np.log(balanced[worst] / balanced[best]))


def test_fold_weights_are_causal(world, planted):
    sessions, _, labels, folds = world
    engine = E.LevelA(planted, folds)
    base = engine.run(labels)
    cut = folds[2].train(sessions)[-1]
    later = sessions > cut
    shocked = planted.copy()
    shocked.loc[later] = shocked.loc[later] * 5.0 + 0.01
    relabelled = labels.copy()
    relabelled.loc[later] = (relabelled.loc[later] + 1) % 4
    moved = E.LevelA(shocked, folds).run(relabelled)
    for name in ("blind_weights", "balanced_weights"):
        pd.testing.assert_frame_equal(getattr(base, name).iloc[:3], getattr(moved, name).iloc[:3])
        assert not getattr(base, name).iloc[3:].equals(getattr(moved, name).iloc[3:])
    early = sessions <= cut
    pd.testing.assert_series_equal(base.balanced_leg.returns[early],
                                   moved.balanced_leg.returns[early])


def test_level_a_is_deterministic(world, planted):
    _, _, labels, folds = world
    a = E.LevelA(planted, folds).run(labels)
    b = E.LevelA(planted, folds).run(labels)
    assert a.reduction == b.reduction
    pd.testing.assert_frame_equal(a.balanced_weights, b.balanced_weights)


def test_folds_that_train_on_their_test_window_are_refused(planted):
    class Leaky:
        number = 1

        def train(self, index):
            return index[:1000]

        def test(self, index):
            return index[900:1200]

    with pytest.raises(ValueError, match="trains on or after"):
        E.LevelA(planted, [Leaky()])


# ------------------------------------------------------------------------------------ P1


@pytest.mark.parametrize("by_fold", [False, True])
def test_p1_is_exact_on_the_quarterly_stamps(world, by_fold):
    sessions, stamped, _, folds = world
    stamps = E.stamps_in_force(stamped, sessions)
    blocks = E.calendar_blocks(stamps.index, sessions, folds) if by_fold else None
    placebo = E.p1_draws(stamps, 200, seed=0, blocks=blocks)
    assert check_placebos(stamps, placebo.draws, groups=blocks).exact
    assert (placebo.tries == 1).all()
    if by_fold:
        assert list(pd.unique(blocks)) == ["0:train", "1:test", "2:test", "3:test", "4:test",
                                           "5:test"]
    table = E.p1_feasibility(stamps, blocks)
    assert table["feasible"].all() and (table["log10_orders"] > 0).all()
    assert table["points"].sum() == len(stamps)


def test_p1_on_sessions_keeps_the_clock_and_nearly_the_occupancy(world):
    sessions, stamped, labels, _ = world
    stamps = E.stamps_in_force(stamped, sessions)
    placebo = E.p1_draws(stamps, 50, seed=1)
    expanded = E.expand_to_sessions(placebo.draws, sessions)
    real = transitions(labels.dropna().to_numpy())
    assert all(transitions(expanded[c].dropna().to_numpy()) == real for c in expanded)
    deviation = E.session_occupancy_deviation(labels, expanded)
    assert 0.0 < deviation < 0.02  # OC-P1b: quarters differ in session count


def test_the_null_is_seeded_and_reproducible(world, flat):
    _, stamped, _, folds = world
    engine = E.LevelA(flat, folds)
    a = E.null_reductions(engine, stamped, 20, seed=5)
    b = E.null_reductions(engine, stamped, 20, seed=5)
    c = E.null_reductions(engine, stamped, 20, seed=6)
    assert np.array_equal(a.values, b.values) and not np.array_equal(a.values, c.values)


def test_null_statistics_records_exposure_and_the_session_grid(world, planted):
    sessions, stamped, labels, folds = world
    engine = E.LevelA(planted, folds)
    test = engine.test_sessions
    nulls = E.null_statistics(
        engine, stamped, 20,
        statistics={
            "reduction": lambda r: r.reduction,
            "gross": lambda r: float(E.gross_exposure(r.balanced_leg.held).loc[test].mean()),
        },
    )
    assert set(nulls) == {"reduction", "gross"} and all(n.finite for n in nulls.values())
    daily = E.null_reductions(engine, labels, 10, stamped=False)
    assert daily.finite and daily.n == 10


def test_an_atom_is_flagged_and_its_shift_mde_refused():
    atom = E.NullDistribution(np.r_[np.zeros(90), np.linspace(0.1, 1.0, 10)])
    assert atom.is_atom and atom.atom_share == pytest.approx(0.9)
    assert np.isnan(atom.mde_shift())
    broken = E.NullDistribution(np.array([0.1, np.nan, 0.2]))
    assert not broken.finite and np.isnan(broken.percentile(0.15))


# ----------------------------------------------------------------------------- bootstrap


def test_the_bootstrap_is_paired_seeded_and_finite(world, planted):
    _, _, labels, folds = world
    result = E.LevelA(planted, folds).run(labels)
    same = E.reduction_bootstrap(result.blind, result.blind, result.test_labels,
                                 mean_block=63, draws=50)
    assert (same == 0.0).all()
    a = E.reduction_bootstrap(result.blind, result.balanced, result.test_labels,
                              mean_block=21, draws=50, seed=3)
    b = E.reduction_bootstrap(result.blind, result.balanced, result.test_labels,
                              mean_block=21, draws=50, seed=3)
    assert np.array_equal(a, b)
    se = E.reduction_se(result.blind, result.balanced, result.test_labels, draws=50)
    assert list(se.index) == [21, 63, 126] and (se > 0).all()


# ------------------------------------------------------------------------------------ P2


def test_ex_ante_volatility_is_the_lagged_rolling_covariance():
    rng = np.random.default_rng(0)
    index = pd.bdate_range("2020-01-01", periods=120)
    r = pd.DataFrame(rng.standard_normal((120, 3)) * 0.01, index=index, columns=list("abc"))
    w = pd.DataFrame(rng.random((120, 3)), index=index, columns=list("abc"))
    vol = E.ex_ante_volatility(w, r, window=20)
    t = 70
    cov = r.iloc[t - 20 : t].cov().to_numpy()
    assert vol.iloc[t] == pytest.approx(np.sqrt(w.iloc[t] @ cov @ w.iloc[t] * 252))
    assert vol.iloc[:20].isna().all() and vol.iloc[20:].notna().all()


def test_gross_matching_equalises_gross_and_a_constant_scale_changes_nothing(world, planted):
    sessions, _, labels, folds = world
    engine = E.LevelA(planted, folds)
    result = engine.run(labels)
    match = E.exposure_matched(result, planted, on="gross")
    test = engine.test_sessions
    matched = E.gross_exposure(result.balanced_leg.held).loc[test] * match.balanced_scale[test]
    np.testing.assert_allclose(matched, E.gross_exposure(result.blind_leg.held).loc[test])
    scaled = E.LevelA(planted, folds, scaling="none")
    twice = scaled.run(labels, weights=scaled.blind_weights.to_numpy())
    assert twice.reduction == 0.0
    assert E.exposure_matched(twice, planted, on="gross").reduction == pytest.approx(0.0)


def test_vol_matching_puts_both_legs_at_the_target(world, planted):
    _, _, labels, folds = world
    engine = E.LevelA(planted, folds)
    result = engine.run(labels)
    match = E.exposure_matched(result, planted, on="vol")
    test = engine.test_sessions
    for leg, scale in ((result.blind_leg, match.blind_scale),
                       (result.balanced_leg, match.balanced_scale)):
        rescaled = leg.held.mul(scale, axis=0)
        np.testing.assert_allclose(E.ex_ante_volatility(rescaled, planted).loc[test], 0.10)
    assert np.isfinite(match.reduction)


# ------------------------------------------------------------------------------- witness


def test_the_witness_uses_training_cutoffs_and_a_lag(world, planted):
    sessions, _, _, folds = world
    driver = planted["equity"]
    paths = E.volatility_witness_labels(driver, folds)
    fold = folds[0]
    train = fold.train(sessions)
    log_rv = E.trailing_log_vol(driver)
    occupancy = paths[fold.number].shift(-1).loc[train].dropna().value_counts(normalize=True)
    np.testing.assert_allclose(occupancy.sort_index(), 0.25, atol=0.01)
    later = driver.copy()
    later.loc[sessions > train[-1]] *= 3.0
    moved = E.volatility_witness_labels(later, folds)[fold.number]
    test = fold.test(sessions)
    first = test[0]
    assert moved.loc[first] == paths[fold.number].loc[first]  # bin of t−1, cut-offs unchanged
    assert (moved.loc[test[1:]] >= paths[fold.number].loc[test[1:]]).all()
    assert paths[fold.number].loc[first] == np.searchsorted(
        np.quantile(log_rv.loc[train].dropna(), [0.25, 0.5, 0.75]),
        log_rv.loc[train[-1]], side="right")


def test_the_witness_on_the_stamp_grid_holds_between_stamps(world, planted):
    sessions, stamped, _, folds = world
    grid = E.stamps_in_force(stamped, sessions).index
    path = E.volatility_witness_labels(planted["equity"], folds, grid=grid)[folds[0].number]
    changes = path.index[1:][(path.to_numpy()[1:] != path.to_numpy()[:-1])
                             & path.notna().to_numpy()[:-1]]
    assert set(changes) <= set(E.stamps_to_sessions(grid, sessions))


def test_the_witness_runs_through_level_a_and_only_downgrades(world, planted):
    _, _, labels, folds = world
    engine = E.LevelA(planted, folds)
    witness = engine.run(E.volatility_witness_labels(planted["equity"], folds))
    assert np.isfinite(witness.reduction)
    assert E.witness_dominates(0.2, 0.3) and not E.witness_dominates(0.3, 0.2)
    assert not E.witness_dominates(0.3, float("nan"))


# ------------------------------------------------------------------------------ B2 / P3


def test_the_draft_map_budgets():
    budget = E.map_risk_budgets(E.DRAFT_MAP)
    np.testing.assert_allclose(budget.to_numpy(), [1 / 8, 3 / 8, 1 / 12, 5 / 24, 5 / 24])
    with pytest.raises(ValueError, match="no sleeve carries"):
        E.map_risk_budgets({**E.DRAFT_MAP, "duration": frozenset({"g-"})})
    kept = E.map_risk_budgets({**E.DRAFT_MAP, "duration": frozenset({"g-"})},
                              empty="renormalise")
    assert kept.sum() == pytest.approx(1.0)


def test_budgets_ignore_the_signs_of_the_map():
    """OC-B2b: flipping every sign gives the same budgets, so the same B2 book."""
    flip = {"g+": "g-", "g-": "g+", "i+": "i-", "i-": "i+"}
    flipped = {s: frozenset(flip[t] for t in tags) for s, tags in E.DRAFT_MAP.items()}
    pd.testing.assert_series_equal(E.map_risk_budgets(E.DRAFT_MAP),
                                   E.map_risk_budgets(flipped))


@pytest.mark.parametrize(("space", "covered", "every"), [
    ("permutation", 120, 120),
    ("profile", 752, 1_024),
    ("axis_consistent", 21_300, 32_768),
])
def test_map_space_sizes_enumerated(space, covered, every):
    assert sum(1 for _ in E.enumerate_maps(space)) == covered == E.map_space_size(space)
    assert (sum(1 for _ in E.enumerate_maps(space, require_all_tags=False)) == every
            == E.map_space_size(space, require_all_tags=False))


def test_map_space_sizes_by_formula():
    assert E.map_space_size("any_subset", require_all_tags=False) == 15 ** 5
    assert E.map_space_size("any_subset") == sum(
        (-1) ** j * int(np.prod([1])) * len(list(itertools.combinations(range(4), j)))
        * (2 ** (4 - j) - 1) ** 5 for j in range(5))
    assert len(E.axis_consistent_sets()) == 8


def test_p3_resolution_in_the_small_spaces():
    permutation = E.map_space_summary("permutation")
    assert (permutation.maps, permutation.budget_vectors, permutation.reference_ties) == (
        120, 60, 2)
    assert permutation.min_p_value == pytest.approx(2 / 120)
    assert permutation.min_p_value > E.sidak_alpha()  # OC-B2c: unreachable at Šidák α
    assert permutation.max_percentile >= 0.95
    profile = E.map_space_summary("profile")
    assert (profile.maps, profile.budget_vectors, profile.reference_ties) == (752, 61, 32)


def test_budget_weight_rules():
    moment = np.diag([0.04, 0.01, 0.09])
    b = np.array([0.5, 0.5, 0.0])
    np.testing.assert_allclose(E.budget_weights(b, moment), [1 / 3, 2 / 3, 0.0])
    np.testing.assert_allclose(E.budget_weights(b, moment, rule="notional"), [0.5, 0.5, 0.0])
    np.testing.assert_allclose(E.budget_weights(b, moment, rule="erc"), [1 / 3, 2 / 3, 0.0],
                               atol=1e-8)


def test_p3_runs_each_book_once_and_ties_equal_maps(world, planted):
    _, _, labels, folds = world
    engine = E.LevelA(planted, folds)
    maps = list(E.enumerate_maps("permutation"))[:12]
    reductions = E.p3_reductions(engine, labels, maps)
    assert reductions.shape == (12,) and np.isfinite(reductions).all()
    keys = [tuple(E.map_risk_budgets(m).round(12)) for m in maps]
    for i, j in itertools.combinations(range(12), 2):
        if keys[i] == keys[j]:
            assert reductions[i] == reductions[j]
    weights = E.b2_fold_weights(engine, E.DRAFT_MAP)
    assert np.allclose(weights.sum(axis=1), 1.0) and (weights > 0).all()


# ------------------------------------------------------------------------------- level C


def test_a_cadence_book_holds_between_rebalances():
    index = pd.bdate_range("2020-01-01", periods=10)
    target = pd.DataFrame({"a": np.linspace(0.1, 1.0, 10), "b": 1.0}, index=index)
    target.iloc[0] = np.nan
    held = E.cadence_weights(target, pd.DatetimeIndex([index[4], index[7]]))
    assert held.iloc[0].isna().all()
    assert (held["a"].iloc[1:4] == target["a"].iloc[1]).all()
    assert (held["a"].iloc[4:7] == target["a"].iloc[4]).all()
    assert (held["a"].iloc[7:] == target["a"].iloc[7]).all()


def _target(sessions: pd.DatetimeIndex, returns: pd.DataFrame, moving: bool) -> pd.DataFrame:
    if not moving:
        return pd.DataFrame(0.2, index=sessions, columns=returns.columns)
    inverse = 1.0 / returns.rolling(63).std().shift(1)
    return inverse.div(inverse.sum(axis=1), axis=0)


def test_level_c_null_is_an_atom_when_the_target_does_not_move(world, flat):
    """OC-C2, the Two Sigma C-1 trap: with a static target every cadence is the same book."""
    sessions, stamped, _, folds = world
    stamps = E.stamps_in_force(stamped, sessions)
    calendar = E.stamps_to_sessions(stamps.index, sessions)
    placebos = E.placebo_rebalances(E.p1_draws(stamps, 20).draws, sessions)
    test = sessions[sessions > folds[0].train(sessions)[-1]]
    null = E.level_c_null(_target(sessions, flat, False), flat, placebos, calendar,
                          bps=1.0, evaluate=test)
    assert null.is_atom and null.atom_share == 1.0


def test_level_c_placebos_keep_the_clock_and_the_statistics_move(world, flat):
    sessions, stamped, _, folds = world
    stamps = E.stamps_in_force(stamped, sessions)
    real = E.stamps_to_sessions(E.transition_stamps(stamps), sessions)
    placebos = E.placebo_rebalances(E.p1_draws(stamps, 20).draws, sessions)
    assert all(len(p) == len(real) for p in placebos)
    calendar = E.stamps_to_sessions(stamps.index, sessions)
    test = sessions[sessions > folds[0].train(sessions)[-1]]
    target = _target(sessions, flat, True)
    comparison = E.level_c_compare(target, flat, real, calendar, bps=1.0, evaluate=test)
    assert comparison.turnover_saving > 0 and comparison.tracking_difference > 0
    assert comparison.rebalances_per_year_conditional < comparison.rebalances_per_year_calendar
    for name in ("sharpe_difference", "turnover_saving", "tracking_difference"):
        null = E.level_c_null(target, flat, placebos, calendar, bps=1.0, evaluate=test,
                              statistic=name)
        assert null.finite and not null.is_atom
    targeted = E.level_c_compare(target, flat, real, calendar, bps=1.0, evaluate=test,
                                 vol_target=True)
    assert np.isfinite(targeted.sharpe_difference)


# ------------------------------------------------------------------ the sibling modules


def test_the_sleeve_names_match_the_sleeve_module():
    sleeves = pytest.importorskip("regime_lab.construction.sleeves")
    assert tuple(sleeves.HEADLINE_SLEEVES) == E.SLEEVES


def test_the_expansion_matches_the_quadrant_module():
    quadrant = pytest.importorskip("regime_lab.construction.quadrant")
    sessions, stamped, labels = _world(9, years=6)
    table = pd.DataFrame({"label": stamped.astype("int64").to_numpy(),
                          "available_at": stamped.index},
                         index=pd.period_range("2003Q3", periods=len(stamped), freq="Q"))
    table.index.name = "quarter"
    theirs = quadrant.daily_labels(table, sessions).astype(float)
    pd.testing.assert_series_equal(theirs, labels, check_names=False, check_index_type=False,
                                   check_freq=False)


# -------------------------------------------------------------------------- power sanity


def test_the_draft_mde_is_one_log_sigma():
    assert E.draft_mde_log_sigma(5938) == pytest.approx(0.204, abs=5e-4)
    two_cells = E.analytic_mde_log_variance_difference(1792, 841)
    assert two_cells > 5 * E.DRAFT_MDE
    assert E.sidak_alpha() == pytest.approx(0.01021, abs=1e-5)

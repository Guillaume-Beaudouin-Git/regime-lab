"""The Bridgewater lock's inference: guard, quarterly P1, bars, verdict power, C and planting.

Synthetic data only; nothing reads ``data/``. What is tested:
- the guard refuses the real path, a relabelled real path, a near-real variant and a
  per-fold mapping of the real path, and lets content-free draws through;
- P1 on quarters is exact within each block, applies the availability masks, and draw
  ``d`` does not depend on the number of draws;
- the fixed-weight null holds the given leg;
- the bar, the p-values and the power of the verdict rule, which is below the power of
  the percentile line whenever the bar sits above the null's 1 − α quantile;
- the common-scale witness, the entropy rank, the planting of a variance effect;
- level C's conditioned draws and its placebo task.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regime_lab.analysis.placebo import run_lengths
from regime_lab.construction import declared as D
from regime_lab.construction import evaluate as E
from regime_lab.construction import inference as I
from regime_lab.construction import sleeves as S


@pytest.fixture(scope="module")
def world():
    sessions = pd.bdate_range("2004-01-02", "2019-12-31")
    quarters = pd.period_range("2004Q1", "2019Q4", freq="Q")
    rng = np.random.default_rng(0)
    labels = pd.Series(rng.choice(4, len(quarters)).astype(float), index=quarters)
    available = pd.Series(quarters.end_time.normalize() + pd.Timedelta(days=45), index=quarters)
    partition = D.Partition("q", labels, available)
    folds = D.quarter_folds(labels, available, sessions, min_quarters=3)
    returns = pd.DataFrame(rng.standard_normal((len(sessions), 5)) * 0.01, index=sessions,
                           columns=list(E.SLEEVES))
    return sessions, partition, folds, returns


# ------------------------------------------------------------------------ guard


def test_best_relabelling_agreement():
    a = np.array([0, 1, 2, 3, 0, 1, -1, 2])
    b = np.array([3, 2, 1, 0, 3, 2, 1, -1])
    assert I.best_relabel_agreement(a, b) == 1.0
    c = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    assert I.best_relabel_agreement(c, c[::-1]) == 1.0
    assert I.best_relabel_agreement(np.array([0, 1]), np.array([-1, -1])) != \
        I.best_relabel_agreement(np.array([0, 1]), np.array([-1, -1]))  # NaN: no overlap
    assert np.isnan(I.best_relabel_agreement(a, b, min_overlap=100))


def test_the_guard_refuses_real_paths_and_their_variants(world):
    sessions, partition, folds, returns = world
    engine = I.GuardedLevelA(returns, folds)
    real = D.fold_paths(partition.labels, partition.available, sessions, folds)
    engine.forbid(real)
    with pytest.raises(I.GuardRefused):
        engine.run(real)
    relabelled = {k: v.map({0.0: 2.0, 1.0: 3.0, 2.0: 0.0, 3.0: 1.0}) for k, v in real.items()}
    with pytest.raises(I.GuardRefused):
        engine.run(relabelled)
    whole = partition.path(sessions)
    with pytest.raises(I.GuardRefused):
        engine.run(whole)  # the unmasked real path, one path for every fold
    with pytest.raises(I.GuardRefused):
        engine.run(whole.shift(3))  # a near-real variant: another lag
    for paths in I.placebo_fold_paths(partition, sessions, folds, 20, seed=7):
        assert np.isfinite(engine.run(paths).reduction)
    assert engine.runs == 20 and engine.closest_seen < I.GUARD_AGREEMENT


# ----------------------------------------------------------------- P1 on quarters


def test_quarterly_p1_is_exact_per_block_and_masked(world):
    sessions, partition, folds, _ = world
    blocks = D.quarter_blocks(partition.labels.index, sessions, folds)
    draws = I.quarter_draws(partition.labels, 30, seed=11, blocks=blocks)
    for name in pd.unique(blocks):
        mine = (blocks == name).to_numpy()
        real = partition.labels.to_numpy()[mine]
        lengths, states = run_lengths(real)
        want = sorted(zip(lengths.tolist(), np.asarray(states).tolist(), strict=True))
        for d in draws.columns:
            got_l, got_s = run_lengths(draws[d].to_numpy()[mine])
            assert sorted(zip(got_l.tolist(), np.asarray(got_s).tolist(), strict=True)) == want
    fewer = I.quarter_draws(partition.labels, 5, seed=11, blocks=blocks)
    pd.testing.assert_frame_equal(fewer, draws.iloc[:, :5])
    maker = I.FoldPathMaker(partition.labels.index, partition.available, sessions, folds)
    paths = maker(draws[0].to_numpy())
    masks = D.fold_masks(partition.available, sessions, folds)
    for fold in folds:
        assert paths[fold.number][~masks[fold.number]].isna().all()
        assert paths[fold.number][masks[fold.number]].notna().all()


def test_the_fixed_weight_null_holds_the_given_leg(world):
    sessions, partition, folds, returns = world
    engine = E.LevelA(returns, folds)
    weights = np.tile([0.1, 0.5, 0.1, 0.2, 0.1], (len(folds), 1))
    seen = []

    def record(result: E.LevelAResult) -> float:
        seen.append(result.balanced_weights.to_numpy())
        return result.reduction

    null = I.null_statistics(engine, partition, 4, statistics={"r": record}, seed=1,
                             weights=weights)
    assert null["r"].finite and len(seen) == 4
    assert all(np.array_equal(w, weights) for w in seen)
    refit = I.null_statistics(engine, partition, 4, statistics={"r": lambda r: r.reduction},
                              seed=1)
    assert not np.array_equal(refit["r"].values, null["r"].values)


# ------------------------------------------------------------- bars and power


def test_the_bar_takes_the_largest_convention_and_respects_the_floor():
    null = E.NullDistribution(np.random.default_rng(0).normal(0.0, 0.15, 4000))
    alpha = E.sidak_alpha()
    bar = I.bar(null, alpha)
    assert bar == pytest.approx(max(0.204, null.mde_shift(alpha), null.mde_sd(alpha)))
    assert I.bar(E.NullDistribution(np.zeros(100)), alpha, floor=0.3) == 0.3  # atom
    assert np.isnan(I.bar(E.NullDistribution(np.array([0.0, np.nan])), alpha))


def test_verdict_power_is_the_joint_rule_and_below_the_percentile_line():
    null = np.random.default_rng(1).normal(0.0, 1.0, 4000)
    alpha = 0.01
    np.testing.assert_allclose(I.p_values(null, [10.0, -10.0]), [1 / 4001, 1.0])
    q = np.quantile(null, 1 - alpha)
    t = q + 1.0  # a bar above the percentile line, as T_A sits above q(1 − α_S)
    assert I.verdict_power(null, t, t, alpha) == pytest.approx(0.5, abs=0.03)
    assert I.verdict_power(null, t, t, alpha) < I.percentile_power(null, t, alpha)
    powers = [I.verdict_power(null, s, t, alpha) for s in (0.0, 1.0, 2.0, 4.0, 8.0)]
    assert powers[0] <= alpha and all(b >= a for a, b in zip(powers, powers[1:], strict=False))
    d80 = I.shift_for_power(null, t, alpha)
    assert I.verdict_power(null, d80, t, alpha) >= 0.80
    assert I.verdict_power(null, d80 - 0.05, t, alpha) < 0.80
    assert d80 == pytest.approx(t - np.quantile(null, 0.20), abs=0.05)
    assert I.verdict_power(null, t, t, alpha, se=100.0) == 0.0  # the bootstrap line binds


def test_the_common_scale_and_the_effective_bar():
    null = np.random.default_rng(2).normal(0.1, 0.2, 2000)
    z = I.z_score(0.5, null)
    assert z == pytest.approx((0.5 - null.mean()) / null.std(ddof=1))
    assert np.isnan(I.z_score(np.nan, null))
    assert I.effective_bar(null, 0.3, [z]) == pytest.approx(0.5)
    assert I.effective_bar(null, 0.9, [z, -1.0]) == 0.9
    assert I.effective_bar(null, 0.3, [np.nan]) == 0.3


def test_the_entropy_rank():
    assert I.entropy_rank(np.eye(5)) == pytest.approx(5.0)
    assert I.entropy_rank(np.ones((4, 4))) == pytest.approx(1.0)
    corr = np.full((3, 3), 0.5) + np.eye(3) * 0.5
    assert 1.0 < I.entropy_rank(corr) < 3.0


def test_planting_scales_the_variance_of_one_cell_only():
    index = pd.bdate_range("2020-01-01", periods=4000)
    rng = np.random.default_rng(3)
    excess = pd.DataFrame(rng.standard_normal((4000, 2)), index=index, columns=["a", "b"])
    path = pd.Series(np.tile([0.0, 1.0], 2000), index=index)
    planted = I.plant_variance(excess, path, {1: {"a": 4.0}})
    in_cell = path == 1.0
    np.testing.assert_allclose(planted.loc[in_cell, "a"], 2.0 * excess.loc[in_cell, "a"])
    pd.testing.assert_series_equal(planted.loc[~in_cell, "a"], excess.loc[~in_cell, "a"])
    pd.testing.assert_series_equal(planted["b"], excess["b"])
    with pytest.raises(ValueError, match="positive"):
        I.plant_variance(excess, path, {1: {"a": 0.0}})


# ------------------------------------------------------------------------ level C


TOY = {"equity": ("^A",), "duration": ("B",), "inflation_linked": ("C",),
       "commodity": ("D",), "precious": ("E",)}


def test_conditioned_draws_keep_the_transition_count():
    sessions = pd.bdate_range("2004-01-02", "2014-12-31")
    stamps_index = pd.date_range("2003-11-14", "2014-11-14", freq="QS-FEB") + pd.Timedelta(days=13)
    rng = np.random.default_rng(4)
    stamps = pd.Series(rng.choice(4, len(stamps_index)).astype(float), index=stamps_index)
    folds = E.stamp_folds(stamps, sessions, min_quarters=2)
    blocks = E.calendar_blocks(stamps.index, sessions, folds)
    test = pd.DatetimeIndex(np.concatenate([f.test(sessions).to_numpy() for f in folds]))
    real = I.transitions_between(stamps, sessions[0], sessions[-1])
    target = I.count_in_window(real, sessions, test)
    draws, ids = I.conditioned_draws(stamps, 10, seed=5, blocks=blocks, sessions=sessions,
                                     window=test, target=target)
    for d in draws.columns:
        moves = I.transitions_between(draws[d], sessions[0], sessions[-1])
        assert I.count_in_window(moves, sessions, test) == target
    again, ids2 = I.conditioned_draws(stamps, 5, seed=5, blocks=blocks, sessions=sessions,
                                      window=test, target=target)
    np.testing.assert_array_equal(ids2, ids[:5])
    with pytest.raises(RuntimeError, match="match"):
        I.conditioned_draws(stamps, 10, seed=5, blocks=blocks, sessions=sessions, window=test,
                            target=10_000, max_draws=50)


def test_the_c_placebo_task_is_a_sharpe_difference_against_the_calendar():
    sessions = pd.bdate_range("2004-01-02", "2012-12-31")
    rng = np.random.default_rng(6)
    excess = pd.DataFrame(rng.normal(0.0002, 0.01, (len(sessions), 5)), index=sessions,
                          columns=["^A", "B", "C", "D", "E"])
    calendar = pd.date_range("2005-02-15", "2012-11-15", freq="QS-FEB") + pd.Timedelta(days=14)
    calendar = pd.DatetimeIndex([d for d in calendar if d in sessions])
    test = sessions[sessions >= "2008-01-01"]
    book, _, _ = S.blind_book(excess, TOY, list(calendar))
    cal_sr = I.sharpe(S.net_returns(book).reindex(test))
    subset = calendar[::2]
    setup = I.CSetup(excess, TOY, (subset, calendar), test, cal_sr)
    value = I.c_placebo_task(setup, 0)
    other, _, _ = S.blind_book(excess, TOY, list(subset))
    assert value == pytest.approx(I.sharpe(S.net_returns(other).reindex(test)) - cal_sr)
    assert I.c_placebo_task(setup, 1) == pytest.approx(0.0)

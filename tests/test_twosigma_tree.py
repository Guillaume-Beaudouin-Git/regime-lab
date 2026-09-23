"""The shared core of the Two Sigma instruments must be the lock's §12, and nothing else.

Every test runs on synthetic data built by `tree.synthetic_tree_data`, which never reads
`data/`. The properties tested are the ones a reading would silently depend on: the lag
never crosses a fold, a placebo draw qualifies exactly the cells the real path does, a
book is read on the test sessions only with turnover restricted first, a NaN never reads
as a verdict, Holm's family never shrinks, and the reading refuses to run unless its
thresholds are committed, unmodified and bitwise equal. The one test that reads the real
store checks shapes and dates only, and skips when the files are absent.
"""

from __future__ import annotations

import dataclasses
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from regime_lab.analysis.placebo import run_lengths
from regime_lab.config import ROOT
from regime_lab.selection import protocol
from regime_lab.selection import tree as T
from regime_lab.selection.context import CONTEXT_FEATURES, qualifying_cells, trailing_mode
from regime_lab.selection.library import LIBRARY


@pytest.fixture(scope="module")
def data() -> T.TreeData:
    return T.synthetic_tree_data(seed=0)


@pytest.fixture(scope="module")
def primary(data: T.TreeData) -> T.Partition:
    return T.partition_paths(data, T.PRIMARY)


@pytest.fixture(scope="module")
def smoothed(data: T.TreeData) -> pd.Series:
    return T.partition_paths(data, T.SMOOTH21).paths


# --------------------------------------------------------------------------- the data


def test_synthetic_data_has_the_shape_of_the_tree(data):
    assert list(data.signals) == list(LIBRARY)
    assert list(data.legs.columns) == list(LIBRARY)
    assert data.legs.index.equals(data.sessions)
    assert len(data.folds) == 5
    union = pd.DatetimeIndex(np.concatenate([f.test(data.sessions) for f in data.folds]))
    assert data.test_sessions.equals(union)
    assert data.test_sessions[0] > data.folds[0].train_end
    assert np.isfinite(data.excess.to_numpy()).all()
    assert np.isfinite(data.legs.to_numpy()).all()
    assert data.log_rv.index.equals(data.sessions)
    assert list(data.features.columns) == list(CONTEXT_FEATURES)
    assert "TreeData(" in repr(data) and "sessions" in repr(data)


def test_excess_is_taken_at_the_instrument_level(data):
    expected = data.returns.sub(data.rf, axis=0)
    pd.testing.assert_frame_equal(data.excess, expected)


def test_a_leg_is_its_signal_alone_net_of_5_bp_on_its_own_turnover(data):
    name = "IND_REV_1M"
    w = data.signals[name].to_numpy()
    gross = (w * data.excess.to_numpy()).sum(axis=1)
    traded = np.r_[0.0, np.abs(np.diff(w, axis=0)).sum(axis=1)]
    expected = gross - 0.0005 * traded
    np.testing.assert_allclose(data.legs[name].to_numpy(), expected, rtol=0, atol=1e-15)


def test_build_refuses_a_session_on_which_a_signal_is_missing():
    inputs = T.synthetic_inputs(seed=1)
    built = T.build_tree_data(**inputs, n_folds=5, test_years=1.0)
    early = inputs["industries"].index[inputs["industries"].index < built.sessions[0]][-1]
    with pytest.raises(ValueError, match="not complete"):
        T.build_tree_data(**inputs, sessions=built.sessions.union([early]))


# ----------------------------------------------------------------------------- variants


def test_the_variants_are_the_declared_ones():
    assert (T.PRIMARY.K, T.PRIMARY.d, T.PRIMARY.smoothing) == (4, 0.5, None)
    assert T.PRIMARY.features == CONTEXT_FEATURES and len(CONTEXT_FEATURES) == 20
    assert [v.K for v in (T.K3, T.K5, T.K6)] == [3, 5, 6]
    assert T.SMOOTH21.smoothing == 21 and T.SMOOTH21.K == 4
    assert (T.D025.d, T.D100.d) == (0.25, 1.00)
    assert T.D025.levels == T.D100.levels == ("A-1",)
    assert T.D025.partition_key == T.D100.partition_key == T.PRIMARY.partition_key
    assert len(T.PIT18.features) == 18
    assert not set(T.PIT18.features) & {"fin_nfci", "fin_nfci_chg13w"}
    assert T.PIT18.role == "control"
    assert T.PRIMARY.cell_floor == 10 and T.K3.cell_floor == 7.5 and T.K6.cell_floor == 15
    assert len(T.DECLARED_ROWS) == len(set(T.DECLARED_ROWS)) == 20
    assert [v.name for v in T.SENSITIVITY_VARIANTS] == [
        "K=3", "K=5", "K=6", "d=0.25", "d=1.00", "smooth21"]


# -------------------------------------------------------------------------------- paths


def test_partition_paths_are_stamped_session_paths(data, primary):
    paths = primary.paths
    assert list(paths.index.names) == ["fold", "segment", "session"]
    assert paths.dtype == np.int64
    for fold in data.folds:
        train = paths.xs((fold.number, "train"), level=("fold", "segment")).index
        test = paths.xs((fold.number, "test"), level=("fold", "segment")).index
        assert train.equals(fold.train(data.sessions))
        assert test.equals(fold.test(data.sessions))
        model = primary.wf.model(fold.number)
        last_train = fold.train(data.sessions)[-1]
        assert paths.loc[(fold.number, "train", last_train)] == model.train_labels.loc[last_train]


def test_smoothing_is_the_trailing_mode_with_20_unlabelled_sessions_per_fold(primary, smoothed):
    pd.testing.assert_series_equal(smoothed, trailing_mode(primary.paths, 21))
    for _, path in smoothed.groupby(level="fold"):
        values = path.to_numpy()
        assert np.isnan(values[:20]).all() and np.isfinite(values[20:]).all()


def test_the_pit_rebuild_fits_the_18_features(data):
    part = T.partition_paths(data, T.PIT18)
    assert all(model.columns == T.PIT_FEATURES for model in part.wf.models)


def test_quantile_bins_are_right_closed_on_the_reference_quantiles():
    reference = pd.Series(np.arange(1.0, 9.0))  # quartile edges 2.75, 4.5, 6.25
    values = pd.Series([1.0, 2.75, 2.76, 4.5, 6.25, 6.3, np.nan, 100.0])
    bins = T.quantile_bins(values, reference, 4)
    np.testing.assert_array_equal(bins.to_numpy(), [0, 0, 1, 1, 2, 3, np.nan, 3])


def test_w1_bins_log_rv_at_each_fold_training_quartiles(data):
    paths = T.witness_paths(data, "W1", 4)
    assert list(paths.index.names) == ["fold", "segment", "session"]
    fold = data.folds[1]
    train = fold.train(data.sessions)
    edges = np.quantile(data.log_rv.loc[train].to_numpy(), [0.25, 0.5, 0.75])
    for segment, dates in (("train", train), ("test", fold.test(data.sessions))):
        got = paths.xs((fold.number, segment), level=("fold", "segment")).to_numpy()
        expected = np.searchsorted(edges, data.log_rv.loc[dates].to_numpy(), side="left")
        np.testing.assert_array_equal(got, expected)
    occupancy = np.bincount(paths.xs((fold.number, "train"), level=("fold", "segment")),
                            minlength=4) / len(train)
    assert np.abs(occupancy - 0.25).max() < 0.01


def test_w2_is_the_log_trailing_second_moment_of_the_equal_weight_leg(data):
    values = T.witness_values(data, "W2")
    blend_leg = data.legs.mean(axis=1)
    expected = np.log((blend_leg**2).rolling(63).mean())
    pd.testing.assert_series_equal(values, expected.rename("W2"))
    paths = T.witness_paths(data, "W2", 4)
    for fold in data.folds:
        train = paths.xs((fold.number, "train"), level=("fold", "segment"))
        assert int(train.isna().sum()) == 62 and train.iloc[:62].isna().all()
        assert paths.xs((fold.number, "test"), level=("fold", "segment")).notna().all()


def test_the_lag_is_one_session_inside_a_fold_and_never_crosses_folds(data, primary):
    lagged = T.lagged_paths(primary.paths)
    for fold in data.folds:
        path = primary.paths.xs(fold.number, level="fold").droplevel("segment")
        one = T.lagged_path(primary.paths, fold.number)
        assert np.isnan(one.iloc[0])
        np.testing.assert_array_equal(one.to_numpy()[1:], path.to_numpy()[:-1])
        first_test, last_train = fold.test(data.sessions)[0], fold.train(data.sessions)[-1]
        assert one.loc[first_test] == path.loc[last_train]
        np.testing.assert_array_equal(lagged.xs(fold.number, level="fold").to_numpy(),
                                      one.to_numpy())


def test_cells_wrap_qualifying_cells_and_count_the_abstaining_sessions(primary):
    c = T.cells(primary.paths)
    table = qualifying_cells(primary.paths)
    pd.testing.assert_frame_equal(c.table, table)
    q = table["qualifies"]
    assert c.n_train == int(q.xs("train", level="segment").sum())
    assert c.n_test == int(q.xs("test", level="segment").sum())
    both = [(f, k) for f in c.train for k in c.train[f] if k in c.test[f]]
    assert list(c.pairs) == both
    lagged = T.lagged_paths(primary.paths)
    count = 0
    for f in c.train:
        held = lagged.xs((f, "test"), level=("fold", "segment")).to_numpy()
        count += sum(not (np.isfinite(h) and int(h) in c.train[f]) for h in held)
    assert c.abstaining == count
    assert len(c.per_fold.split("/")) == 5


# ------------------------------------------------------------------------------ placebo


def _block_counts(paths: pd.Series) -> dict:
    out = {}
    for key, block in paths.dropna().groupby(level=["fold", "segment"]):
        lengths, states = run_lengths(block.to_numpy())
        out[key] = sorted(zip(states.tolist(), lengths.tolist(), strict=True))
    return out


def test_placebo_draws_are_uniform_exact_and_keep_every_block(primary):
    draws = T.placebo_draws(primary.paths, n=20)
    assert draws.method == "uniform" and draws.exact and draws.n == 20
    real = _block_counts(primary.paths)
    for j in (0, 7, 19):
        drawn = draws.draw(j)
        assert drawn.index.equals(primary.paths.index) and drawn.dtype == np.int64
        assert _block_counts(drawn) == real
    assert not draws.draw(0).equals(primary.paths)


def test_a_placebo_draw_qualifies_exactly_the_cells_of_the_real_path(primary):
    draws = T.placebo_draws(primary.paths, n=5)
    real = T.cells(primary.paths)
    for j in range(5):
        drawn = T.cells(draws.draw(j))
        pd.testing.assert_frame_equal(drawn.table, real.table)


def test_smoothed_unlabelled_sessions_are_left_out_of_the_placebo(smoothed):
    draws = T.placebo_draws(smoothed, n=10)
    assert draws.exact
    for j in range(10):
        drawn = draws.draw(j)
        np.testing.assert_array_equal(drawn.isna().to_numpy(), smoothed.isna().to_numpy())
        assert _block_counts(drawn) == _block_counts(smoothed)


def test_the_placebo_refuses_an_unlabelled_session_inside_a_block(primary):
    broken = primary.paths.astype(float).copy()
    broken.iloc[100] = np.nan
    with pytest.raises(ValueError, match="inside the block"):
        T.placebo_draws(broken, n=2)


def test_a_draw_depends_on_the_seed_and_its_index_only(primary):
    few, more = T.placebo_draws(primary.paths, n=3), T.placebo_draws(primary.paths, n=6)
    pd.testing.assert_series_equal(few.draw(2), more.draw(2))
    other = T.placebo_draws(primary.paths, n=3, seed=1)
    assert not few.draw(2).equals(other.draw(2))


# -------------------------------------------------------------------------------- books


def _one_hot_tables(K: int, data: T.TreeData) -> dict[int, pd.DataFrame]:
    names = list(LIBRARY)
    table = pd.DataFrame(0.0, index=range(K), columns=names)
    for k in range(K):
        table.iloc[k, k] = 1.0
    return {f.number: table for f in data.folds}


def test_the_traded_mix_is_equal_weight_until_the_first_test_session(data, primary):
    traded = T.traded_mix(data.sessions, data.folds, _one_hot_tables(4, data), primary.paths)
    pre = traded.mix.loc[data.sessions < data.test_sessions[0]]
    assert np.allclose(pre.to_numpy(), 0.1)
    assert traded.fallback.index.equals(data.test_sessions)


def test_each_fold_trades_its_own_path_lagged_one_session(data, primary):
    tables = _one_hot_tables(4, data)
    traded = T.traded_mix(data.sessions, data.folds, tables, primary.paths)
    train_cells = T.cells(primary.paths).train
    for fold in data.folds:
        lagged = T.lagged_path(primary.paths, fold.number).loc[fold.test(data.sessions)]
        held = traded.mix.loc[lagged.index]
        for session, state in lagged.items():
            row = held.loc[session].to_numpy()
            if np.isfinite(state) and int(state) in train_cells[fold.number]:
                assert row[int(state)] == 1.0 and row.sum() == 1.0
                assert not traded.fallback.loc[session]
            else:
                assert np.allclose(row, 0.1) and traded.fallback.loc[session]


def test_non_qualifying_rows_abstain_and_a_control_fold_is_all_fallback(data, primary):
    c = T.cells(primary.paths)
    fold = data.folds[0]
    only = {f: frozenset() if f == fold.number else s for f, s in c.train.items()}
    traded = T.traded_mix(data.sessions, data.folds, _one_hot_tables(4, data), primary.paths,
                          train_cells=only)
    assert traded.fallback.loc[fold.test(data.sessions)].all()
    tables = {**_one_hot_tables(4, data), fold.number: None}
    fallback = {f.number: pd.Series(np.r_[1.0, np.zeros(9)], index=list(LIBRARY))
                for f in data.folds}
    traded = T.traded_mix(data.sessions, data.folds, tables, primary.paths, fallback)
    held = traded.mix.loc[fold.test(data.sessions)].to_numpy()
    assert (held[:, 0] == 1.0).all() and traded.fallback.loc[fold.test(data.sessions)].all()
    assert 0.0 < traded.fallback_share <= 1.0
    with pytest.raises(KeyError):
        T.traded_mix(data.sessions, data.folds, {1: None}, primary.paths)


def test_a_per_fold_mix_holds_each_fold_its_own_mix(data):
    mixes = {f.number: pd.Series(np.eye(10)[f.number], index=list(LIBRARY))
             for f in data.folds}
    mix = T.per_fold_mix(data.sessions, data.folds, mixes)
    assert np.allclose(mix.loc[data.sessions < data.test_sessions[0]].to_numpy(), 0.1)
    for fold in data.folds:
        held = mix.loc[fold.test(data.sessions)].to_numpy()
        assert (held[:, fold.number] == 1.0).all() and (held.sum(axis=1) == 1.0).all()


def test_a_book_is_the_protocol_pipeline_read_on_the_test_sessions(data, primary):
    traded = T.traded_mix(data.sessions, data.folds, _one_hot_tables(4, data), primary.paths)
    book = T.build_book(data, traded.mix)
    weights = protocol.blend(data.signals, traded.mix)
    targeted = protocol.target_volatility(weights, data.excess)
    test = data.test_sessions
    for bps in (5.0, 10.0, 20.0):
        net = protocol.net_of_costs(targeted.returns, targeted.weights, bps=bps).loc[test]
        pd.testing.assert_series_equal(book.net(bps), net)
    assert book.turnover == protocol.annual_turnover(targeted.weights.loc[test])
    assert book.sigma == pytest.approx(np.sqrt(252) * book.net5.std(ddof=1), rel=1e-12)
    assert book.cap_share == pytest.approx(targeted.cap_binds.loc[test].astype(float).mean())
    assert book.missing == 0
    beta = protocol.realised_beta(book.net5, data.market.loc[test])
    assert book.beta() == beta
    shown = repr(book)
    assert "sigma" in shown and "mean" not in shown and "sharpe" not in shown.lower()


def test_the_control_is_the_equal_weight_blend_and_untargeted_books_hold_as_given(data):
    control = T.control_book(data)
    equal = T.build_book(data, protocol.equal_mix(list(LIBRARY), data.sessions))
    pd.testing.assert_series_equal(control.net5, equal.net5)
    held = control.book.weights
    raw = T.build_book(data, weights=held, targeted=False)
    assert np.isnan(raw.cap_share)
    expected = (held * data.excess).sum(axis=1).where(held.notna().all(axis=1))
    net = protocol.net_of_costs(expected, held, bps=5.0).loc[data.test_sessions]
    pd.testing.assert_series_equal(raw.net5, net, check_names=False)
    with pytest.raises(ValueError):
        T.build_book(data, protocol.equal_mix(list(LIBRARY), data.sessions), weights=held)


# ----------------------------------------------------------------------- pair statistics


def _legs(n: int = 700, seed: int = 3) -> tuple[pd.Series, pd.Series]:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-04", periods=n)
    common = rng.normal(0.0, 0.006, n)
    a = pd.Series(common + rng.normal(0.0004, 0.002, n), index=idx)
    b = pd.Series(common + rng.normal(0.0, 0.002, n), index=idx)
    return a, b


def test_the_pair_threshold_is_the_blinded_mde_with_a_floor():
    a, b = _legs()
    result = T.pair_threshold(a, b, draws=150)
    z = protocol.mde_z(0.05 / 6)
    for block in T.POWER_BLOCKS:
        direct = protocol.blinded_mde(a.to_numpy(), b.to_numpy(), mean_block=block, draws=150)
        assert result.se[block] == direct.se
        assert result.mde[block] == pytest.approx(direct.mde, rel=1e-12)
        assert result.mde_corrected[block] == result.se[block] * z
    assert result.threshold == max(0.338, max(result.mde_corrected.values()))
    assert result.se_star == max(result.se.values())
    assert result.se[result.block_star] == result.se_star
    assert result.observed < protocol.BLIND_TOLERANCE
    assert set(result.as_dict()) >= {"se", "threshold", "se_star", "block_star"}


def test_the_pair_threshold_reads_no_mean():
    a, b = _legs()
    base = T.pair_threshold(a, b, draws=100)
    shifted = T.pair_threshold(a + 0.01, b - 0.003, draws=100)
    for block in T.POWER_BLOCKS:
        assert shifted.se[block] == pytest.approx(base.se[block], rel=1e-9)


def test_the_pair_threshold_is_nan_on_a_missing_leg_and_the_same_on_three_workers():
    a, b = _legs()
    gap = a.copy()
    gap.iloc[10] = np.nan
    missing = T.pair_threshold(gap, b, draws=50)
    assert np.isnan(missing.threshold) and np.isnan(missing.se_star)
    one = T.pair_threshold(a, b, draws=60, workers=1)
    three = T.pair_threshold(a, b, draws=60, workers=3)
    assert one == three


def test_the_pair_reading_takes_the_larger_p_and_the_sign_rules():
    a, b = _legs()
    t = protocol.paired_hac_t(a, b, lags=6)
    assert t > 0
    reading = T.pair_reading(0.4, 0.2, a, b)
    from scipy import stats

    assert reading.p_boot == pytest.approx(2 * (1 - stats.norm.cdf(2.0)))
    assert reading.p_hac == pytest.approx(2 * (1 - stats.norm.cdf(abs(t))))
    assert reading.p == max(reading.p_boot, reading.p_hac) == reading.p_raw
    against = T.pair_reading(-0.4, 0.2, a, b)
    assert against.p_hac == 1.0 and against.p == 1.0
    wrong_sign = T.pair_reading(0.4, 0.2, b, a)
    assert wrong_sign.t_hac < 0 and wrong_sign.p_hac == 1.0 and wrong_sign.p == 1.0
    gap = a.copy()
    gap.iloc[3] = np.nan
    assert np.isnan(T.pair_reading(0.4, 0.2, gap, b).p)
    assert np.isnan(T.pair_reading(np.nan, 0.2, a, b).p)


def test_the_kill_is_anchored_to_the_bar_and_capped_at_12():
    assert T.kill(5.0, 0.338, 0.0788).k_kill == 12.0
    low = T.kill(5.0, 0.338, 0.05)
    assert low.k_kill == pytest.approx(0.338 * 0.05 * 10_000 / 20)
    assert low.fires is False
    assert T.kill(low.k_kill + 1e-9, 0.338, 0.05).fires is True
    assert T.kill(low.k_kill, 0.338, 0.05).fires is False
    assert T.kill(np.nan, 0.338, 0.05).fires is None


def test_sharpe_and_sd_refuse_a_gap():
    x = pd.Series([0.01, -0.005, 0.002, 0.004])
    assert T.sharpe(x) == pytest.approx(np.sqrt(252) * x.mean() / x.std(ddof=1))
    assert T.realised_sd(x) == pytest.approx(np.sqrt(252) * x.std(ddof=1))
    x.iloc[1] = np.nan
    assert np.isnan(T.sharpe(x)) and np.isnan(T.realised_sd(x))


def test_the_null_summary_is_nan_on_a_single_broken_draw():
    null = np.arange(1000, dtype=float)
    s = T.null_summary(null)
    assert (s.q50, s.q99) == (np.percentile(null, 50), np.percentile(null, 99))
    assert s.resolution == s.q99 - s.q50 and s.mde_c == s.q99 - s.q20
    null[5] = np.inf
    assert np.isnan(T.null_summary(null).q99)


# ------------------------------------------------------------------------------ verdicts


def test_level_precedence_follows_13_2():
    assert T.level_verdict(["PASS", "PASS"]) == "PASS"
    assert T.level_verdict(["PASS", "DOMINATED"]) == "DOMINATED"
    assert T.level_verdict(["UNDERPOWERED", "NOT SHOWN"]) == "UNDERPOWERED"
    assert T.level_verdict(["NOT SHOWN", "FAIL (cost)"]) == "FAIL (cost)"
    assert T.level_verdict(["DOWNGRADED (beta)", "UNDECIDABLE"]) == "UNDECIDABLE"
    assert T.level_verdict(["DOMINATED (volatility)", "DOWNGRADED (PIT)"]) == "DOWNGRADED (PIT)"
    assert T.level_verdict(["UNDECIDED", "UNDERPOWERED"]) == "UNDECIDED"
    assert T.level_verdict(["NOT SHOWN"]) == "NOT SHOWN"
    with pytest.raises(ValueError):
        T.level_verdict(["MAYBE"])


PASSING = dict(delta5=0.5, delta10=0.4, threshold=0.338, p=0.001, kill_fires=False,
               beta_pct=0.5, diff_pct=0.99, delta_w=0.1, fallback_share=0.07,
               leg_missing=False)


@pytest.mark.parametrize(("change", "verdict"), [
    ({}, "PASS"),
    ({"p": np.nan}, "UNDECIDABLE"),
    ({"beta_pct": np.nan}, "UNDECIDABLE"),
    ({"kill_fires": None}, "UNDECIDABLE"),
    ({"leg_missing": True}, "UNDECIDABLE"),
    ({"fallback_share": 0.51}, "UNDECIDABLE"),
    ({"other_finite": False}, "UNDECIDABLE"),
    ({"kill_fires": True, "delta5": -1.0}, "FAIL (cost)"),
    ({"delta5": 0.0}, "FAIL"),
    ({"delta10": 0.0}, "UNDECIDED"),
    ({"delta5": 0.3}, "UNDERPOWERED"),
    ({"p": 0.009}, "UNDERPOWERED"),
    ({"p": 0.009, "holm_rejected": True}, "PASS"),
    ({"beta_pct": 0.95}, "DOWNGRADED (beta)"),
    ({"diff_pct": 0.949}, "DOWNGRADED (not conditional)"),
    ({"delta_w": 0.5}, "DOMINATED (volatility)"),
])
def test_the_paired_verdict_lines_apply_in_order(change, verdict):
    assert T.paired_verdict(**{**PASSING, **change}) == verdict


@pytest.mark.parametrize(("change", "verdict"), [
    ({}, "PASS"),
    ({"exact": False}, "UNDECIDABLE"),
    ({"p": np.nan}, "UNDECIDABLE"),
    ({"dominated": None}, "UNDECIDABLE"),
    ({"other_undecidable": True}, "UNDECIDABLE"),
    ({"gate_failed": True}, "FAIL"),
    ({"gate_failed": None}, "UNDECIDABLE"),
    ({"p": 0.02}, "NOT SHOWN"),
    ({"p": 0.02, "powered": True}, "FAIL"),
    ({"p": 0.009}, "NOT SHOWN"),
    ({"p": 0.009, "holm_rejected": True}, "PASS"),
    ({"dominated": True}, "DOMINATED"),
])
def test_the_placebo_verdicts(change, verdict):
    base = dict(statistic=0.3, p=0.002, exact=True, dominated=False)
    assert T.placebo_verdict(**{**base, **change}) == verdict


def test_pit_downgrades_only_a_pass_and_requires_the_rebuild():
    assert T.apply_pit("PASS", "PASS") == "PASS"
    assert T.apply_pit("PASS", "DOMINATED") == "DOWNGRADED (PIT)"
    assert T.apply_pit("NOT SHOWN", None) == "NOT SHOWN"
    with pytest.raises(ValueError):
        T.apply_pit("PASS", None)
    assert T.robust_label("PASS", [0.2, -0.01]) == "PASS (not robust)"
    assert T.robust_label("PASS", [0.2, np.nan]) == "PASS"
    assert T.robust_label("FAIL", [-1.0]) == "FAIL"


# ---------------------------------------------------------------------------------- Holm


def test_holm_steps_down_over_the_six_primaries():
    result = T.holm({"A-1": 0.001, "A-2": 0.009, "B-1": 0.012, "B-2": 0.5, "C-1": 0.02})
    assert result.rejected == {"A-1", "A-2", "B-1"}
    assert list(result.table["test"])[:3] == ["A-1", "A-2", "B-1"]
    np.testing.assert_allclose(result.table["bar"], 0.05 / np.array([6, 5, 4, 3, 2, 1]))
    assert result.p["C-2"] == 1.0


def test_holm_never_shrinks_the_family():
    assert T.holm({"A-1": 0.01}).rejected == frozenset()
    missing = T.holm({"A-1": 0.008, "A-2": None, "B-1": np.nan})
    assert missing.rejected == {"A-1"} and missing.p["A-2"] == missing.p["B-1"] == 1.0
    undecidable = T.holm({"A-1": 0.001}, verdicts={"A-1": "UNDECIDABLE"})
    assert undecidable.rejected == frozenset() and undecidable.p["A-1"] == 1.0
    assert T.holm({"C-2": 0.0}).p["C-2"] == 1.0
    with pytest.raises(ValueError):
        T.holm({"A-3": 0.01})
    with pytest.raises(ValueError):
        T.holm({"A-1": 1.5})


# -------------------------------------------------------------------------------- trials


def test_a_trial_row_carries_the_13_5_schema(tmp_path):
    path = tmp_path / "trials.parquet"
    config, metrics = T.log_trial("A-1", T.PRIMARY, sharpe=0.1, delta=0.05, threshold=0.34,
                                  p=1.0, verdict="UNDERPOWERED", sessions=5031, path=path)
    assert config["prespec"] == "docs/PRESPEC_TWOSIGMA.md, LOCKED 2026-09-23"
    assert (config["role"], config["variant"], config["K"], config["d"]) == (
        "primary", "primary", 4, 0.5)
    assert config["library"] == list(LIBRARY) and config["build"] == "3b64ab2"
    assert config["context_train_start"] == "1995-01-04" and config["anchor"] == "end"
    assert config["placebo"] == {"method": "uniform", "draws": 1000, "seed": 0,
                                 "blocks": "(fold, segment)"}
    assert (config["cost_bps"], config["vol_target"], config["cap"]) == (5.0, 0.10, 3.0)
    frame = pd.read_parquet(path)
    assert set(frame.columns) >= {"m_sharpe", "m_delta", "m_threshold", "m_p", "m_p2",
                                  "m_placebo_pct", "m_sessions", "m_verdict", "m_note"}
    assert frame["family"].tolist() == ["twosigma"]
    a2, _ = T.log_trial("A-2", T.PRIMARY, sharpe=np.nan, delta=0.1, threshold=0.01,
                        p=0.3, verdict="NOT SHOWN", sessions=5031, path=path)
    assert a2["d"] is None
    sens, _ = T.log_trial("B", T.K3, sharpe=0.2, delta=-0.1, threshold=0.4, p=1.0, p2=0.4,
                          verdict="FAIL", sessions=5031, path=path)
    assert (sens["role"], sens["variant"], sens["K"], sens["d"]) == ("sensitivity", "K=3", 3,
                                                                       None)
    d_row, _ = T.log_trial("A-1", T.D100, sharpe=0.2, delta=0.1, threshold=0.41, p=0.2,
                           verdict="UNDERPOWERED", sessions=5031, path=path)
    assert d_row["d"] == 1.0
    assert len(pd.read_parquet(path)) == 4


@pytest.mark.parametrize(("test", "variant", "kwargs", "match"), [
    ("A-1", T.PIT18, {}, "no trial row"),
    ("A-1", T.K3, {}, "declared"),
    ("A", T.D025, {}, "declared"),
    ("A-1", T.PRIMARY, {"verdict": "GOOD"}, "verdict"),
    ("C-2", T.PRIMARY, {"verdict": "PASS"}, "never a PASS"),
    ("A-1", T.PRIMARY, {"delta": np.nan}, "non-finite"),
    ("A-1", T.PRIMARY, {"sessions": 0}, "sessions"),
])
def test_a_trial_row_is_refused_outside_the_schema(tmp_path, test, variant, kwargs, match):
    base = dict(sharpe=0.1, delta=0.05, threshold=0.34, p=0.5, verdict="UNDERPOWERED",
                sessions=5031, path=tmp_path / "t.parquet")
    with pytest.raises(ValueError, match=match):
        T.log_trial(test, variant, **{**base, **kwargs})


def test_a_second_row_for_the_same_test_is_refused(tmp_path):
    path = tmp_path / "t.parquet"
    kwargs = dict(sharpe=0.1, delta=0.05, threshold=0.34, p=0.5, verdict="UNDERPOWERED",
                  sessions=5031, path=path)
    T.log_trial("A-1", T.PRIMARY, **kwargs)
    with pytest.raises(ValueError, match="already logged"):
        T.log_trial("A-1", T.PRIMARY, **kwargs)
    kwargs["verdict"] = "UNDECIDABLE"
    T.log_trial("A-1", T.D025, **{**kwargs, "delta": np.nan})


# ------------------------------------------------------------------------ the thresholds

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


THRESHOLDS = {"T_A1": 0.3380000000000001, "se": {21: 0.1, 63: 0.12, 126: 0.11},
              "block_star": 63, "cells": 14}


def _write(repo: Path, values=THRESHOLDS, section="A") -> Path:
    path = repo / T.THRESHOLDS_FILE
    T.write_thresholds(values, section=section, path=path, root=repo, inputs=INPUTS)
    return path


def _verify(repo: Path, values=THRESHOLDS, section="A"):
    return T.verify_for_reading(values, section=section, path=repo / T.THRESHOLDS_FILE,
                                root=repo, inputs=INPUTS)


def test_the_reading_runs_only_on_committed_bitwise_thresholds(repo):
    path = _write(repo)
    with pytest.raises(T.ReadingRefused, match="not tracked"):
        _verify(repo)
    _git(repo, "add", T.THRESHOLDS_FILE)
    _git(repo, "commit", "-q", "-m", "thresholds")
    stored = _verify(repo)
    assert stored["T_A1"] == THRESHOLDS["T_A1"] and stored["se"]["63"] == 0.12
    payload = json.loads(path.read_text())
    assert payload["packages"] == T.locked_versions(repo) == T.installed_versions()
    assert payload["inputs"] == T.input_hashes(repo, INPUTS)
    assert payload["git_head"]["A"] and payload["prespec"].endswith("(30f7d69)")
    assert payload["thresholds"]["A"]["T_A1"]["float.hex"] == (0.3380000000000001).hex()

    nudged = {**THRESHOLDS, "T_A1": np.nextafter(THRESHOLDS["T_A1"], 1.0)}
    with pytest.raises(T.ReadingRefused, match="bitwise"):
        _verify(repo, nudged)
    with pytest.raises(T.ReadingRefused, match="bitwise"):
        _verify(repo, {**THRESHOLDS, "extra": 1})
    with pytest.raises(T.ReadingRefused, match="section"):
        _verify(repo, THRESHOLDS, section="B")


def test_the_reading_refuses_a_modified_file_an_input_or_a_version(repo):
    path = _write(repo)
    _git(repo, "add", T.THRESHOLDS_FILE)
    _git(repo, "commit", "-q", "-m", "thresholds")
    original = path.read_text()
    path.write_text(original.replace('"cells": 14', '"cells": 15'))
    with pytest.raises(T.ReadingRefused, match="HEAD"):
        _verify(repo)
    _git(repo, "add", T.THRESHOLDS_FILE)
    with pytest.raises(T.ReadingRefused, match="HEAD"):
        _verify(repo)
    _git(repo, "reset", "-q", "--hard")
    _verify(repo)
    (repo / INPUTS[0]).write_bytes(b"republished")
    with pytest.raises(T.ReadingRefused, match="SHA-256"):
        _verify(repo)
    _git(repo, "checkout", "-q", "--", ".")
    (repo / INPUTS[0]).write_bytes(INPUTS[0].encode() * 10)
    _verify(repo)
    lock = (repo / "uv.lock").read_text()
    numpy_version = T.locked_versions(repo)["numpy"]
    (repo / "uv.lock").write_text(lock.replace(
        f'name = "numpy"\nversion = "{numpy_version}"', 'name = "numpy"\nversion = "0.0.1"'))
    with pytest.raises(T.ReadingRefused, match="package versions"):
        _verify(repo)


def test_sections_merge_and_non_finite_thresholds_are_never_written(repo):
    _write(repo)
    _write(repo, {"MDE_C": 1.5, "S_star": 7.9}, section="C")
    stored = T.read_thresholds(repo / T.THRESHOLDS_FILE)["thresholds"]
    assert set(stored) == {"A", "C"} and stored["C"]["S_star"] == 7.9
    with pytest.raises(ValueError, match="non-finite"):
        _write(repo, {"T_B2": float("nan")}, section="B")
    assert T.decode_thresholds(T.encode_thresholds({"x": [0.1, 2, True, "s"]})) == {
        "x": [0.1, 2, True, "s"]}


# ------------------------------------------------------------------------------ the pool


def _probe(shared: float, j: int) -> tuple[int, float, float]:
    return j, shared * j, float(np.random.default_rng(j).random())


def test_the_pool_map_is_ordered_and_independent_of_the_worker_count():
    serial = T.draw_map(_probe, 9, 2.0, workers=1)
    pooled = T.draw_map(_probe, 9, 2.0, workers=2)
    assert serial == pooled == [_probe(2.0, j) for j in range(9)]
    assert T.draw_map(_probe, 0, 2.0, workers=4) == []


def _pairs_of_draw(shared: tuple[T.TreeData, T.PlaceboDraws], j: int) -> tuple[int, int]:
    data, draws = shared
    return T.cells(draws.draw(j)).n_pairs, len(data.sessions)


def test_the_pool_carries_the_tree_and_the_placebo_to_its_workers(data, primary):
    draws = T.placebo_draws(primary.paths, n=4)
    serial = T.draw_map(_pairs_of_draw, 4, (data, draws), workers=1)
    pooled = T.draw_map(_pairs_of_draw, 4, (data, draws), workers=2)
    assert serial == pooled == [(T.cells(primary.paths).n_pairs, len(data.sessions))] * 4


# -------------------------------------------------------------------- the real store


@pytest.mark.skipif(not all((ROOT / f).exists() for f in T.INPUT_FILES),
                    reason="the store is not on disk")
def test_the_real_store_loads_into_the_declared_shape():
    real = T.load_tree_data()
    assert len(real.sessions) == 7946
    assert (real.sessions[0], real.sessions[-1]) == (pd.Timestamp("1995-01-04"),
                                                     pd.Timestamp("2026-07-31"))
    assert len(real.folds) == 5 and len(real.test_sessions) == 5031
    assert [len(f.test(real.sessions)) for f in real.folds] == [1007, 1007, 1007, 1006, 1004]
    assert real.excess.shape == (7946, 49) and real.legs.shape == (7946, 10)
    assert np.isfinite(real.legs.to_numpy()).all()
    hashes = T.input_hashes()
    assert list(hashes) == list(T.INPUT_FILES) and all(len(h) == 64 for h in hashes.values())


def test_dataclasses_are_frozen(data):
    with pytest.raises(dataclasses.FrozenInstanceError):
        data.sessions = None  # type: ignore[misc]

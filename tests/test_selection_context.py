"""The Two Sigma context partition: fitted on training rows only, applied row by row.

The tests that matter are the causality ones. They do not inspect the code; they
perturb the future and assert that the past does not move — every moment, slope,
centroid and label of a fold whose cut-off precedes the perturbation must be
bit-identical — and they perturb one test row and assert that no other label moves.

Everything runs on synthetic panels. The one test that reads the real features
reproduces the disclosed P3 clock and skips when the data are absent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regime_lab.analysis.placebo import transitions
from regime_lab.config import CACHE, RAW
from regime_lab.selection.context import (
    CONTEXT_FEATURES,
    LOG_RV,
    align_labels,
    as_of,
    context_panel,
    eta_squared,
    fit_context,
    fold_windows,
    load_context_features,
    load_log_realised_vol,
    log_realised_vol,
    placebo_groups,
    qualifying_cells,
    session_paths,
    trailing_mode,
    transitions_within,
    walk_forward_context,
)
from regime_lab.selection.folds import walk_forward_folds
from regime_lab.selection.protocol import map_states

N_INIT = 3


def _panel(n: int = 3_200, features: int = 5, seed: int = 0) -> pd.DataFrame:
    """Weekday panel: a persistent three-state latent, loadings on log vol, noise."""
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2000-01-03", periods=n)
    moves = rng.random(n) > 0.97
    latent = np.cumsum(np.where(moves, rng.integers(1, 3, n), 0)) % 3
    centres = rng.normal(0.0, 1.5, (3, features))
    log_rv = np.empty(n)
    log_rv[0] = np.log(0.15)
    shocks = rng.normal(0.0, 0.06, n)
    for t in range(1, n):
        log_rv[t] = np.log(0.15) + 0.98 * (log_rv[t - 1] - np.log(0.15)) + shocks[t]
    loadings = rng.normal(0.0, 1.0, features)
    values = centres[latent] + np.outer(log_rv, loadings) + rng.normal(0, 0.5, (n, features))
    panel = pd.DataFrame(values, index=index, columns=[f"f{i}" for i in range(features)])
    panel[LOG_RV] = log_rv
    return panel


def _folds(panel: pd.DataFrame, sessions: pd.DatetimeIndex | None = None):
    return walk_forward_folds(panel.index if sessions is None else sessions,
                              n_folds=3, test_years=2.0)


# ------------------------------------------------------------------------ inputs


def test_as_of_reads_the_last_observation_and_never_the_next():
    values = pd.Series([1.0, 2.0, 3.0], index=pd.to_datetime(["2020-01-02", "2020-01-06",
                                                               "2020-01-09"]))
    dates = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-05", "2020-01-08",
                            "2020-01-10"])
    assert as_of(values, pd.DatetimeIndex(dates)).tolist()[1:] == [1.0, 1.0, 2.0, 3.0]
    assert np.isnan(as_of(values, pd.DatetimeIndex(dates)).iloc[0])


def test_log_realised_vol_includes_the_session_it_is_stamped_on():
    rng = np.random.default_rng(0)
    prices = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 60))),
                       index=pd.bdate_range("2020-01-01", periods=60))
    lv = log_realised_vol(prices, window=21)
    returns = np.log(prices).diff()
    expected = np.log(returns.iloc[39 - 20 : 40].std() * np.sqrt(252))
    assert lv.iloc[39] == pytest.approx(expected)
    assert lv.iloc[:21].isna().all() and lv.iloc[21:].notna().all()
    bumped = prices.copy()
    bumped.iloc[45:] *= 1.5
    pd.testing.assert_series_equal(log_realised_vol(bumped).iloc[:45], lv.iloc[:45])


def test_the_context_panel_needs_every_feature_and_keeps_complete_rows():
    index = pd.bdate_range("2020-01-01", periods=10)
    features = pd.DataFrame(1.0, index=index, columns=list(CONTEXT_FEATURES))
    features.iloc[2, 3] = np.nan
    log_rv = pd.Series(np.log(0.2), index=index[1:])
    panel = context_panel(features, log_rv)
    assert list(panel.columns) == [*CONTEXT_FEATURES, LOG_RV]
    assert list(panel.index) == [d for i, d in enumerate(index) if i not in (0, 2)]
    with pytest.raises(KeyError):
        context_panel(features.drop(columns=CONTEXT_FEATURES[0]), log_rv)


# -------------------------------------------------------------------- one fit


def test_the_fit_standardises_and_orthogonalises_on_its_own_rows():
    train = _panel().iloc[:1_500]
    model = fit_context(train, n_init=N_INIT)
    z = model.transform(train)
    v = ((train[LOG_RV] - train[LOG_RV].mean()) / train[LOG_RV].std(ddof=0)).to_numpy()
    assert np.abs(z.mean().to_numpy()).max() < 1e-10
    assert np.abs(z.to_numpy().T @ v).max() < 1e-8
    raw = (train.drop(columns=LOG_RV) - train.drop(columns=LOG_RV).mean()) / train.drop(
        columns=LOG_RV).std(ddof=0)
    slopes = raw.to_numpy().T @ v / (v @ v)
    np.testing.assert_allclose(model.slopes.to_numpy(), slopes)
    pd.testing.assert_series_equal(model.train_labels, model.predict(train))
    assert model.orthogonalised and model.centroids.shape == (4, 5)


def test_test_rows_are_scaled_with_training_moments():
    panel = _panel()
    train, test = panel.iloc[:1_500], panel.iloc[1_500:1_600]
    model = fit_context(train, n_init=N_INIT)
    x = train.drop(columns=LOG_RV)
    z = (test.drop(columns=LOG_RV) - x.mean()) / x.std(ddof=0)
    v = (test[LOG_RV] - train[LOG_RV].mean()) / train[LOG_RV].std(ddof=0)
    expected = z - np.outer(v.to_numpy(), model.slopes.to_numpy())
    pd.testing.assert_frame_equal(model.transform(test), expected)


def test_the_fit_is_bit_identical_from_one_call_to_the_next():
    train = _panel().iloc[:1_800]
    first = fit_context(train, n_init=N_INIT)
    for _ in range(3):
        again = fit_context(train, n_init=N_INIT)
        np.testing.assert_array_equal(again.centroids, first.centroids)
        pd.testing.assert_series_equal(again.train_labels, first.train_labels)


def test_the_unorthogonalised_fit_needs_no_volatility():
    panel = _panel().iloc[:1_500]
    with_vol = fit_context(panel, orthogonalise=False, n_init=N_INIT)
    without = fit_context(panel.drop(columns=LOG_RV), orthogonalise=False, n_init=N_INIT)
    pd.testing.assert_series_equal(with_vol.train_labels, without.train_labels)
    assert not with_vol.orthogonalised


def test_a_constant_training_feature_is_refused():
    panel = _panel().iloc[:500].copy()
    panel["f0"] = 1.0
    with pytest.raises(ValueError, match="constant"):
        fit_context(panel, n_init=N_INIT)


def test_a_test_label_depends_only_on_its_row_and_the_fit():
    panel = _panel()
    model = fit_context(panel.iloc[:1_500], n_init=N_INIT)
    test = panel.iloc[1_500:1_900]
    batch = model.predict(test)
    rows = [0, 17, 99, 250, 399]
    alone = pd.concat([model.predict(test.iloc[[r]]) for r in rows])
    pd.testing.assert_series_equal(alone, batch.iloc[rows])
    shocked = test.copy()
    shocked.iloc[99] = shocked.iloc[99] * -4.0 + 3.0
    moved = model.predict(shocked)
    keep = np.arange(len(test)) != 99
    pd.testing.assert_series_equal(moved[keep], batch[keep])


# ------------------------------------------------------------------ walk-forward


def test_align_labels_undoes_a_renumbering():
    rng = np.random.default_rng(0)
    reference = pd.Series(rng.integers(0, 4, 500))
    permutation = np.array([2, 0, 3, 1])
    candidate = pd.Series(permutation[reference.to_numpy()])
    candidate.iloc[:40] = rng.integers(0, 4, 40)
    mapping = align_labels(reference, candidate, 4)
    np.testing.assert_array_equal(mapping[permutation], np.arange(4))


def test_the_windows_tile_the_feature_calendar_by_the_cutoffs():
    panel = _panel()
    sessions = panel.index.delete(np.arange(5, len(panel.index), 53))
    folds = _folds(panel, sessions)
    wf = walk_forward_context(panel, folds, n_init=N_INIT)
    assert not wf.labels.index.has_duplicates
    expected = panel.index[(panel.index > folds[0].train_cutoff)
                           & (panel.index <= folds[-1].test_cutoff)]
    assert wf.labels.index.equals(expected)
    for fold, model in zip(folds, wf.models, strict=True):
        train, test = fold_windows(panel.index, fold)
        assert model.train_labels.index.equals(train)
        assert wf.test_labels(fold.number).index.equals(test)
        assert model.train_end <= fold.train_cutoff < test[0]
    assert wf.numbers == (1, 2, 3)


def test_the_walk_forward_is_causal():
    panel = _panel()
    folds = _folds(panel)
    base = walk_forward_context(panel, folds, n_init=N_INIT)
    cut = folds[1].train_cutoff + pd.Timedelta(days=200)
    later = panel.index > cut
    perturbed = panel.copy()
    perturbed.loc[later] = perturbed.loc[later] * 2.5 - 1.0
    other = walk_forward_context(perturbed, folds, n_init=N_INIT)
    for k in (0, 1):
        a, b = base.models[k], other.models[k]
        pd.testing.assert_series_equal(a.train_labels, b.train_labels)
        pd.testing.assert_series_equal(a.mean, b.mean)
        pd.testing.assert_series_equal(a.scale, b.scale)
        pd.testing.assert_series_equal(a.slopes, b.slopes)
        np.testing.assert_array_equal(a.centroids, b.centroids)
        assert (a.vol_mean, a.vol_scale) == (b.vol_mean, b.vol_scale)
    assert base.agreement[:2] == other.agreement[:2]
    early = base.labels.index <= cut
    pd.testing.assert_series_equal(base.labels[early], other.labels[early])
    pd.testing.assert_series_equal(base.aligned[early], other.aligned[early])
    assert not base.models[2].mean.equals(other.models[2].mean)


def test_train_from_moves_every_training_start():
    panel = _panel()
    sessions = panel.index[panel.index >= panel.index[300]]
    folds = _folds(panel, sessions)
    own = walk_forward_context(panel, folds, n_init=N_INIT)
    early = walk_forward_context(panel, folds, n_init=N_INIT, train_from=panel.index[0])
    assert all(m.train_start == sessions[0] for m in own.models)
    assert all(m.train_start == panel.index[0] for m in early.models)
    assert own.labels.index.equals(early.labels.index)


def test_transitions_split_into_within_and_across_folds():
    panel = _panel()
    wf = walk_forward_context(panel, _folds(panel), n_init=N_INIT)
    within = sum(transitions(wf.test_labels(k)) for k in wf.numbers)
    assert wf.within_fold_transitions() == within
    assert transitions_within(wf.aligned, wf.fold) == within
    assert wf.boundary_transitions() == transitions(wf.aligned) - within
    assert 0 <= wf.boundary_transitions() <= len(wf.numbers) - 1
    assert wf.agreement[0] == 1.0 and all(0.0 <= a <= 1.0 for a in wf.agreement)


# ------------------------------------------------------- paths on the sessions


def test_session_paths_read_each_fold_in_its_own_numbering():
    panel = _panel()
    sessions = panel.index.delete(np.arange(7, len(panel.index), 41))
    folds = _folds(panel, sessions)
    wf = walk_forward_context(panel, folds, n_init=N_INIT)
    paths = session_paths(wf, folds, sessions)
    assert paths.dtype == np.int64
    assert paths.index.names == ["fold", "segment", "session"]
    for fold in folds:
        train = paths.xs((fold.number, "train"), level=["fold", "segment"])
        test = paths.xs((fold.number, "test"), level=["fold", "segment"])
        assert train.index.equals(fold.train(sessions))
        assert test.index.equals(fold.test(sessions))
        model = wf.model(fold.number)
        np.testing.assert_array_equal(
            train.to_numpy(), model.train_labels.reindex(train.index).to_numpy())
        np.testing.assert_array_equal(
            test.to_numpy(), wf.test_labels(fold.number).reindex(test.index).to_numpy())
    groups = placebo_groups(paths)
    assert groups.nunique() == 2 * len(folds)
    assert groups.iloc[0] == "1:train"


def test_the_mix_held_on_a_session_uses_the_label_stamped_one_session_before():
    panel = _panel()
    sessions = panel.index.delete(np.arange(3, len(panel.index), 29))
    folds = _folds(panel, sessions)
    wf = walk_forward_context(panel, folds, n_init=N_INIT)
    paths = session_paths(wf, folds, sessions)
    one_hot = pd.DataFrame(np.eye(4), index=range(4), columns=[f"s{i}" for i in range(4)])
    for fold in folds:
        path = paths.xs(fold.number, level="fold").droplevel("segment")
        held = map_states(path, one_hot).iloc[1:].to_numpy().argmax(axis=1)
        np.testing.assert_array_equal(held, path.shift(1).iloc[1:].to_numpy().astype(int))
        first_test = fold.test(sessions)[0]
        last_train = fold.train(sessions)[-1]
        mix = map_states(path, one_hot).loc[first_test]
        assert mix.to_numpy().argmax() == wf.model(fold.number).train_labels.loc[last_train]


def test_session_paths_refuse_foreign_folds():
    panel = _panel()
    folds = _folds(panel)
    wf = walk_forward_context(panel, folds, n_init=N_INIT)
    with pytest.raises(ValueError, match="folds"):
        session_paths(wf, folds[:2], panel.index)


# --------------------------------------------------------------------- eta squared


def test_eta_squared_on_a_hand_computed_case():
    labels = pd.Series([0, 0, 1, 1])
    values = pd.Series([1.0, 3.0, 5.0, 7.0])
    assert eta_squared(labels, values) == pytest.approx(16.0 / 20.0)
    assert eta_squared(labels, pd.Series([1.0, 2.0, 2.0, 1.0])) == pytest.approx(0.0)
    assert eta_squared(pd.Series([0, 0, 1, 1]), pd.Series([2.0, 2.0, 5.0, 5.0])) == 1.0
    assert np.isnan(eta_squared(labels, pd.Series([4.0] * 4)))
    assert np.isnan(eta_squared(labels, pd.Series([np.nan] * 4)))


def test_eta_squared_uses_common_dates_only():
    index = pd.bdate_range("2020-01-01", periods=6)
    labels = pd.Series([0, 0, 1, 1, 0, 1], index=index)
    values = pd.Series([1.0, 3.0, 5.0, 7.0], index=index[:4])
    assert eta_squared(labels, values) == pytest.approx(0.8)


# ------------------------------------------------------ cells and smoothing


def _paths(blocks: dict[tuple[int, str], list[float]]) -> pd.Series:
    """A stamped path in :func:`session_paths` layout from per-block label lists."""
    pieces, start = [], pd.Timestamp("2001-01-01")
    for (fold, segment), labels in blocks.items():
        dates = pd.bdate_range(start, periods=len(labels))
        start = dates[-1] + pd.Timedelta(days=1)
        index = pd.MultiIndex.from_arrays(
            [np.full(len(labels), fold), np.full(len(labels), segment), dates],
            names=["fold", "segment", "session"],
        )
        pieces.append(pd.Series(np.asarray(labels, dtype=float), index=index))
    return pd.concat(pieces).rename("state")


def test_a_cell_needs_both_the_sessions_and_the_episodes():
    three_short = [0] * 20 + [1] * 5 + [0] * 20 + [1] * 5 + [0] * 23 + [1] * 5   # 63, 3
    two_long = [0] * 40 + [1] * 5 + [0] * 40                                     # 80, 2
    many_short = ([2] * 6 + [1] * 1) * 10                                        # 60, 10
    cells = qualifying_cells(_paths({(1, "train"): three_short, (1, "test"): two_long,
                                     (2, "train"): many_short, (2, "test"): [0] * 5}))
    assert cells.loc[(1, "train", 0)].tolist() == [63, 3, True]
    assert cells.loc[(1, "test", 0)].tolist() == [80, 2, False]
    assert cells.loc[(2, "train", 2)].tolist() == [60, 10, False]
    assert cells.loc[(2, "test", 1)].tolist() == [0, 0, False]
    assert set(cells.index.get_level_values("state")) == {0, 1, 2}
    loose = qualifying_cells(_paths({(1, "train"): many_short}), min_sessions=60)
    assert bool(loose.loc[(1, "train", 2), "qualifies"])


def test_an_unlabelled_session_belongs_to_no_cell_and_still_separates_episodes():
    labels = [np.nan] * 3 + [0] * 30 + [np.nan] + [0] * 40
    cells = qualifying_cells(_paths({(1, "train"): labels}))
    assert cells.loc[(1, "train", 0)].tolist() == [70, 2, False]


def test_the_trailing_mode_breaks_ties_low_and_does_not_back_fill():
    path = _paths({(1, "train"): [2, 2, 1, 1, 0, 0, 0], (1, "test"): [1, 1, 1]})
    smooth = trailing_mode(path, window=4).to_numpy()
    assert np.isnan(smooth[:3]).all()
    # windows [2,2,1,1] -> tie -> 1; [2,1,1,0] -> 1; [1,1,0,0] -> 0; [1,0,0,0] -> 0; ...
    np.testing.assert_array_equal(smooth[3:], [1, 1, 0, 0, 0, 0, 1])


def test_the_trailing_mode_reads_neither_the_future_nor_another_fold():
    rng = np.random.default_rng(3)
    first = rng.integers(0, 4, 300).tolist()
    second = rng.integers(0, 4, 300).tolist()
    base = _paths({(1, "train"): first[:200], (1, "test"): first[200:],
                   (2, "train"): second[:200], (2, "test"): second[200:]})
    smooth = trailing_mode(base)
    for cut in (50, 199, 250):
        changed = first[:cut] + [(v + 1) % 4 for v in first[cut:]]
        moved = trailing_mode(_paths({(1, "train"): changed[:200], (1, "test"): changed[200:],
                                      (2, "train"): second[:200], (2, "test"): second[200:]}))
        fold1 = base.index.get_level_values("fold") == 1
        before = np.flatnonzero(fold1)[:cut]
        np.testing.assert_array_equal(moved.to_numpy()[before], smooth.to_numpy()[before])
        np.testing.assert_array_equal(moved.to_numpy()[~fold1], smooth.to_numpy()[~fold1])
    fold2 = smooth.xs(2, level="fold").to_numpy()
    assert np.isnan(fold2[:20]).all() and np.isfinite(fold2[20:]).all()
    with pytest.raises(ValueError):
        trailing_mode(_paths({(1, "train"): [0, np.nan, 1]}))


# ------------------------------------------------------------------ real data


@pytest.mark.skipif(
    not ((CACHE / "features.parquet").exists()
         and (RAW / "prices" / "cross_asset.parquet").exists()),
    reason="features or prices absent",
)
def test_the_disclosed_full_sample_clock_reproduces():
    panel = context_panel(load_context_features(), load_log_realised_vol())
    assert len(panel) == 9_007
    raw = fit_context(panel, orthogonalise=False).train_labels
    resid = fit_context(panel).train_labels
    assert transitions(raw) == 223
    assert transitions(resid) == 287
    assert eta_squared(resid, panel[LOG_RV]) == pytest.approx(0.048, abs=5e-4)
    assert eta_squared(raw, as_of(np.exp(load_log_realised_vol()), raw.index)) == \
        pytest.approx(0.421, abs=5e-4)

"""Level A of the Two Sigma tree must be the lock's §12.5-§12.7, and nothing else.

Every test runs on synthetic data (``tree.synthetic_tree_data`` with 49 industries, or a
hand-built path); nothing here reads ``data/``, so nothing needs to skip when the store
is absent. What is tested is what a reading would silently depend on: ``m`` is the
static-neutralised mean contrast of §12.5 and nothing else; the selector trades it
lagged, fold by fold; A-2's statistic is the plain mean of Spearman correlations of the
§12.5 profiles; the instrument reads no Sharpe, no beta and not the real ``R``; a planted
state effect is found and its absence is not; a volatility effect is credited to the
witness; the PIT rebuild can only downgrade; nothing after a date moves anything before
it; a NaN never reads as a verdict; and the reading refuses to run unless its
thresholds are committed, unmodified and bitwise equal.
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
from scipy import stats

from regime_lab.analysis.placebo import transitions
from regime_lab.config import ROOT
from regime_lab.selection import level_a as L
from regime_lab.selection import protocol
from regime_lab.selection import tree as T
from regime_lab.selection.context import LOG_RV, context_panel, eta_squared
from regime_lab.selection.library import LIBRARY

PLANTED_DRAWS = 120  # the fewest with 1/(n + 1) <= 0.05/6: a lock can hold
BOOT = 200
SCALE = 0.0005  # daily shift of a leg per unit of the planted profile


def _effect(seed: int = 1, k: int = 4) -> np.ndarray:
    effect = np.random.default_rng(seed).normal(size=(k, len(LIBRARY)))
    return effect - effect.mean(axis=1, keepdims=True)


def plant(data: T.TreeData, state: pd.Series, effect: np.ndarray, scale: float) -> T.TreeData:
    """Shift every signal's gross return by ``scale × effect[state held at t]``.

    The state held at *t* is ``state`` lagged one session. The instrument returns receive
    the minimum-norm perturbation ``e_t = W_t⁺ b_t`` of the ten weight rows ``W_t``, so
    each signal's leg moves by exactly ``b_t`` and every book on the blend moves with it.
    """
    held = state.astype(float).shift(1).to_numpy()
    shift = np.zeros((len(data.sessions), len(LIBRARY)))
    ok = np.isfinite(held)
    shift[ok] = effect[held[ok].astype(int)] * scale
    weights = np.stack([data.signals[name].to_numpy() for name in LIBRARY], axis=1)
    e = np.einsum("tns,ts->tn", np.linalg.pinv(weights), shift)
    excess = data.excess + e
    return dataclasses.replace(data, excess=excess, returns=excess.add(data.rf, axis=0),
                               legs=T.signal_legs(data.signals, excess))


def _fold5(paths: pd.Series) -> pd.Series:
    """Fold 5's stamped path covers every session: training then test."""
    return paths.xs(5, level="fold").droplevel("segment")


@pytest.fixture(scope="module")
def data() -> T.TreeData:
    return T.synthetic_tree_data(seed=0, n_industries=49, test_years=2.0, end="2019-12-31")


@pytest.fixture(scope="module")
def primary(data) -> T.Partition:
    return T.partition_paths(data, T.PRIMARY)


@pytest.fixture(scope="module")
def cells(primary) -> T.Cells:
    return T.cells(primary.paths)


@pytest.fixture(scope="module")
def planted(data, primary) -> T.TreeData:
    return plant(data, _fold5(primary.paths), _effect(), SCALE)


@pytest.fixture(scope="module")
def planted_run(planted, primary):
    return L._run(planted, T.PRIMARY, PLANTED_DRAWS, boot_draws=BOOT, partition=primary,
                  workers=1)


@pytest.fixture(scope="module")
def planted_reading(planted, planted_run) -> dict:
    return L._read(planted, planted_run, workers=1)


@pytest.fixture(scope="module")
def null_reading(data, primary) -> dict:
    run = L._run(data, T.PRIMARY, PLANTED_DRAWS, boot_draws=BOOT, partition=primary, workers=1)
    return L._read(data, run, workers=1)


# ------------------------------------------------------------------ a hand-built path


def _hand() -> tuple[pd.DataFrame, pd.Series]:
    """Two folds on 800 sessions. Training: episodes of 20 cycling states 0, 1, 2, with
    two episodes of state 3 (non-qualifying: 2 episodes). Fold 1's test visits 0 and 1
    (both qualify); fold 2's test has state 0 qualifying and state 1 not (30 sessions)."""
    sessions = pd.bdate_range("2001-01-01", periods=800)
    train = np.repeat(np.arange(30) % 3, 20)
    train[100:120] = 3
    train[300:320] = 3
    test1 = np.repeat(np.arange(8) % 2, 25)
    test2 = np.concatenate([[0] * 25, [1] * 10, [0] * 25, [1] * 10, [0] * 25, [1] * 10,
                            [0] * 95])
    pieces = []
    for fold, n_train, test in ((1, 400, test1), (2, 600, test2)):
        for segment, labels, dates in (("train", train[:n_train], sessions[:n_train]),
                                       ("test", test, sessions[n_train:n_train + 200])):
            index = pd.MultiIndex.from_arrays(
                [np.full(len(dates), fold), np.full(len(dates), segment), dates],
                names=["fold", "segment", "session"])
            pieces.append(pd.Series(labels, index=index))
    paths = pd.concat(pieces).rename("state").astype(np.int64)
    legs = pd.DataFrame(np.random.default_rng(3).normal(0.0, 0.01, (800, len(LIBRARY))),
                        index=sessions, columns=list(LIBRARY))
    return legs, paths


def _expected_profile(legs, paths, fold, qualifying, segment, states):
    """§12.5 steps 1-3 written out with pandas, independently of the module."""
    path = paths.xs(fold, level="fold")
    held = path.droplevel("segment").astype(float).shift(1)
    dates = path.xs(segment, level="segment").index
    held, x = held.loc[dates], legs.loc[dates]
    pooled = x[held.isin(sorted(qualifying))]
    rows = {}
    for k in states:
        if k in qualifying:
            rows[k] = np.sqrt(252) * (x[held == k].mean() - pooled.mean()) / pooled.std(ddof=1)
        else:
            rows[k] = pd.Series(np.nan, index=legs.columns)
    return pd.DataFrame(rows).T, held


def test_the_hand_path_has_the_cells_it_was_built_with():
    _, paths = _hand()
    c = T.cells(paths)
    assert c.train == {1: frozenset({0, 1, 2}), 2: frozenset({0, 1, 2})}
    assert c.test == {1: frozenset({0, 1}), 2: frozenset({0})}
    assert c.pairs == ((1, 0), (1, 1), (2, 0))


@pytest.mark.parametrize("fold", [1, 2])
def test_the_training_profile_is_the_lock_mean_contrast(fold):
    legs, paths = _hand()
    c = T.cells(paths)
    got = L.training_profile(legs, paths, fold, c)
    expected, held = _expected_profile(legs, paths, fold, c.train[fold], "train", (0, 1, 2, 3))
    np.testing.assert_allclose(got.to_numpy(), expected.to_numpy(), rtol=1e-12, atol=0)
    assert got.loc[3].isna().all()  # state 3 does not qualify: NaN row
    # Σ_k n_k D_ik = 0 on the training window (§12.5 step 3)
    n = held.value_counts().reindex([0, 1, 2]).to_numpy(float)
    np.testing.assert_allclose(n @ got.loc[[0, 1, 2]].to_numpy(), 0.0, atol=1e-9)
    assert np.isnan(held.iloc[0])  # the first training session holds no state


def test_the_map_is_row_centred_with_unit_rms_and_stays_occupancy_neutral():
    legs, paths = _hand()
    c = T.cells(paths)
    d = L.training_profile(legs, paths, 1, c).loc[[0, 1, 2]]
    m = L.estimate_m(legs, paths, 1, c)
    centred = d.sub(d.mean(axis=1), axis=0)
    rms = np.sqrt((centred.to_numpy() ** 2).mean())
    np.testing.assert_allclose(m.loc[[0, 1, 2]].to_numpy(), (centred / rms).to_numpy(),
                               rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(m.loc[[0, 1, 2]].sum(axis=1), 0.0, atol=1e-12)
    assert np.sqrt(np.mean(m.loc[[0, 1, 2]].to_numpy() ** 2)) == pytest.approx(1.0, abs=1e-12)
    assert m.loc[3].isna().all()
    held = paths.xs(1, level="fold").droplevel("segment").shift(1)
    held = held.loc[paths.xs((1, "train"), level=("fold", "segment")).index]
    n = held.value_counts().reindex([0, 1, 2]).to_numpy(float)
    np.testing.assert_allclose(n @ m.loc[[0, 1, 2]].to_numpy(), 0.0, atol=1e-9)
    table = protocol.tilt_table(m, d=0.5)
    np.testing.assert_allclose(table.sum(axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(table.loc[3], 0.1)  # a NaN row holds 1/10 (step 6)
    raw = ((1 + 0.5 * m.loc[0]) / 10).clip(lower=0)
    np.testing.assert_allclose(table.loc[0], raw / raw.sum(), rtol=1e-12)


def test_the_map_ignores_any_static_shift_of_a_leg():
    """Static neutralisation (§12.5 steps 2-3): a constant added to a leg on every session
    moves no D and no m — only the state profile is left."""
    legs, paths = _hand()
    c = T.cells(paths)
    shifted = legs + np.linspace(-0.01, 0.02, len(LIBRARY))
    for fold in (1, 2):
        pd.testing.assert_frame_equal(L.estimate_m(shifted, paths, fold, c),
                                      L.estimate_m(legs, paths, fold, c), rtol=1e-9)


def test_a_fold_without_a_qualifying_state_or_with_zero_rms_holds_the_control():
    legs, paths = _hand()
    c = T.cells(paths)
    none = dataclasses.replace(c, train={1: frozenset(), 2: c.train[2]})
    assert L.estimate_m(legs, paths, 1, none) is None
    assert L.estimate_m(legs, paths, 2, none) is not None
    rows = np.array([True, True, False, True])
    flat = np.array([[0.5] * 10, [-0.25] * 10, [np.nan] * 10, [2.0] * 10])
    assert L._tilt_map(flat, rows) is None  # signals alike in every state: RMS exactly 0
    assert L._tilt_map(flat, np.zeros(4, dtype=bool)) is None
    nan_rms = flat.copy()
    nan_rms[0, 0] = np.nan
    assert np.isnan(L._tilt_map(nan_rms, rows)[rows]).all()  # NaN is not zero: not finite


def test_the_test_profile_is_computed_identically_on_the_test_cells():
    legs, paths = _hand()
    c = T.cells(paths)
    got = L.test_profile(legs, paths, 1, c)
    expected, held = _expected_profile(legs, paths, 1, c.test[1], "test", (0, 1, 2, 3))
    np.testing.assert_allclose(got.to_numpy(), expected.to_numpy(), rtol=1e-12, atol=0)
    # the first test session holds the label stamped on the last training session
    assert held.iloc[0] == paths.xs((1, "train"), level=("fold", "segment")).iloc[-1]
    # one qualifying test cell: its profile is exactly zero, so rho = 0 (§12.7)
    single = L.test_profile(legs, paths, 2, c)
    assert (single.loc[0] == 0.0).all() and single.loc[1].isna().all()
    rt = L.rank_transfer(legs, paths, c)
    assert rt.rho[(2, 0)] == 0.0
    assert set(rt.rho) == set(c.pairs)


def test_spearman_takes_average_ranks_and_reads_zero_on_zero_rank_variance():
    a = np.array([1.0, 2.0, 2.0, 3.0, 5.0])
    b = np.array([2.0, 1.0, 4.0, 3.0, 5.0])
    assert L._spearman(a, b) == pytest.approx(stats.spearmanr(a, b).statistic, abs=0)
    ranks = stats.rankdata(a)  # average ranks: [1, 2.5, 2.5, 4, 5]
    assert L._spearman(a, b) == pytest.approx(np.corrcoef(ranks, stats.rankdata(b))[0, 1])
    assert L._spearman(np.zeros(5), b) == 0.0
    assert L._spearman(a, np.full(5, 7.0)) == 0.0
    assert np.isnan(L._spearman(np.r_[np.nan, a[1:]], b))


def test_R_is_the_plain_mean_over_the_qualifying_pairs(data, primary, cells):
    rt = L.rank_transfer(data.legs, primary.paths, cells)
    assert tuple(rt.rho) == cells.pairs
    assert float(np.mean([rt.rho[p] for p in cells.pairs])) == rt.R
    for fold, value in rt.per_fold.items():
        assert value == pytest.approx(np.mean([v for (f, _), v in rt.rho.items() if f == fold]))
    assert all(-1.0 <= v <= 1.0 for v in rt.rho.values())


# ------------------------------------------------------------------------- the selector


def test_the_selector_trades_each_fold_s_tilt_lagged_and_equal_weight_before(data, primary,
                                                                             cells):
    sel = L.selector_mix(data, primary.paths)
    mix = sel.traded.mix
    before = mix.index < data.folds[0].test_start
    np.testing.assert_array_equal(mix.loc[before].to_numpy(), 0.1)
    np.testing.assert_allclose(mix.sum(axis=1), 1.0, atol=1e-12)
    for fold in data.folds:
        m = L.estimate_m(data.legs, primary.paths, fold.number, cells)
        pd.testing.assert_frame_equal(sel.maps[fold.number], m)
        table = sel.tables[fold.number]
        pd.testing.assert_frame_equal(table, protocol.tilt_table(m, d=0.5))
        test = fold.test(data.sessions)
        held = T.lagged_path(primary.paths, fold.number).loc[test]
        for t in test[::97]:
            expected = table.loc[int(held.loc[t])] if held.loc[t] in cells.train[fold.number] \
                else pd.Series(0.1, index=list(LIBRARY))
            np.testing.assert_allclose(mix.loc[t].to_numpy(), expected.to_numpy(), atol=0)
    assert sel.fallback_sessions == cells.abstaining
    assert repr(sel) == (f"Selector(5 folds, 0 at the control, {cells.abstaining} "
                         "fallback test sessions, finite=True)")  # counts only


def test_abstaining_states_and_control_folds_are_the_fallback(data, primary, cells):
    layout = L._layout(data.legs, primary.paths)
    states = L._states(cells)
    dropped = min(cells.train[1])
    train = dict(cells.train)
    train[1] = cells.train[1] - {dropped}
    train[2] = frozenset()
    sel = L._selector(data, primary.paths, layout, train, states, 0.5)
    assert sel.maps[2] is None and sel.control_folds == 1
    held = T.lagged_path(primary.paths, 1).loc[data.folds[0].test(data.sessions)]
    expected = int((held == dropped).sum()) + len(data.folds[1].test(data.sessions))
    assert sel.fallback_sessions == expected
    flagged = sel.traded.fallback[sel.traded.fallback].index
    np.testing.assert_array_equal(sel.traded.mix.loc[flagged].to_numpy(), 0.1)


def test_the_d_rows_share_the_primary_map_and_read_a1_only(data, primary):
    base = L.selector_mix(data, primary.paths, T.PRIMARY)
    for variant in (T.D025, T.D100):
        tilted = L.selector_mix(data, primary.paths, variant)
        for f in base.maps:
            pd.testing.assert_frame_equal(tilted.maps[f], base.maps[f])
            pd.testing.assert_frame_equal(tilted.tables[f],
                                          protocol.tilt_table(base.maps[f], d=variant.d))
    out = L.instrument(data, T.D025, 4, boot_draws=50, partition=primary, workers=1)
    assert out["d"] == 0.25 and "A-2" not in out and "nfci" not in out
    assert L.section_name(T.PRIMARY) == "A" and L.section_name(T.D025) == "A:d=0.25"
    with pytest.raises(ValueError, match="not committed"):
        L.section_name(T.PIT18)
    assert L.pit_variant(T.PRIMARY) is T.PIT18
    for variant in (T.K3, T.SMOOTH21, T.D100):
        pit = L.pit_variant(variant)
        assert (pit.K, pit.smoothing, pit.d) == (variant.K, variant.smoothing, variant.d)
        assert pit.features == T.PIT_FEATURES and pit.role == "control"


# ----------------------------------------------------------------------- the instrument

TOP = {"lock", "variant", "K", "d", "smoothing", "features", "placebo", "bootstrap", "counts",
       "A-1", "A-2", "nfci", "undecidable"}
A1 = {"se", "mde_0.05", "mde_0.05/6", "T_A1", "se_star", "block_star", "standalone_threshold",
      "turnover", "kill", "sigma", "cap_share", "missing_sessions"}
A2 = {"cells", "q50", "q95", "q99", "q99-q50"}
COUNTS = {"test_sessions", "cells_possible", "train_cells", "abstaining_rows", "test_cells",
          "a2_cells", "a2_cells_per_fold", "cell_floor", "control_folds", "fallback_sessions",
          "fallback_share", "map_finite"}
NFCI = {"features", "test_sessions", "agree_sessions", "agreement", "agreement_per_fold",
        "transitions_per_year", "eta2_log_rv"}


def test_the_instrument_reads_no_sharpe_no_beta_and_not_the_real_R(data, primary, monkeypatch):
    """§13.1 step 2: the instrument prints only the allow-list, and never computes a
    Sharpe, a beta, a placebo p or percentile, or R on the real partition."""

    def refuse(*args, **kwargs):
        raise AssertionError("the instrument computed a return statistic")

    for owner, name in ((T, "sharpe"), (T, "pair_reading"), (protocol, "realised_beta"),
                        (protocol, "placebo_p_value"), (protocol, "placebo_percentile"),
                        (L, "rank_transfer")):
        monkeypatch.setattr(owner, name, refuse)
    real = primary.paths.to_numpy(float)
    calls: list[bool] = []
    original = L._transfer

    def spy(layout, labels, cells_, states):
        calls.append(bool(np.array_equal(labels, real, equal_nan=True)))
        return original(layout, labels, cells_, states)

    monkeypatch.setattr(L, "_transfer", spy)
    out = L.instrument(data, T.PRIMARY, 8, boot_draws=50, partition=primary, workers=1)
    assert len(calls) == 8 and not any(calls)
    assert set(out) == TOP
    assert set(out["A-1"]) == A1 and set(out["A-2"]) == A2
    assert set(out["counts"]) == COUNTS and set(out["nfci"]) == NFCI
    json.dumps(out, allow_nan=False)  # plain JSON, nothing non-finite as a number
    T.encode_thresholds(out)


def test_the_instrument_is_the_core_threshold_the_kill_and_the_null(data, primary, cells):
    out = L.instrument(data, T.PRIMARY, 8, boot_draws=50, partition=primary, workers=1)
    sel = L.selector_mix(data, primary.paths)
    book, control = T.build_book(data, sel.traded.mix), T.control_book(data)
    pair = T.pair_threshold(book.net5, control.net5, draws=50, seed=0)
    a1 = out["A-1"]
    assert a1["T_A1"] == pair.threshold == max(0.338, *pair.mde_corrected.values())
    assert a1["se_star"] == pair.se[a1["block_star"]]
    assert a1["mde_0.05/6"] == {str(b): v for b, v in pair.mde_corrected.items()}
    kill = T.kill(book.turnover - control.turnover, pair.threshold, book.sigma)
    assert a1["kill"] == {"k_kill": kill.k_kill, "fires": kill.fires}
    assert kill.k_kill == min(12.0, pair.threshold * book.sigma * 10_000 / 20)
    assert a1["turnover"] == {"selector": book.turnover, "control": control.turnover,
                              "delta": book.turnover - control.turnover}
    assert a1["sigma"] == {"selector": book.sigma, "control": control.sigma}
    years = sum(f.test_years for f in data.folds)
    assert a1["standalone_threshold"] == protocol.standalone_sharpe_threshold(
        years, alpha=0.05 / 6)
    counts = out["counts"]
    assert counts["a2_cells"] == cells.n_pairs and counts["train_cells"] == cells.n_train
    assert counts["abstaining_rows"] == 20 - cells.n_train
    assert counts["fallback_sessions"] == sel.fallback_sessions
    assert counts["test_sessions"] == len(data.test_sessions)
    draws = T.placebo_draws(primary.paths, n=8)
    null = [L.rank_transfer(data.legs, draws.draw(j), cells).R for j in range(8)]
    summary = T.null_summary(null)
    assert out["A-2"] == {"cells": cells.n_pairs, "q50": summary.q50, "q95": summary.q95,
                          "q99": summary.q99, "q99-q50": summary.resolution}
    pit = T.partition_paths(data, T.PIT18)
    assert out["nfci"] == T.printable(L.nfci_diagnostic(data, primary.paths, pit, 4))
    assert out["undecidable"] == {"A-1": [], "A-2": []}


def test_the_instrument_is_bitwise_deterministic_and_independent_of_the_workers(data, primary):
    one = L.instrument(data, T.PRIMARY, 6, boot_draws=50, partition=primary, workers=1)
    again = L.instrument(data, T.PRIMARY, 6, boot_draws=50, partition=primary, workers=1)
    pooled = L.instrument(data, T.PRIMARY, 6, boot_draws=50, partition=primary, workers=2)
    assert json.dumps(one, sort_keys=True) == json.dumps(again, sort_keys=True)
    assert json.dumps(one, sort_keys=True) == json.dumps(pooled, sort_keys=True)
    cells_ = T.cells(primary.paths)
    shared = (data, L._layout(data.legs, primary.paths), T.placebo_draws(primary.paths, n=3),
              cells_.train, L._states(cells_), 0.5, 0.0)
    assert T.draw_map(L._arm_job, 3, shared, workers=1) == \
        T.draw_map(L._arm_job, 3, shared, workers=2)


def test_the_instrument_refuses_foreign_draws_or_a_foreign_partition(data, primary):
    with pytest.raises(ValueError, match="seed 0"):
        L.instrument(data, T.PRIMARY, T.placebo_draws(primary.paths, n=2, seed=1),
                     boot_draws=50, partition=primary, workers=1)
    with pytest.raises(ValueError, match="not K=3"):
        L.instrument(data, T.K3, 2, boot_draws=50, partition=primary, workers=1)


@pytest.mark.parametrize("variant", [T.K3, T.SMOOTH21])
def test_a_sensitivity_instrument_carries_its_own_cells_and_no_nfci(data, variant):
    out = L.instrument(data, variant, 4, boot_draws=50, workers=1)
    assert set(out) == TOP - {"nfci"}
    assert out["K"] == variant.K and out["smoothing"] == int(variant.smoothing or 0)
    assert out["counts"]["cells_possible"] == 5 * variant.K
    assert out["counts"]["cell_floor"] == 5 * variant.K / 2
    assert out["placebo"]["exact"] is True


def test_the_nfci_diagnostic_counts_agreement_after_alignment(data, primary):
    same = L.nfci_diagnostic(data, primary.paths, primary, 4)
    assert same["agreement"] == 1.0 and same["agree_sessions"] == len(data.test_sessions)
    assert set(same["agreement_per_fold"].values()) == {1.0}
    n = sum(transitions(primary.paths.xs((f.number, "test"), level=("fold", "segment")))
            for f in data.folds)
    assert same["transitions_per_year"] == n / sum(f.test_years for f in data.folds)
    panel = T.variant_panel(data, T.PIT18)
    assert same["eta2_log_rv"] == eta_squared(primary.wf.aligned, panel[LOG_RV])
    relabelled = T.Partition(primary.paths.map({0: 2, 1: 3, 2: 0, 3: 1}), primary.wf)
    assert L.nfci_diagnostic(data, primary.paths, relabelled, 4)["agreement"] == 1.0
    flipped = primary.paths.copy()
    rows = flipped.index.get_locs([2, "test"])[:50]
    flipped.iloc[rows] = (flipped.iloc[rows] + 1) % 4
    changed = L.nfci_diagnostic(data, primary.paths, T.Partition(flipped, primary.wf), 4)
    assert changed["agree_sessions"] == len(data.test_sessions) - 50
    assert changed["agreement_per_fold"]["2"] < 1.0


# -------------------------------------------------------------------------- the reading


def test_a_planted_state_effect_is_found_by_both_locks(planted_reading):
    a1, a2 = planted_reading["A-1"], planted_reading["A-2"]
    assert a1["verdict_lines"] == T.PASS and a1["verdict"] == T.PASS
    assert a1["delta"]["5"] > a1["T_A1"] and a1["p"] <= T.BONFERRONI
    assert a1["diff_pct"] >= 0.95 and a1["beta_pct"] < 0.95 and a1["delta_w"] < a1["delta"]["5"]
    assert a2["verdict_lines"] == T.PASS and a2["verdict"] == T.PASS
    assert a2["R"] > a2["q99"] and a2["p"] == 1 / (PLANTED_DRAWS + 1) and a2["R_W"] < a2["R"]
    assert planted_reading["verdicts"] == {"A-1": T.PASS, "A-2": T.PASS, "A": T.PASS}
    assert planted_reading["level_verdict"] == T.PASS
    # a would-be PASS is rebuilt on the 18 features, with its own draws and thresholds
    assert a1["pit"]["verdict_lines"] == T.PASS and a2["pit"]["verdict_lines"] == T.PASS
    assert planted_reading["pit_instrument"]["features"] == 18
    assert not planted_reading["lock_constants"]
    assert planted_reading["holm_p"] == {"A-1": a1["p"], "A-2": a2["p"]}


def test_no_effect_is_not_found(null_reading):
    a1, a2 = null_reading["A-1"], null_reading["A-2"]
    assert a1["verdict"] != T.PASS and a2["verdict"] == T.NOT_SHOWN and a2["p"] > 0.01
    assert null_reading["verdicts"]["A"] != T.PASS
    assert null_reading["pit_instrument"] is None and a1["pit"] is None and a2["pit"] is None


def test_a_volatility_effect_is_credited_to_the_witness(data, primary):
    w1 = T.witness_paths(data, "W1", 4)
    keyed = plant(data, _fold5(w1), _effect(), SCALE)
    run = L._run(keyed, T.PRIMARY, 30, boot_draws=BOOT, partition=primary, workers=1)
    reading = L._read(keyed, run, workers=1)
    assert reading["A-1"]["delta_w"] >= reading["A-1"]["delta"]["5"]
    assert reading["A-2"]["R_W"] >= reading["A-2"]["R"]
    assert reading["A-1"]["verdict"] != T.PASS and reading["A-2"]["verdict"] != T.PASS


def test_the_pit_rebuild_downgrades_a_lock_it_does_not_reproduce(planted, planted_run):
    """§8 control 5 wired end to end: replace the 18-feature partition by a partition
    without the planted content (the W1 bins) and both would-be PASSes are DOWNGRADED."""
    w1 = T.witness_paths(planted, "W1", 4)
    swapped = dataclasses.replace(
        planted_run, pit_partition=T.Partition(w1, planted_run.pit_partition.wf))
    reading = L._read(planted, swapped, workers=1)
    for lock in ("A-1", "A-2"):
        assert reading[lock]["verdict_lines"] == T.PASS
        assert reading[lock]["pit"]["verdict_lines"] != T.PASS
        assert reading[lock]["verdict"] == T.DOWNGRADED_PIT
    assert reading["verdicts"]["A"] == T.DOWNGRADED_PIT


def test_a_placebo_arm_is_the_selector_rebuilt_on_the_draw(data, primary, cells):
    draws = T.placebo_draws(primary.paths, n=3)
    control = T.control_book(data)
    control_sharpe = T.sharpe(control.net5)
    shared = (data, L._layout(data.legs, primary.paths), draws, cells.train,
              L._states(cells), 0.5, control_sharpe)
    for j in range(3):
        sel = L.selector_mix(data, draws.draw(j))
        book = T.build_book(data, sel.traded.mix)
        assert L._arm_job(shared, j) == (T.sharpe(book.net5) - control_sharpe, book.beta())
        assert not sel.maps[3].equals(L.selector_mix(data, primary.paths).maps[3])
        assert L._null_job(shared[1:3] + (cells, L._states(cells)), j) == \
            L.rank_transfer(data.legs, draws.draw(j), cells).R


def test_perturbing_the_data_after_a_date_changes_nothing_before_it(data, primary):
    fold3 = data.folds[2]
    test3 = fold3.test(data.sessions)
    cut = test3[len(test3) // 2]
    rng = np.random.default_rng(11)
    after = data.sessions > cut
    excess = data.excess.copy()
    excess.loc[after] += rng.normal(0.0, 0.01, (int(after.sum()), excess.shape[1]))
    features = data.features.copy()
    later = features.index > cut
    features.loc[later] += rng.normal(0.0, 1.0, (int(later.sum()), features.shape[1]))
    changed = dataclasses.replace(
        data, excess=excess, returns=excess.add(data.rf, axis=0),
        legs=T.signal_legs(data.signals, excess), features=features,
        context=context_panel(features, data.log_rv_raw))
    part = T.partition_paths(changed, T.PRIMARY)
    old, new = L.selector_mix(data, primary.paths), L.selector_mix(changed, part.paths)
    for fold in (1, 2, 3):
        pd.testing.assert_frame_equal(new.maps[fold], old.maps[fold], check_exact=True)
    assert not new.maps[5].equals(old.maps[5])
    upto = data.sessions <= cut
    pd.testing.assert_frame_equal(new.traded.mix.loc[upto], old.traded.mix.loc[upto],
                                  check_exact=True)
    assert not new.traded.mix.loc[~upto].equals(old.traded.mix.loc[~upto])
    old_net = T.build_book(data, old.traded.mix).net_full[5.0]
    new_net = T.build_book(changed, new.traded.mix).net_full[5.0]
    np.testing.assert_allclose(new_net.loc[upto], old_net.loc[upto], rtol=0, atol=1e-15)


# ------------------------------------------------------------------- non-finite readings


def test_a_non_finite_leg_makes_both_locks_undecidable_before_their_reading(data, primary):
    legs = data.legs.copy()
    legs.loc[data.folds[0].train(data.sessions)[100], "IND_SKEW"] = np.nan
    broken = dataclasses.replace(data, legs=legs)
    run = L._run(broken, T.PRIMARY, 6, boot_draws=50, partition=primary, workers=1)
    assert run.printout["counts"]["map_finite"] is False
    assert run.printout["A-2"]["q50"] == T.NON_FINITE
    pre = run.printout["undecidable"]
    assert any("map m is not finite" in r for r in pre["A-1"])
    assert any("placebo draws of R are not finite" in r for r in pre["A-2"])
    reading = L._read(broken, run, workers=1)
    assert reading["verdicts"] == {"A-1": T.UNDECIDABLE, "A-2": T.UNDECIDABLE,
                                   "A": T.UNDECIDABLE}
    assert not reading["A-1"]["read"] and not reading["A-2"]["read"]
    assert reading["A-1"]["undecidable_reasons"] == pre["A-1"]
    # §13.4: a lock UNDECIDABLE before its reading is not read and spends no trial row
    assert L.trial_rows(reading, T.PRIMARY) == {} and reading["trial_rows"] == {}
    assert L.trial_rows(reading, T.K3) == {} and L.trial_rows(reading, T.D025) == {}


def test_a_undecidable_during_the_reading_is_logged_with_its_cause(null_reading, tmp_path):
    reading = json.loads(json.dumps(T.plain(null_reading)))
    reading["A-1"]["verdict_inputs"]["delta_w"] = float("nan")
    reading["A-1"]["undecidable_reasons"] = ["the witness map is not finite"]
    verdicts = L.lock_verdicts(reading)
    assert verdicts["A-1"] == T.UNDECIDABLE and verdicts["A"] == T.UNDECIDABLE
    rows = L.trial_rows(reading, T.PRIMARY, verdicts)
    assert rows["A-1"]["verdict"] == T.UNDECIDABLE
    assert rows["A-1"]["note"] == "UNDECIDABLE: the witness map is not finite"
    for test, row in rows.items():
        T.log_trial(test, T.PRIMARY, **row, path=tmp_path / "trials.parquet")


def test_a_missing_return_on_a_test_session_is_undecidable(data, primary):
    excess = data.excess.copy()
    excess.loc[data.test_sessions[40], excess.columns[0]] = np.nan
    broken = dataclasses.replace(data, excess=excess)
    run = L._run(broken, T.PRIMARY, 4, boot_draws=50, partition=primary, workers=1)
    a1 = run.printout["A-1"]
    assert a1["T_A1"] == T.NON_FINITE
    assert a1["kill"] == {"k_kill": T.NON_FINITE, "fires": T.NON_FINITE}
    assert a1["missing_sessions"]["selector"] > 0
    T.encode_thresholds(run.printout)  # recorded as non-finite, never as a number
    assert any("paired leg is missing" in r for r in run.printout["undecidable"]["A-1"])
    reading = L._read(broken, run, workers=1)
    assert reading["verdicts"]["A-1"] == T.UNDECIDABLE and not reading["A-1"]["read"]
    assert set(reading["trial_rows"]) <= {"A-2"}


def test_the_final_holm_step_reads_from_the_stored_inputs(planted_reading):
    stored = json.loads(json.dumps(T.plain(planted_reading)))
    assert L.lock_verdicts(stored) == planted_reading["verdicts"]
    held = L.lock_verdicts(stored, {"A-1", "A-2"})
    assert held == {"A-1": T.PASS, "A-2": T.PASS, "A": T.PASS}
    assert L.lock_verdicts(stored, set())["A-1"] == T.UNDERPOWERED
    assert L.lock_verdicts(stored, set())["A-2"] == T.NOT_SHOWN
    with pytest.raises(ValueError, match="six primaries"):
        L.lock_verdicts({**stored, "role": "sensitivity"}, {"A-1"})


# ------------------------------------------------------------------------ verdict lines

A1_PASS = dict(delta5=0.5, delta10=0.4, threshold=0.338, p=0.001, kill_fires=False,
               beta_pct=0.5, diff_pct=0.99, delta_w=0.1, fallback_share=0.07,
               leg_missing=False)


@pytest.mark.parametrize(("change", "verdict"), [
    ({}, T.PASS),
    ({"delta5": np.nan}, T.UNDECIDABLE),
    ({"delta10": np.inf}, T.UNDECIDABLE),
    ({"diff_pct": np.nan}, T.UNDECIDABLE),  # one placebo draw not finite
    ({"beta_pct": np.nan, "kill_fires": True}, T.UNDECIDABLE),
    ({"kill_fires": None}, T.UNDECIDABLE),
    ({"leg_missing": True, "delta5": -1.0}, T.UNDECIDABLE),
    ({"fallback_share": 0.5001}, T.UNDECIDABLE),
    ({"fallback_share": 0.5}, T.PASS),
    ({"undecidable": True, "kill_fires": True}, T.UNDECIDABLE),
    ({"kill_fires": True, "delta5": -1.0}, T.FAIL_COST),  # line 2 before line 3
    ({"delta5": 0.0}, T.FAIL),
    ({"delta5": -0.2, "delta10": -0.3}, T.FAIL),
    ({"delta10": 0.0}, T.UNDECIDED),
    ({"delta5": 0.1, "delta10": -0.1}, T.UNDECIDED),  # line 4 before line 5
    ({"delta5": 0.3379}, T.UNDERPOWERED),
    ({"delta5": 0.338}, T.PASS),  # at the bar
    ({"p": 0.0084}, T.UNDERPOWERED),  # not rejected at 0.05/6
    ({"p": 0.0084, "holm_rejected": True}, T.PASS),  # rejected at the final Holm step
    ({"holm_rejected": False}, T.UNDERPOWERED),
    ({"beta_pct": 0.95}, T.DOWNGRADED_BETA),
    ({"beta_pct": 0.95, "diff_pct": 0.1}, T.DOWNGRADED_BETA),  # line 6 before line 7
    ({"diff_pct": 0.9499}, T.DOWNGRADED_NOT_CONDITIONAL),
    ({"diff_pct": 0.5, "delta_w": 9.0}, T.DOWNGRADED_NOT_CONDITIONAL),  # 7 before 8
    ({"delta_w": 0.5}, T.DOMINATED_VOLATILITY),  # Δ_W ≥ Δ, ties included
])
def test_a1_lines_apply_in_the_lock_order(change, verdict):
    assert L.a1_lines(**{**A1_PASS, **change}) == verdict


def test_the_a1_pit_line_downgrades_only_a_would_be_pass():
    assert L.a1_verdict(**A1_PASS, rebuild=T.PASS) == T.PASS
    for rebuild in (T.UNDERPOWERED, T.UNDECIDABLE, T.DOMINATED_VOLATILITY, T.FAIL):
        assert L.a1_verdict(**A1_PASS, rebuild=rebuild) == T.DOWNGRADED_PIT
    with pytest.raises(ValueError, match="PIT rebuild"):
        L.a1_verdict(**A1_PASS)
    assert L.a1_verdict(**{**A1_PASS, "delta5": -1.0}) == T.FAIL  # no rebuild needed


A2_PASS = dict(R=0.5, p=0.001, exact=True, R_W=0.1, cells=14, floor=10.0)


@pytest.mark.parametrize(("change", "verdict"), [
    ({}, T.PASS),
    ({"R": np.nan}, T.UNDECIDABLE),
    ({"p": np.nan}, T.UNDECIDABLE),  # any R^(j) not finite
    ({"exact": False}, T.UNDECIDABLE),
    ({"cells": 9}, T.UNDECIDABLE),
    ({"cells": 10}, T.PASS),
    ({"cells": 7, "floor": 7.5}, T.UNDECIDABLE),  # K = 3
    ({"R_W": np.nan}, T.UNDECIDABLE),
    ({"undecidable": True}, T.UNDECIDABLE),
    ({"p": 0.0099}, T.NOT_SHOWN),  # p ≤ 0.01 but not rejected at 0.05/6
    ({"p": 0.0099, "holm_rejected": True}, T.PASS),
    ({"p": 0.0101, "holm_rejected": True}, T.NOT_SHOWN),
    ({"holm_rejected": False}, T.NOT_SHOWN),
    ({"p": 0.5, "R_W": 0.9}, T.NOT_SHOWN),  # not holding comes before dominated
    ({"R_W": 0.5}, T.DOMINATED),  # R_W ≥ R, ties included
    ({"R_W": 0.9}, T.DOMINATED),
])
def test_a2_lines_follow_12_7(change, verdict):
    assert L.a2_lines(**{**A2_PASS, **change}) == verdict


def test_a2_never_reads_fail_and_its_pit_clause_only_downgrades():
    for p in (0.001, 0.02, 0.5, 1.0):
        for r_w in (-0.5, 0.5, 0.9):
            for R in (-0.3, 0.0, 0.5):
                assert L.a2_lines(**{**A2_PASS, "p": p, "R_W": r_w, "R": R}) != T.FAIL
    assert L.a2_verdict(**A2_PASS, rebuild=T.PASS) == T.PASS
    assert L.a2_verdict(**A2_PASS, rebuild=T.NOT_SHOWN) == T.DOWNGRADED_PIT
    assert L.a2_verdict(**A2_PASS, rebuild=T.DOMINATED) == T.DOWNGRADED_PIT
    with pytest.raises(ValueError, match="PIT rebuild"):
        L.a2_verdict(**A2_PASS)


# --------------------------------------------------------------------------- trial rows


def test_trial_rows_follow_13_5_and_are_accepted_by_the_log(planted_reading, null_reading,
                                                            tmp_path):
    path = tmp_path / "trials.parquet"
    rows = L.trial_rows(planted_reading, T.PRIMARY)
    a1, a2 = planted_reading["A-1"], planted_reading["A-2"]
    assert set(rows) == {"A-1", "A-2"}
    assert rows["A-1"] == {"sharpe": a1["sharpe"]["selector"]["5"], "delta": a1["delta"]["5"],
                           "threshold": a1["T_A1"], "p": a1["p"], "placebo_pct": a1["diff_pct"],
                           "sessions": planted_reading["sessions"], "verdict": T.PASS,
                           "note": ""}
    assert np.isnan(rows["A-2"]["sharpe"])
    assert (rows["A-2"]["delta"], rows["A-2"]["threshold"], rows["A-2"]["p"]) == (
        a2["R"], a2["q99"], a2["p"])
    for test, row in rows.items():
        T.log_trial(test, T.PRIMARY, **row, path=path)
    level = L.trial_rows(planted_reading, T.K3)
    assert set(level) == {"A"} and level["A"]["p2"] == a2["p"]
    assert level["A"]["verdict"] == planted_reading["verdicts"]["A"]
    T.log_trial("A", T.K3, **level["A"], path=path)
    d_row = L.trial_rows(planted_reading, T.D025)
    assert set(d_row) == {"A-1"} and np.isnan(d_row["A-1"]["p2"])
    T.log_trial("A-1", T.D025, **d_row["A-1"], path=path)
    flagged = L.trial_rows(null_reading, T.SMOOTH21, primary_level=T.PASS)["A"]
    assert null_reading["A-1"]["delta"]["5"] <= 0 and "PASS (not robust)" in flagged["note"]


# -------------------------------------------------------------------------- the refusal

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


def test_the_reading_runs_only_on_committed_bitwise_thresholds(repo, data, primary):
    path = repo / T.THRESHOLDS_FILE
    kwargs = dict(boot_draws=50, partition=primary, workers=1, path=path, root=repo,
                  inputs=INPUTS)
    with pytest.raises(T.ReadingRefused, match="no threshold file"):
        L.read(data, T.PRIMARY, 6, **kwargs)
    printout = L.instrument(data, T.PRIMARY, 6, boot_draws=50, partition=primary, workers=1)
    T.write_thresholds(printout, section=L.section_name(T.PRIMARY), path=path, root=repo,
                       inputs=INPUTS)
    with pytest.raises(T.ReadingRefused, match="not tracked"):
        L.read(data, T.PRIMARY, 6, **kwargs)
    _git(repo, "add", T.THRESHOLDS_FILE)
    _git(repo, "commit", "-q", "-m", "thresholds")
    reading = L.read(data, T.PRIMARY, 6, **kwargs)
    assert reading["A-1"]["verdict"] in T.LOCK_VERDICTS
    assert reading["A-2"]["verdict"] in T.LOCK_VERDICTS
    assert reading["instrument"] == printout
    assert not reading["lock_constants"]
    stored = T.read_thresholds(path)["thresholds"]["A"]
    assert L.read(data, T.PRIMARY, 6, stored, **kwargs)["A-1"]["verdict"] == \
        reading["A-1"]["verdict"]
    with pytest.raises(T.ReadingRefused, match="thresholds passed differ"):
        L.read(data, T.PRIMARY, 6, {**stored, "K": 5}, **kwargs)
    with pytest.raises(T.ReadingRefused, match="bitwise"):
        L.read(data, T.PRIMARY, 6, **{**kwargs, "boot_draws": 51})
    with pytest.raises(T.ReadingRefused, match="no committed thresholds"):
        L.read(data, T.D025, 6, **kwargs)
    path.write_text(path.read_text().replace('"lock"', '"lock" ', 1))
    with pytest.raises(T.ReadingRefused, match="differs from its committed version"):
        L.read(data, T.PRIMARY, 6, **kwargs)

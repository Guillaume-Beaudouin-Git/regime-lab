"""Level B of the Two Sigma tree must compute §12.8, §12.9 and §8 control 5, and nothing else.

Every test runs on synthetic data from `tree.synthetic_tree_data` (49 industries, 10
signals, five one-year test folds); nothing here reads `data/`. The properties tested:
the lock's formulas on cases computed by hand; a planted state-dependent second moment is
detected by B-1 and a planted state risk-parity gain by B-2, a content-free partition by
neither, and a volatility-driven gain is beaten by W1 at both locks; the instrument
returns only what §13.1 step 2 allows and computes no G, Sharpe or
beta; nothing before a date moves when the data after it change; the same seed gives the
same bits whatever the worker count; a NaN reads UNDECIDABLE; every verdict line is
reachable in the lock's order; and the reading refuses to run unless its thresholds are
committed, unmodified and bitwise equal.
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
import statsmodels.api as sm
from scipy import stats

from regime_lab.analysis.bootstrap import stationary_indices
from regime_lab.config import ROOT
from regime_lab.selection import level_b as B
from regime_lab.selection import protocol
from regime_lab.selection import tree as T
from regime_lab.selection.library import LIBRARY

DRAWS = 120  # the fewest with 1/(n+1) <= 0.05/6, so that a B-1 PASS is reachable
BOOT = 200  # bootstrap draws where the bootstrap is not under test (the lock: 2,000)
PLANT = np.where((np.arange(10)[None, :] + np.arange(4)[:, None]) % 4 == 0, 2.5, 0.8)


# ------------------------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def data() -> T.TreeData:
    return T.synthetic_tree_data(seed=0, n_industries=49)


def _paths(data: T.TreeData, labels: pd.Series) -> pd.Series:
    """One label per session, laid out as `context.session_paths` (every fold shares it)."""
    pieces = []
    for fold in data.folds:
        for segment, dates in (("train", fold.train(data.sessions)),
                               ("test", fold.test(data.sessions))):
            index = pd.MultiIndex.from_arrays(
                [np.full(len(dates), fold.number), np.full(len(dates), segment), dates],
                names=["fold", "segment", "session"])
            pieces.append(pd.Series(labels.reindex(dates).to_numpy(), index=index))
    out = pd.concat(pieces).rename("state")
    return out if out.isna().any() else out.astype(np.int64)


def _markov(sessions: pd.DatetimeIndex, K: int, stay: float, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    s = np.zeros(len(sessions), dtype=np.int64)
    for t in range(1, len(sessions)):
        s[t] = s[t - 1] if rng.random() < stay else (s[t - 1] + rng.integers(1, K)) % K
    return pd.Series(s, index=sessions)


def _plant(data: T.TreeData, labels: pd.Series) -> T.TreeData:
    """Legs whose per-signal scale depends on the state: second-moment content, no mean."""
    return dataclasses.replace(data, legs=data.legs * PLANT[labels.to_numpy()])


@pytest.fixture(scope="module")
def regime(data: T.TreeData) -> pd.Series:
    return _markov(data.sessions, 4, 0.97, 7)


@pytest.fixture(scope="module")
def regime_paths(data: T.TreeData, regime: pd.Series) -> pd.Series:
    return _paths(data, regime)


@pytest.fixture(scope="module")
def kmeans(data: T.TreeData) -> pd.Series:
    return T.partition_paths(data).paths


def _b1(data: T.TreeData, paths: pd.Series, n: int = DRAWS) -> dict:
    """B-1 through the public functions, as the reading composes them (§12.8)."""
    draws = T.placebo_draws(paths, n, 0)
    cells = T.cells(paths)
    null = B.b1_null(data.legs, paths, draws, cells, workers=1)
    losses = B.forecast_losses(data.legs, paths, cells)
    g = B.g_statistic(losses)
    witnesses = {w: B.forecast_losses(data.legs, T.witness_paths(data, w, 4))
                 for w in B.WITNESSES}
    tests = B.witness_tests({w: witnesses[w].state - losses.state for w in witnesses},
                            draws=200)
    p = protocol.placebo_p_value(g, null)
    verdict = B.b1_verdict(statistic=g, p=p, exact=draws.exact, p_w1=tests["W1"].p,
                           p_w2=tests["W2"].p, pooled_pd=True, cells_ok=True,
                           fallback_share=losses.fallback_share)
    return {"G": g, "null": null, "p": p, "tests": tests, "verdict": verdict}


# ------------------------------------------------------------- the matrices, by hand


def test_positive_definiteness_is_the_lock_rule():
    assert B.is_positive_definite(np.eye(3))
    tiny = np.diag([1.0, 1.0, 2e-12])  # trace 2 + 2e-12: 2e-12 <= 1e-12 x trace
    assert not B.is_positive_definite(tiny)
    assert B.is_positive_definite(np.diag([1.0, 1.0, 3e-12]))
    assert not B.is_positive_definite(np.array([[1.0, np.nan], [np.nan, 1.0]]))
    assert not B.is_positive_definite(np.zeros((2, 2)))
    assert not B.is_positive_definite(np.ones(3))


def test_second_moments_are_about_zero_over_lagged_training_cells(data, regime_paths):
    cells = T.cells(regime_paths)
    for fold in data.folds:
        m = B.second_moments(data.legs, regime_paths, fold.number, cells)
        train = fold.train(data.sessions)
        lagged = T.lagged_path(regime_paths, fold.number).loc[train]
        x = data.legs.loc[train].to_numpy()
        has = lagged.notna().to_numpy()
        assert not has[0] and has[1:].all()
        pooled = sum(np.outer(v, v) for v in x[has]) / has.sum()
        np.testing.assert_allclose(m.pooled, pooled, rtol=1e-12, atol=0)
        assert m.n_pooled == len(train) - 1 and m.pooled_pd
        assert set(m.states) == set(cells.train[fold.number]) and not m.fallback_states
        for k, matrix in m.states.items():
            rows = x[(lagged == k).to_numpy()]
            np.testing.assert_allclose(matrix, rows.T @ rows / len(rows), rtol=1e-12)
            np.testing.assert_array_equal(matrix, matrix.T)
        assert m.matrix(np.nan) is m.pooled and m.matrix(99.0) is m.pooled
        assert m.matrix(float(min(m.states))) is m.states[min(m.states)]
        assert "SecondMoments(" in repr(m) and "[" not in repr(m)


def test_a_cell_that_is_not_positive_definite_falls_back_and_is_counted(data, regime,
                                                                         regime_paths):
    held = regime.shift(1)
    legs = data.legs.copy()
    legs.loc[(held == 2).to_numpy(), LIBRARY[0]] = 0.0
    m = B.second_moments(legs, regime_paths, 3)
    assert m.fallback_states == frozenset({2}) and 2 not in m.states and m.pooled_pd
    assert m.matrix(2.0) is m.pooled
    printout = B.instrument(dataclasses.replace(data, legs=legs), T.PRIMARY,
                            T.placebo_draws(regime_paths, 4), paths=regime_paths, workers=1,
                            boot_draws=BOOT)
    assert printout["matrices"]["cells_not_positive_definite"] == 5
    lagged_test = pd.concat([T.lagged_path(regime_paths, f.number).loc[f.test(data.sessions)]
                             for f in data.folds])
    assert printout["fallback"]["test_sessions"] == int((lagged_test == 2).sum())


def test_qlike_is_log_det_plus_the_mahalanobis_term():
    rng = np.random.default_rng(1)
    a = rng.normal(size=(4, 4))
    sigma = a @ a.T + 4 * np.eye(4)
    x = rng.normal(size=(7, 4))
    expected = np.log(np.linalg.det(sigma)) + np.einsum(
        "ij,jk,ik->i", x, np.linalg.inv(sigma), x)
    np.testing.assert_allclose(B.qlike(sigma, x), expected, rtol=1e-12)
    diag = np.diag([0.5, 2.0])
    assert B.qlike(diag, np.array([1.0, 2.0]))[0] == pytest.approx(np.log(1.0) + 2.0 + 2.0)
    stack = np.stack([sigma, 2 * sigma, sigma, np.zeros((4, 4)), sigma, sigma, sigma])
    stacked = B.qlike(stack, x)
    row = [B.qlike(s, v)[0] for s, v in zip(stack, x, strict=True)]
    np.testing.assert_allclose(stacked[[0, 1, 2, 4, 5, 6]], np.array(row)[[0, 1, 2, 4, 5, 6]],
                               rtol=1e-12)
    assert np.isnan(stacked[3]) and np.isnan(row[3])
    assert np.isnan(B.qlike(np.diag([1.0, -1.0]), np.ones(2))).all()
    assert np.isnan(B.qlike(np.eye(2), np.array([[1.0, np.nan]])))[0]


def test_g_is_the_mean_qlike_gain_over_every_test_session(data, regime_paths):
    cells = T.cells(regime_paths)
    losses = B.forecast_losses(data.legs, regime_paths, cells)
    assert losses.sessions.equals(data.test_sessions)

    def ell(s: np.ndarray, v: np.ndarray) -> float:
        return float(np.log(np.linalg.det(s)) + v @ np.linalg.inv(s) @ v)

    gains = []
    for fold in data.folds:
        train, test = fold.train(data.sessions), fold.test(data.sessions)
        lagged = T.lagged_path(regime_paths, fold.number)
        x = data.legs.loc[train].to_numpy()
        held = lagged.loc[train].to_numpy()
        pooled = x[np.isfinite(held)].T @ x[np.isfinite(held)] / np.isfinite(held).sum()
        per = {k: x[held == k].T @ x[held == k] / (held == k).sum()
               for k in cells.train[fold.number]}
        for t in test:
            v = data.legs.loc[t].to_numpy()
            k = lagged.loc[t]
            sigma = per.get(int(k), pooled) if np.isfinite(k) else pooled
            gains.append(ell(pooled, v) - ell(sigma, v))
    assert B.g_statistic(losses) == pytest.approx(np.mean(gains), rel=1e-9)
    np.testing.assert_allclose(losses.gain, gains, rtol=1e-8, atol=1e-10)
    assert (losses.gain[losses.fallback] == 0).all()


def test_the_lag_is_the_cores_one_session_inside_each_fold(data, kmeans):
    rows = B._fold_rows(data.legs, kmeans)
    labels = kmeans.to_numpy(float)
    for r in rows:
        core = T.lagged_path(kmeans, r.number).to_numpy()
        np.testing.assert_array_equal(B._lagged(labels[r.start:r.stop]), core)


def test_g_diag_keeps_the_state_variances_and_the_pooled_correlation():
    rng = np.random.default_rng(2)
    a, b = rng.normal(size=(5, 5)), rng.normal(size=(5, 5))
    pooled, state = a @ a.T + np.eye(5), b @ b.T + np.eye(5)
    forecast = B.diagonal_forecast(state, pooled)
    np.testing.assert_allclose(np.diag(forecast), np.diag(state), rtol=1e-12)
    sd = np.sqrt(np.diag(forecast))
    sd0 = np.sqrt(np.diag(pooled))
    np.testing.assert_allclose(forecast / np.outer(sd, sd), pooled / np.outer(sd0, sd0),
                               rtol=1e-12)


def test_the_correlation_share_is_the_locks_formula():
    rho, n = 0.035, 10
    corr = np.full((n, n), rho) + (1 - rho) * np.eye(n)
    sd = np.linspace(0.5, 3.0, n)
    assert B.correlation_share(corr * np.outer(sd, sd)) == pytest.approx(3.15 / 13.15)
    assert B.correlation_share(np.eye(n)) == 0.0
    assert np.isnan(B.correlation_share(np.diag([1.0, 0.0])))


def test_the_witness_p_is_the_larger_of_the_hac_and_bootstrap_readings():
    rng = np.random.default_rng(3)
    d = rng.normal(0.05, 1.0, 300) + 0.3 * np.sin(np.arange(300) / 9)
    got = B.witness_test(d, draws=200)
    se = {}
    for block in (21, 63, 126):
        r = np.random.default_rng(0)
        se[block] = np.std([d[stationary_indices(300, block, r)].mean() for _ in range(200)],
                           ddof=1)
    block = max(se, key=se.get)
    t = sm.OLS(d, np.ones(300)).fit(cov_type="HAC", cov_kwds={"maxlags": 6}).tvalues[0]
    assert got.block == block and got.se_boot[block] == pytest.approx(se[block], rel=1e-12)
    assert got.t_hac == pytest.approx(t, rel=1e-12)
    assert got.p_hac == pytest.approx(stats.norm.sf(t), rel=1e-12)
    assert got.p_boot == pytest.approx(stats.norm.sf(d.mean() / se[block]), rel=1e-12)
    assert got.p == max(got.p_hac, got.p_boot) and got.beaten == (got.p <= 0.05)
    pair = B.witness_tests({"a": d, "b": -d[::-1]}, draws=200)
    assert pair["a"] == got and pair["b"] == B.witness_test(-d[::-1], draws=200)
    broken = B.witness_tests({"a": d, "b": np.r_[d[:-1], np.nan]}, draws=50)
    assert np.isnan(broken["b"].p) and broken["b"].beaten is None
    assert broken["a"] == B.witness_test(d, draws=50)


# ------------------------------------------------------------------ the B-2 arms


def test_the_arms_are_risk_parity_per_fold_and_per_lagged_state(data, regime_paths):
    cells = T.cells(regime_paths)
    arms = B.risk_parity_arms(data, regime_paths, cells)
    first = data.test_sessions[0]
    before = data.sessions[data.sessions < first]
    np.testing.assert_allclose(arms.pooled_mix.loc[before].to_numpy(), 0.1)
    np.testing.assert_allclose(arms.state_mix.mix.loc[before].to_numpy(), 0.1)
    for fold in data.folds:
        m = B.second_moments(data.legs, regime_paths, fold.number, cells)
        pooled = protocol.risk_parity_mix(m.pooled)
        test = fold.test(data.sessions)
        np.testing.assert_allclose(arms.pooled_mix.loc[test].to_numpy(),
                                   np.tile(pooled, (len(test), 1)), rtol=1e-12)
        lagged = T.lagged_path(regime_paths, fold.number).loc[test]
        for t in test[:40]:
            expected = protocol.risk_parity_mix(m.matrix(lagged.loc[t]))
            np.testing.assert_allclose(arms.state_mix.mix.loc[t].to_numpy(), expected,
                                       rtol=1e-12)
    gross = arms.state_weights.abs().sum(axis=1)
    np.testing.assert_allclose(gross, arms.pooled_weights.abs().sum(axis=1), rtol=1e-12)
    expected = T.build_book(data, weights=arms.state_weights)
    pd.testing.assert_series_equal(arms.state.net5, expected.net5)
    losses = B.forecast_losses(data.legs, regime_paths, cells)
    np.testing.assert_array_equal(arms.state_mix.fallback.to_numpy(), losses.fallback)
    assert "RiskParityArms(" in repr(arms) and "sigma=" in repr(arms)


def test_the_instrument_prints_only_what_13_1_allows(data, regime_paths):
    draws = T.placebo_draws(regime_paths, 30, 0)
    printout = B.instrument(data, T.PRIMARY, draws, paths=regime_paths, workers=1,
                            boot_draws=BOOT)
    assert set(printout) == {"level", "variant", "K", "smoothing", "features",
                             "test_sessions", "cells", "matrices", "fallback", "placebo",
                             "bootstrap", "B-1", "B-2", "undecidable"}
    assert set(printout["B-1"]) == {"null", "non_finite_draws", "correlation_share_fold5"}
    assert set(printout["B-1"]["null"]) == {"q50", "q95", "q99", "q99-q50"}
    assert set(printout["B-2"]) == {"mde", "T_B2", "turnover", "kill", "sigma", "cap_share",
                                    "missing_test_sessions"}
    text = json.dumps(printout).lower()
    for forbidden in ("sharpe", "beta", "\"g\"", "p_b", "witness", "mean"):
        assert forbidden not in text
    T.encode_thresholds(printout)
    assert printout["undecidable"] == {"B-1": [], "B-2": []}
    cells = T.cells(regime_paths)
    assert printout["cells"]["qualifying_training"] == cells.n_train
    assert printout["test_sessions"] == len(data.test_sessions)

    arms = B.risk_parity_arms(data, regime_paths, cells)
    pair = T.pair_threshold(arms.state.net5, arms.pooled.net5, draws=BOOT)
    assert printout["bootstrap"] == {"method": "stationary", "draws": BOOT, "seed": 0,
                                     "blocks": [21, 63, 126]}
    assert printout["B-2"]["T_B2"] == pair.threshold == max(0.338, max(pair.mde_corrected.values()))
    assert printout["B-2"]["mde"]["se_star"] == pair.se_star
    delta_turnover = arms.state.turnover - arms.pooled.turnover
    assert printout["B-2"]["turnover"]["delta"] == delta_turnover
    k_kill = min(12.0, pair.threshold * arms.state.sigma * 10_000 / 20)
    assert printout["B-2"]["kill"]["k_kill"] == pytest.approx(k_kill, rel=1e-15)
    assert printout["B-2"]["kill"]["fires"] == (delta_turnover > k_kill)
    null = B.b1_null(data.legs, regime_paths, draws, cells, workers=1)
    assert printout["B-1"]["null"]["q99"] == np.percentile(null, 99)
    assert printout["B-1"]["correlation_share_fold5"] == B.correlation_share(
        B.second_moments(data.legs, regime_paths, 5, cells).pooled)


def test_the_instrument_reads_no_statistic_of_the_real_partition(data, regime_paths,
                                                                  monkeypatch):
    real = regime_paths.to_numpy(float)
    original = B._losses

    def guarded(rows, labels, train_cells, **kwargs):
        assert not np.array_equal(labels, real, equal_nan=True), "the real G was computed"
        return original(rows, labels, train_cells, **kwargs)

    def refuse(*args, **kwargs):
        raise AssertionError("the instrument computed a reading")

    monkeypatch.setattr(B, "_losses", guarded)
    for name in ("forecast_losses", "witness_tests", "placebo_arms"):
        monkeypatch.setattr(B, name, refuse)
    monkeypatch.setattr(T, "sharpe", refuse)
    monkeypatch.setattr(T, "pair_reading", refuse)
    monkeypatch.setattr(protocol, "realised_beta", refuse)
    monkeypatch.setattr(protocol, "paired_hac_t", refuse)
    printout = B.instrument(data, T.PRIMARY, T.placebo_draws(regime_paths, 6), paths=regime_paths,
                            workers=1, boot_draws=BOOT)
    assert printout["B-1"]["non_finite_draws"] == 0


def test_the_instrument_is_bitwise_deterministic_whatever_the_workers(data, regime_paths):
    draws = T.placebo_draws(regime_paths, 8, 0)
    one = B.instrument(data, T.PRIMARY, draws, paths=regime_paths, workers=1, boot_draws=BOOT)
    pooled = B.instrument(data, T.PRIMARY, draws, paths=regime_paths, workers=2,
                          boot_draws=BOOT)
    assert (json.dumps(T.encode_thresholds(one), sort_keys=True)
            == json.dumps(T.encode_thresholds(pooled), sort_keys=True))
    same = B.b1_null(data.legs, regime_paths, T.placebo_draws(regime_paths, 8, 0), workers=1)
    other = B.b1_null(data.legs, regime_paths, T.placebo_draws(regime_paths, 8, 1), workers=1)
    assert np.percentile(same, 99) == one["B-1"]["null"]["q99"]
    assert not np.array_equal(same, other)


def test_the_null_is_g_on_each_draw_through_the_same_code(data, regime_paths):
    draws = T.placebo_draws(regime_paths, 5, 0)
    null = B.b1_null(data.legs, regime_paths, draws, workers=1)
    for j in range(draws.n):
        g = B.g_statistic(B.forecast_losses(data.legs, draws.draw(j)))
        assert null[j] == g
    np.testing.assert_array_equal(null, B.b1_null(data.legs, regime_paths, draws, workers=2))


# --------------------------------------------------------------------- power sanity


def test_a_planted_second_moment_is_detected_and_beats_both_witnesses(data, regime,
                                                                      regime_paths):
    result = _b1(_plant(data, regime), regime_paths)
    assert result["G"] > result["null"].max()
    assert result["p"] == 1 / (DRAWS + 1) <= T.BONFERRONI
    assert result["tests"]["W1"].beaten and result["tests"]["W2"].beaten
    assert result["verdict"] == T.PASS


def test_a_content_free_partition_is_not_shown(data, regime_paths):
    result = _b1(data, regime_paths)
    assert result["p"] > 0.05
    assert result["verdict"] == T.NOT_SHOWN


def test_a_volatility_driven_gain_is_dominated_by_w1(data):
    bins = T.quantile_bins(data.log_rv, data.log_rv, 4).astype(np.int64)
    rng = np.random.default_rng(5)
    noisy = bins.copy()
    flip = rng.random(len(noisy)) < 0.25
    noisy[flip] = rng.integers(0, 4, int(flip.sum()))
    result = _b1(_plant(data, bins), _paths(data, noisy))
    assert result["p"] <= T.BONFERRONI
    assert result["tests"]["W1"].p > 0.05 and not result["tests"]["W1"].beaten
    assert result["verdict"] == T.DOMINATED


RP_PLANT = np.where((np.arange(10)[None, :] + np.arange(4)[:, None]) % 4 == 0, 3.0, 0.5)


def _pure(data: T.TreeData, labels: pd.Series, seed: int = 11) -> T.TreeData:
    """Signal *i* holds instrument *i* alone, so books trade exactly the legs. The ten
    legs share one positive mean and a per-signal scale that depends on the state: risk
    parity on the state's matrix earns more per unit of risk than on the pooled one."""
    rng = np.random.default_rng(seed)
    n, columns = len(data.sessions), data.excess.columns
    values = rng.normal(0.0, 0.008, (n, len(columns)))
    values[:, :10] = 0.0004 + 0.006 * RP_PLANT[labels.to_numpy()] * rng.normal(size=(n, 10))
    excess = pd.DataFrame(values, index=data.sessions, columns=columns)
    signals = {name: pd.DataFrame(np.tile(np.eye(len(columns))[i], (n, 1)), index=data.sessions,
                                  columns=columns) for i, name in enumerate(LIBRARY)}
    return dataclasses.replace(data, signals=signals, excess=excess,
                               legs=T.signal_legs(signals, excess))


def _b2(data: T.TreeData, paths: pd.Series, n: int = 40) -> dict:
    """B-2 through the composition :func:`B.read` uses, minus the git guard (§12.9)."""
    inst = B._build_instrument(data, T.PRIMARY, n, paths=paths, workers=1, boot_draws=BOOT)
    return B._evaluate(data, inst, workers=1, boot_draws=BOOT)["B-2"]


def test_a_planted_state_risk_parity_gain_passes_b2(data, regime, regime_paths):
    b2 = _b2(_pure(data, regime), regime_paths)
    got = b2["verdict_inputs"]
    assert got["delta5"] >= got["threshold"] >= T.THRESHOLD_FLOOR
    assert got["p"] <= T.BONFERRONI and got["kill_fires"] is False
    assert got["diff_pct"] >= T.PLACEBO_PCT_BAR > got["beta_pct"]
    assert got["delta_w"] < got["delta5"]
    assert b2["verdict_before_pit"] == T.PASS


def test_a_content_free_partition_does_not_pass_b2(data, regime):
    b2 = _b2(_pure(data, regime), _paths(data, _markov(data.sessions, 4, 0.97, 99)))
    got = b2["verdict_inputs"]
    assert got["delta5"] <= 0 and got["p"] == 1.0
    assert got["diff_pct"] < T.PLACEBO_PCT_BAR
    assert b2["verdict_before_pit"] == T.FAIL


def test_a_volatility_driven_b2_gain_is_beaten_by_the_w1_arm(data):
    bins = T.quantile_bins(data.log_rv, data.log_rv, 4).astype(np.int64)
    rng = np.random.default_rng(5)
    noisy = bins.copy()
    flip = rng.random(len(noisy)) < 0.25
    noisy[flip] = rng.integers(0, 4, int(flip.sum()))
    b2 = _b2(_pure(data, bins), _paths(data, noisy))
    got = b2["verdict_inputs"]
    assert got["delta_w"] >= got["delta5"] > 0
    # the flicker of the noisy partition costs turnover: line 2 applies before line 8
    assert got["kill_fires"] and b2["verdict_before_pit"] == T.FAIL_COST
    assert B.b2_verdict(**{**got, "kill_fires": False, "delta10": 0.1, "threshold": 0.1,
                           "p": 0.0, "beta_pct": 0.5}) == T.DOMINATED_VOLATILITY


# ------------------------------------------------------------------------- causality


def test_perturbing_the_data_after_a_date_changes_nothing_before_it(data, regime_paths):
    cut = data.folds[2].test(data.sessions)[100]
    after = data.sessions > cut
    rng = np.random.default_rng(9)
    excess = data.excess.copy()
    excess.loc[after] = excess.loc[after] * 3.0 + rng.normal(0, 0.01, (after.sum(), 49))
    legs = T.signal_legs(data.signals, excess)
    moved = dataclasses.replace(data, excess=excess, legs=legs)
    assert not np.allclose(legs.loc[after], data.legs.loc[after])

    for fold in data.folds[:3]:
        a = B.second_moments(data.legs, regime_paths, fold.number)
        b = B.second_moments(moved.legs, regime_paths, fold.number)
        np.testing.assert_array_equal(a.pooled, b.pooled)
        assert set(a.states) == set(b.states)
        for k in a.states:
            np.testing.assert_array_equal(a.states[k], b.states[k])
    base = B.forecast_losses(data.legs, regime_paths)
    shifted = B.forecast_losses(moved.legs, regime_paths)
    early = (base.sessions <= cut)
    np.testing.assert_array_equal(base.gain[early], shifted.gain[early])
    assert not np.array_equal(base.gain[~early], shifted.gain[~early])

    arms, arms_moved = B.risk_parity_arms(data, regime_paths), B.risk_parity_arms(moved,
                                                                                  regime_paths)
    upto = data.sessions <= cut
    pd.testing.assert_frame_equal(arms.state_mix.mix.loc[upto], arms_moved.state_mix.mix.loc[upto])
    for name in ("state", "pooled"):
        x = getattr(arms, name).net_full[5.0].loc[upto]
        y = getattr(arms_moved, name).net_full[5.0].loc[upto]
        pd.testing.assert_series_equal(x, y)
    w2 = T.witness_values(data, "W2").loc[upto]
    pd.testing.assert_series_equal(w2, T.witness_values(moved, "W2").loc[upto])


# ----------------------------------------------------------------------- non-finite


def test_a_missing_leg_or_a_singular_pooled_matrix_is_undecidable(data, regime_paths):
    draws = T.placebo_draws(regime_paths, 4)
    gap = data.legs.copy()
    gap.iloc[-3, 2] = np.nan
    printout = B.instrument(dataclasses.replace(data, legs=gap), T.PRIMARY, draws,
                            paths=regime_paths, workers=1, boot_draws=BOOT)
    assert any("missing" in r for r in printout["undecidable"]["B-1"])
    assert printout["B-1"]["non_finite_draws"] == 4
    assert printout["B-1"]["null"]["q99"] == B.NON_FINITE
    assert np.isnan(B.g_statistic(B.forecast_losses(gap, regime_paths)))

    flat = data.legs.copy()
    flat[LIBRARY[4]] = 0.0
    singular = dataclasses.replace(data, legs=flat)
    printout = B.instrument(singular, T.PRIMARY, draws, paths=regime_paths, workers=1,
                            boot_draws=BOOT)
    assert printout["matrices"]["pooled_positive_definite"] == [False] * 5
    assert printout["B-2"]["T_B2"] == B.NON_FINITE
    assert printout["B-2"]["kill"]["fires"] == B.NON_FINITE
    assert any("could not be built" in r for r in printout["undecidable"]["B-2"])
    assert B.risk_parity_arms(singular, regime_paths) is None
    inst = B._build_instrument(singular, T.PRIMARY, draws, paths=regime_paths, workers=1,
                               boot_draws=BOOT)
    evaluation = B._evaluate(singular, inst, workers=1, boot_draws=20)
    for lock in ("B-1", "B-2"):
        assert evaluation[lock]["verdict_before_pit"] == T.UNDECIDABLE
        assert not evaluation[lock]["read"]
    assert np.isnan(evaluation["B-1"]["G"]) and np.isnan(evaluation["B-2"]["delta"][5.0])
    reading = {**evaluation, "test_sessions": len(data.test_sessions), "pit": None}
    verdicts = B.lock_verdicts(reading)
    assert verdicts == {"B-1": T.UNDECIDABLE, "B-2": T.UNDECIDABLE, "B": T.UNDECIDABLE}
    assert B.trial_rows(reading, T.PRIMARY, verdicts) == {}
    assert B.trial_rows(reading, T.K3, verdicts) == {}


def test_a_missing_return_on_a_test_session_makes_b2_undecidable(data, regime_paths):
    excess = data.excess.copy()
    excess.iloc[-10, 3] = np.nan
    hole = dataclasses.replace(data, excess=excess)
    draws = T.placebo_draws(regime_paths, 4)
    printout = B.instrument(hole, T.PRIMARY, draws, paths=regime_paths, workers=1,
                            boot_draws=BOOT)
    assert printout["B-2"]["missing_test_sessions"]["state"] > 0
    assert printout["B-2"]["T_B2"] == B.NON_FINITE
    assert printout["undecidable"]["B-1"] == []
    inst = B._build_instrument(hole, T.PRIMARY, draws, paths=regime_paths, workers=1,
                               boot_draws=BOOT)
    evaluation = B._evaluate(hole, inst, workers=1, boot_draws=20)
    assert evaluation["B-2"]["verdict_before_pit"] == T.UNDECIDABLE
    assert not evaluation["B-2"]["read"] and evaluation["B-1"]["read"]
    assert np.isnan(evaluation["B-2"]["placebo_delta"]).all()
    assert np.isfinite(evaluation["B-1"]["G"])
    reading = {**evaluation, "test_sessions": len(data.test_sessions), "pit": None}
    verdicts = B.lock_verdicts(reading)
    assert set(B.trial_rows(reading, T.PRIMARY, verdicts)) == {"B-1"}
    assert set(B.trial_rows(reading, T.K3, verdicts)) == {"B"}


# -------------------------------------------------------------------------- verdicts

B1_PASSING = dict(statistic=0.4, p=0.001, exact=True, p_w1=0.01, p_w2=0.02, pooled_pd=True,
                  cells_ok=True, fallback_share=0.07)


@pytest.mark.parametrize(("change", "verdict"), [
    ({}, "PASS"),
    ({"statistic": np.nan}, "UNDECIDABLE"),
    ({"p": np.nan}, "UNDECIDABLE"),
    ({"exact": False}, "UNDECIDABLE"),
    ({"pooled_pd": False}, "UNDECIDABLE"),
    ({"cells_ok": False}, "UNDECIDABLE"),
    ({"fallback_share": 0.51}, "UNDECIDABLE"),
    ({"p_w2": np.nan}, "UNDECIDABLE"),
    ({"p": 0.011}, "NOT SHOWN"),
    ({"p": 0.009}, "NOT SHOWN"),
    ({"p": 0.009, "holm_rejected": True}, "PASS"),
    ({"p": 0.001, "holm_rejected": False}, "NOT SHOWN"),
    ({"p": 0.5, "p_w1": 0.9}, "NOT SHOWN"),
    ({"p_w1": 0.051}, "DOMINATED"),
    ({"p_w2": 0.2}, "DOMINATED"),
    ({"p_w1": 0.05, "p_w2": 0.05}, "PASS"),
])
def test_the_b1_lines_apply_in_order(change, verdict):
    assert B.b1_verdict(**{**B1_PASSING, **change}) == verdict


B2_PASSING = dict(delta5=0.5, delta10=0.4, threshold=0.338, p=0.001, kill_fires=False,
                  beta_pct=0.5, diff_pct=0.99, delta_w=0.1, fallback_share=0.07,
                  leg_missing=False, exact=True, pooled_pd=True, cells_ok=True)


@pytest.mark.parametrize(("change", "verdict"), [
    ({}, "PASS"),
    ({"diff_pct": np.nan}, "UNDECIDABLE"),
    ({"kill_fires": None}, "UNDECIDABLE"),
    ({"leg_missing": True}, "UNDECIDABLE"),
    ({"fallback_share": 0.5001}, "UNDECIDABLE"),
    ({"exact": False}, "UNDECIDABLE"),
    ({"pooled_pd": False, "kill_fires": True}, "UNDECIDABLE"),
    ({"cells_ok": False}, "UNDECIDABLE"),
    ({"kill_fires": True, "delta5": 2.0}, "FAIL (cost)"),
    ({"delta5": 0.0, "delta10": -0.1}, "FAIL"),
    ({"delta10": 0.0}, "UNDECIDED"),
    ({"delta5": 0.337}, "UNDERPOWERED"),
    ({"p": 0.0084}, "UNDERPOWERED"),
    ({"p": 0.0084, "holm_rejected": True}, "PASS"),
    ({"beta_pct": 0.95, "diff_pct": 0.1}, "DOWNGRADED (beta)"),
    ({"diff_pct": 0.94, "delta_w": 0.9}, "DOWNGRADED (not conditional)"),
    ({"delta_w": 0.5}, "DOMINATED (volatility)"),
])
def test_the_b2_lines_apply_in_order(change, verdict):
    assert B.b2_verdict(**{**B2_PASSING, **change}) == verdict


def test_pit_is_line_9_and_only_downgrades():
    assert T.apply_pit(B.b2_verdict(**B2_PASSING), "PASS") == "PASS"
    assert T.apply_pit(B.b2_verdict(**B2_PASSING), "UNDERPOWERED") == "DOWNGRADED (PIT)"
    assert T.apply_pit(B.b1_verdict(**B1_PASSING), "DOMINATED") == "DOWNGRADED (PIT)"
    assert T.apply_pit(B.b1_verdict(**{**B1_PASSING, "p": 0.5}), None) == "NOT SHOWN"


def test_the_variants_level_b_reads(data):
    assert B.pit_variant(T.PRIMARY) is T.PIT18
    k3 = B.pit_variant(T.K3)
    assert (k3.K, k3.smoothing, k3.features, k3.role) == (3, None, T.PIT_FEATURES, "control")
    smooth = B.pit_variant(T.SMOOTH21)
    assert smooth.smoothing == 21 and len(smooth.features) == 18
    assert B.section_name(T.PRIMARY) == "B" and B.section_name(T.K5) == "B:K=5"
    assert B.threshold_section is B.section_name
    with pytest.raises(ValueError, match="not committed"):
        B.section_name(T.PIT18)
    with pytest.raises(ValueError, match="level B"):
        B.instrument(data, T.D025)
    with pytest.raises(ValueError, match="PIT rebuild"):
        B.instrument(data, T.PIT18)


def test_the_placebo_is_seed_0_uniform_on_these_paths(data, regime_paths, kmeans):
    partition = T.Partition(regime_paths, None)
    by_count = B.instrument(data, T.PRIMARY, 6, partition=partition, workers=1, boot_draws=BOOT)
    given = B.instrument(data, T.PRIMARY, T.placebo_draws(regime_paths, 6, 0), paths=regime_paths,
                         workers=1, boot_draws=BOOT)
    assert by_count == given and by_count["placebo"] == {"method": "uniform", "draws": 6,
                                                         "seed": 0, "exact": True}
    with pytest.raises(ValueError, match="seed 0"):
        B.instrument(data, T.PRIMARY, T.placebo_draws(regime_paths, 6, 1), paths=regime_paths)
    with pytest.raises(ValueError, match="not drawn on these paths"):
        B.instrument(data, T.PRIMARY, T.placebo_draws(kmeans, 6, 0), paths=regime_paths)
    with pytest.raises(ValueError, match="not both"):
        B.instrument(data, T.PRIMARY, 6, partition=partition, paths=regime_paths)


def test_a_smoothed_partition_leaves_its_unlabelled_sessions_out(data):
    smoothed = T.partition_paths(data, T.SMOOTH21).paths
    for fold in data.folds:
        m = B.second_moments(data.legs, smoothed, fold.number)
        assert m.n_pooled == len(fold.train(data.sessions)) - 21
    printout = B.instrument(data, T.SMOOTH21, T.placebo_draws(smoothed, 4), paths=smoothed,
                            workers=1, boot_draws=BOOT)
    assert (printout["variant"], printout["smoothing"], printout["K"]) == ("smooth21", 21, 4)
    assert printout["placebo"]["exact"] and printout["B-1"]["non_finite_draws"] == 0
    assert B.section_name(T.SMOOTH21) == "B:smooth21"


def test_the_pit_rebuild_refits_the_18_features_with_its_own_draws(data, regime,
                                                                    regime_paths):
    planted = _plant(data, regime)
    rebuild = B.pit_rebuild(planted, T.PRIMARY, n_draws=10, workers=1, boot_draws=20)
    assert rebuild.variant is T.PIT18 and rebuild.instrument["features"] == 18
    assert rebuild.instrument["placebo"]["draws"] == 10
    assert set(rebuild.verdicts) == {"B-1", "B-2"}
    assert rebuild.verdicts["B-1"] in (T.NOT_SHOWN, T.UNDECIDABLE)  # 1/11 > 0.01: cannot hold
    assert T.apply_pit(T.PASS, rebuild.verdicts["B-1"]) == T.DOWNGRADED_PIT


# ----------------------------------------------------------------- the reading, guarded

INPUTS = ("data/raw/fake.parquet",)


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t",
                    *args], check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / INPUTS[0]).parent.mkdir(parents=True)
    (tmp_path / INPUTS[0]).write_bytes(b"synthetic")
    shutil.copy(ROOT / "uv.lock", tmp_path / "uv.lock")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", "uv.lock")
    _git(tmp_path, "commit", "-q", "-m", "init")
    return tmp_path


def test_the_reading_runs_only_on_committed_thresholds_and_logs_the_13_5_rows(data, kmeans,
                                                                           repo, tmp_path):
    latent = kmeans.xs(5, level="fold").droplevel("segment")
    planted = _plant(data, latent)
    draws = T.placebo_draws(kmeans, DRAWS, 0)
    printout = B.instrument(planted, T.PRIMARY, draws, paths=kmeans, workers=1,
                            boot_draws=BOOT)
    path = repo / T.THRESHOLDS_FILE
    guard = dict(root=repo, path=path, inputs=INPUTS)
    T.write_thresholds(printout, section="B", **guard)
    guard = {**guard, "boot_draws": BOOT}
    stored = T.read_thresholds(path)["thresholds"]["B"]
    with pytest.raises(T.ReadingRefused, match="not tracked"):
        B.read(planted, T.PRIMARY, draws, stored, paths=kmeans, workers=1, **guard)
    _git(repo, "add", T.THRESHOLDS_FILE)
    _git(repo, "commit", "-q", "-m", "thresholds")
    with pytest.raises(T.ReadingRefused, match="bitwise"):
        B.read(planted, T.PRIMARY, DRAWS + 1, stored, paths=kmeans, workers=1, **guard)
    tampered = json.loads(json.dumps(stored))
    tampered["B-2"]["T_B2"] = 0.2
    with pytest.raises(T.ReadingRefused, match="not the committed"):
        B.read(planted, T.PRIMARY, draws, tampered, paths=kmeans, workers=1, **guard)

    reading = B.read(planted, T.PRIMARY, draws, stored, paths=kmeans, workers=1,
                     pit_draws=DRAWS, **guard)
    assert reading["instrument"] == printout
    b1 = reading["B-1"]
    assert b1["p"] == 1 / (DRAWS + 1) and b1["verdict_before_pit"] == T.PASS
    assert reading["pit"] is not None and reading["pit"]["variant"] == "pit18"
    assert reading["pit"]["verdicts"]["B-1"] == T.PASS
    assert reading["verdicts"]["B-1"] == T.PASS
    assert reading["verdicts"]["B"] == T.level_verdict([reading["verdicts"]["B-1"],
                                                        reading["verdicts"]["B-2"]])
    b2 = reading["B-2"]
    assert b2["reading"]["delta"] == b2["delta"][5.0]
    assert len(b2["placebo_delta"]) == DRAWS and np.isfinite(b2["placebo_delta"]).all()
    assert b2["diff_pct"] == protocol.placebo_percentile(b2["delta"][5.0], b2["placebo_delta"])
    assert np.isfinite(b2["delta_w"])
    assert reading["holm_p"] == {"B-1": b1["p"], "B-2": b2["reading"]["p"]}
    assert B.lock_verdicts(reading, rejected=set())["B-1"] == T.NOT_SHOWN
    assert B.lock_verdicts(reading, rejected={"B-1"})["B-1"] == T.PASS

    trials = tmp_path / "trials.parquet"
    rows = reading["trial_rows"]
    assert set(rows) == {"B-1", "B-2"}
    for test, row in rows.items():
        kwargs = {k: v for k, v in row.items() if k != "test"}
        T.log_trial(test, T.PRIMARY, path=trials, **kwargs)
    assert rows["B-1"]["delta"] == b1["G"] and rows["B-1"]["threshold"] == b1["threshold"]
    assert rows["B-2"]["threshold"] == printout["B-2"]["T_B2"]
    assert rows["B-2"]["sessions"] == len(data.test_sessions)


def test_a_sensitivity_logs_one_level_row():
    reading = {
        "test_sessions": 100,
        "B-1": {"G": 0.1, "threshold": 0.2, "p": 0.3, "placebo_pct": 0.4},
        "B-2": {"sharpe_state": 0.5, "delta": {5.0: -0.1}, "threshold": 0.338,
                "reading": {"p": 1.0}, "diff_pct": 0.2},
    }
    verdicts = {"B-1": "NOT SHOWN", "B-2": "FAIL", "B": "FAIL"}
    rows = B.trial_rows(reading, T.K3, verdicts)
    assert rows == {"B": {"sharpe": 0.5, "delta": -0.1, "threshold": 0.338, "p": 1.0,
                          "placebo_pct": 0.2, "p2": 0.3, "sessions": 100, "verdict": "FAIL",
                          "note": ""}}
    flagged = B.trial_rows(reading, T.K3, verdicts, primary_level=T.PASS)["B"]
    assert flagged["note"] == "PASS (not robust): delta <= 0 at this sensitivity (§12.13)"
    assert B.trial_rows(reading, T.PIT18, {}) == {}
    with pytest.raises(ValueError, match="six primaries"):
        B.lock_verdicts({**reading, "role": "sensitivity"}, {"B-1"})

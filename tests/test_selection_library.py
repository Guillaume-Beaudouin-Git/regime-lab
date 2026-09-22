"""The Two Sigma signal library: causal, correctly normalised, and measured in weight space.

The causality tests are the ones that matter. Twelve signals means twelve places for a
look-ahead to hide — a rolling window aligned one row late, a seasonal average that
includes the current year — and a test that perturbs the future and asserts the past
does not move catches every one of them whatever the implementation looks like.

Everything runs on synthetic data. The single test that reads the real panel is the
admission-critical reproduction of §1 P2, and it skips when the panel is absent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regime_lab.config import RAW
from regime_lab.selection.library import (
    EXCLUDED,
    INDUSTRY_CANDIDATES,
    LIBRARY,
    TREND_CANDIDATES,
    annual_turnover,
    complete_sessions,
    components_for_share,
    cross_sectional_z,
    effective_rank,
    eigenvalues,
    equal_weight_blend,
    geometry,
    industry_library,
    industry_scores,
    load_panel,
    position_correlation,
    positions,
    rolling_beta,
    same_month_seasonality,
    time_series_z,
    trend_universe_scores,
    window_return,
)


def _industries(sessions: int = 1_600, legs: int = 12, seed: int = 0):
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2000-01-03", periods=sessions)
    market = pd.Series(rng.normal(0.0003, 0.010, sessions), index=index, name="ff_mkt-rf")
    loadings = rng.uniform(0.5, 1.5, legs)
    noise = rng.normal(0.0, 0.012, (sessions, legs))
    returns = pd.DataFrame(
        market.to_numpy()[:, None] * loadings + noise,
        index=index,
        columns=[f"ind_{i:02d}" for i in range(legs)],
    )
    return returns, market


def _prices(sessions: int = 700, legs: int = 8, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2005-01-03", periods=sessions)
    steps = rng.normal(0.0002, 0.012, (sessions, legs))
    columns = [f"X{i}" for i in range(legs)]
    return pd.DataFrame(100.0 * np.exp(np.cumsum(steps, axis=0)), index=index, columns=columns)


# ------------------------------------------------------------------- the declared sets


def test_the_library_is_the_twelve_candidates_minus_the_two_exclusions():
    assert len(INDUSTRY_CANDIDATES) == 12
    assert set(EXCLUDED) == {"IND_ACCEL", "IND_REV_1W"}
    assert len(LIBRARY) == 10
    assert set(LIBRARY) | set(EXCLUDED) == set(INDUSTRY_CANDIDATES)
    assert [n for n in INDUSTRY_CANDIDATES if n in LIBRARY] == list(LIBRARY)
    assert len(TREND_CANDIDATES) == 11


# ------------------------------------------------------------------------- transforms


def test_cross_sectional_z_is_centred_scaled_clipped_and_refuses_a_flat_row():
    frame = pd.DataFrame(
        [[1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 0.0, 100.0], [5.0, 5.0, 5.0, 5.0]],
        columns=list("abcd"),
    )
    z = cross_sectional_z(frame)
    np.testing.assert_allclose(z.iloc[0].mean(), 0.0, atol=1e-12)
    np.testing.assert_allclose(z.iloc[0].std(ddof=0), 1.0)
    wide = pd.DataFrame([np.r_[np.zeros(19), 100.0]])
    assert cross_sectional_z(wide).iloc[0, -1] == 3.0
    assert cross_sectional_z(wide).iloc[0].max() <= 3.0
    assert z.iloc[2].isna().all()


def test_time_series_z_divides_by_the_trailing_sd_without_demeaning():
    rng = np.random.default_rng(2)
    frame = pd.DataFrame(rng.normal(0.5, 1.0, (400, 2)), columns=["a", "b"])
    z = time_series_z(frame, window=100, clip=np.inf)
    t = 250
    expected = frame["a"].iloc[t] / frame["a"].iloc[t - 99 : t + 1].std()
    np.testing.assert_allclose(z["a"].iloc[t], expected)
    assert z.iloc[:99].isna().all().all()
    assert time_series_z(frame * 50.0, window=100).abs().max().max() <= 3.0 + 1e-12


def test_window_return_sums_exactly_the_skipped_window():
    lr = pd.DataFrame({"a": np.arange(30, dtype=float)})
    out = window_return(lr, 10, 3)
    t = 20
    assert out["a"].iloc[t] == lr["a"].iloc[t - 9 : t - 2].sum()
    assert out["a"].iloc[: 10 - 3 - 1 + 3].isna().all()


def test_rolling_beta_recovers_a_known_loading():
    rng = np.random.default_rng(3)
    index = pd.bdate_range("2010-01-04", periods=600)
    market = pd.Series(rng.normal(0.0, 0.01, 600), index=index)
    legs = pd.DataFrame(
        {"double": 2.0 * market + rng.normal(0, 1e-5, 600), "short": -market}, index=index
    )
    beta = rolling_beta(legs, market, window=252).iloc[-1]
    np.testing.assert_allclose(beta["double"], 2.0, atol=1e-2)
    np.testing.assert_allclose(beta["short"], -1.0, atol=1e-12)


def test_seasonality_averages_the_same_month_of_past_years_only():
    rng = np.random.default_rng(4)
    index = pd.bdate_range("1990-01-01", "2006-12-31")
    lr = pd.DataFrame({"a": rng.normal(0.0, 0.01, len(index))}, index=index)
    seasonal = same_month_seasonality(lr, years=10, min_years=5)

    marches = {y: lr.loc[f"{y}-03", "a"].sum() for y in range(1993, 2003)}
    day = lr.loc["2003-03"].index[5]
    np.testing.assert_allclose(seasonal.loc[day, "a"], np.mean(list(marches.values())))

    # January 1994 has four prior Januaries, one short of the minimum; 1995 has five.
    assert seasonal.loc["1994-01", "a"].isna().all()
    assert seasonal.loc["1995-01", "a"].notna().all()
    # One value per month, stamped on every session of it.
    assert seasonal.loc["2003-03", "a"].nunique() == 1

    bumped = lr.copy()
    bumped.loc["2003-03"] += 0.5
    after = same_month_seasonality(bumped, years=10, min_years=5)
    pd.testing.assert_series_equal(after.loc[:"2004-02", "a"], seasonal.loc[:"2004-02", "a"])
    assert not np.isclose(after.loc["2004-03", "a"].iloc[0], seasonal.loc["2004-03", "a"].iloc[0])


def test_positions_are_lagged_one_session_and_unit_gross():
    scores = pd.DataFrame(
        [[1.0, -1.0, 2.0], [0.0, 0.0, 0.0], [3.0, 1.0, -4.0], [np.nan, np.nan, np.nan]],
        columns=list("abc"),
    )
    held = positions(scores)
    assert held.iloc[0].isna().all()
    np.testing.assert_allclose(held.iloc[1].to_numpy(), np.array([1.0, -1.0, 2.0]) / 4.0)
    assert held.iloc[2].isna().all(), "an all-zero session must not read as a held zero"
    np.testing.assert_allclose(held.iloc[3].abs().sum(), 1.0)


# ---------------------------------------------------------------------------- signals


def test_industry_positions_at_t_cannot_see_returns_at_t_or_after():
    # Long enough that IND_LTREV (skips 252 sessions) and IND_SEASON (sees a shock only
    # a year later) both register the perturbation, so the test perturbs every signal.
    returns, market = _industries(sessions=2_000)
    split = returns.index[1_450]
    baseline = industry_library(returns, market, INDUSTRY_CANDIDATES)

    shocked_returns, shocked_market = returns.copy(), market.copy()
    rng = np.random.default_rng(9)
    shocked_returns.loc[split:] += rng.normal(0.0, 0.05, shocked_returns.loc[split:].shape)
    shocked_market.loc[split:] *= -3.0
    shocked = industry_library(shocked_returns, shocked_market, INDUSTRY_CANDIDATES)

    for name in INDUSTRY_CANDIDATES:
        pd.testing.assert_frame_equal(
            baseline[name].loc[:split], shocked[name].loc[:split], obj=f"{name}: the past moved"
        )
        after = returns.index[1_452]
        assert not baseline[name].loc[after:].equals(shocked[name].loc[after:]), name


def test_industry_scores_are_the_twelve_bounded_z_scores():
    returns, market = _industries()
    scores = industry_scores(returns, market)
    assert list(scores) == list(INDUSTRY_CANDIDATES)
    for name, frame in scores.items():
        assert frame.abs().max().max() <= 3.0 + 1e-12, name
        assert frame.iloc[-1].notna().all(), name


def test_the_library_is_complete_after_its_binding_warm_up():
    returns, market = _industries()
    weights = industry_library(returns, market)
    assert list(weights) == list(LIBRARY)
    sessions = complete_sessions(weights)
    assert len(sessions) > 0
    # IND_LTREV needs 1,260 returns, plus the one-session lag.
    assert sessions.min() >= returns.index[1_260]
    for name, frame in weights.items():
        np.testing.assert_allclose(frame.loc[sessions].abs().sum(axis=1), 1.0, err_msg=name)


def test_industry_library_refuses_a_name_outside_the_candidates():
    returns, market = _industries(sessions=300)
    with pytest.raises(ValueError, match="not an industry candidate"):
        industry_library(returns, market, ["IND_MOM_12_1", "IND_CARRY"])


def test_trend_universe_scores_are_eleven_causal_and_bounded():
    prices = _prices()
    scores = trend_universe_scores(prices)
    assert list(scores) == list(TREND_CANDIDATES)
    for name in ("BRKOUT_100", "BRKOUT_20"):
        values = scores[name].to_numpy()
        assert np.nanmin(values) >= -1.0 and np.nanmax(values) <= 1.0
    for name in TREND_CANDIDATES:
        assert np.nanmax(np.abs(scores[name].to_numpy())) <= 3.0 + 1e-12, name

    split = prices.index[600]
    shocked = prices.copy()
    shocked.loc[split:] *= 1.4
    before = {k: positions(v) for k, v in scores.items()}
    after = {k: positions(v) for k, v in trend_universe_scores(shocked).items()}
    for name in TREND_CANDIDATES:
        pd.testing.assert_frame_equal(before[name].loc[:split], after[name].loc[:split], obj=name)


# ------------------------------------------------------------------------ diagnostics


def _panel(seed: int, sessions: int = 300, legs: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.normal(size=(sessions, legs)), index=pd.bdate_range("2012-01-02", periods=sessions)
    )


def test_position_correlation_reads_identical_opposite_and_unrelated_signals():
    a, b = _panel(10), _panel(11)
    corr, sessions = position_correlation({"a": a, "neg": -a, "b": b})
    assert len(sessions) == len(a)
    np.testing.assert_allclose(np.diag(corr), 1.0)
    np.testing.assert_allclose(corr.loc["a", "neg"], -1.0)
    assert abs(corr.loc["a", "b"]) < 0.06
    np.testing.assert_allclose(corr.to_numpy(), corr.to_numpy().T)


def test_position_correlation_ignores_per_session_scale():
    a, b = _panel(12), _panel(13)
    blended = 0.6 * a + 0.8 * b
    scale = pd.Series(np.random.default_rng(14).uniform(0.1, 9.0, len(a)), index=a.index)
    plain, _ = position_correlation({"a": a, "mix": blended})
    scaled, _ = position_correlation({"a": a, "mix": blended.mul(scale, axis=0)})
    np.testing.assert_allclose(plain.to_numpy(), scaled.to_numpy(), atol=1e-12)


def test_min_legs_keeps_only_legs_every_signal_defines():
    a, b = _panel(15, sessions=4, legs=6), _panel(16, sessions=4, legs=6)
    a.iloc[0, :3] = np.nan          # session 0: only 3 common legs, dropped at min_legs=4
    b.iloc[1, 0] = np.nan           # session 1: 5 common legs, kept
    corr, sessions = position_correlation({"a": a, "b": b}, min_legs=4)
    assert list(sessions) == list(a.index[1:])

    legs = a.iloc[1].notna() & b.iloc[1].notna()
    daily = [np.corrcoef(a.iloc[1][legs], b.iloc[1][legs])[0, 1]]
    daily += [np.corrcoef(a.iloc[t], b.iloc[t])[0, 1] for t in (2, 3)]
    np.testing.assert_allclose(corr.loc["a", "b"], np.mean(daily))
    assert len(complete_sessions({"a": a, "b": b})) == 2


def test_effective_rank_and_components_on_known_structures():
    assert effective_rank(np.eye(5)) == pytest.approx(5.0)
    assert components_for_share(np.eye(5), 0.90) == 5
    assert effective_rank(np.ones((4, 4))) == pytest.approx(1.0)
    assert components_for_share(np.ones((4, 4)), 0.90) == 1
    two_blocks = np.kron(np.eye(2), np.ones((3, 3)))
    assert effective_rank(two_blocks) == pytest.approx(2.0)


def test_eigenvalues_are_descending_and_never_negative():
    values = eigenvalues(np.diag([1.0, 3.0, 2.0]))
    np.testing.assert_allclose(values, [3.0, 2.0, 1.0])
    # A correlation matrix averaged over sessions need not be positive semi-definite;
    # the rounding residue below zero is floored rather than counted as a dimension.
    indefinite = np.array([[1.0, 1.0 + 1e-12], [1.0 + 1e-12, 1.0]])
    assert (eigenvalues(indefinite) >= 0.0).all()
    assert eigenvalues(pd.DataFrame(np.eye(3))).tolist() == [1.0, 1.0, 1.0]


def test_geometry_of_a_subset_is_the_submatrix_on_the_same_sessions():
    a, b, c = _panel(17), _panel(18), _panel(19)
    weights = {"a": a, "b": 0.5 * a + b, "c": c}
    full = geometry(weights)
    part = geometry(weights, ["a", "b"])
    assert part.sessions.equals(full.sessions)
    np.testing.assert_allclose(
        part.correlation.to_numpy(), full.correlation.loc[["a", "b"], ["a", "b"]].to_numpy()
    )
    assert part.max_abs_corr == pytest.approx(abs(full.pair("a", "b")))
    assert full.strongest_pairs(1)[0][:2] == ("a", "b")


def test_equal_weight_blend_is_the_plain_average():
    a, b = _panel(20), _panel(21)
    blend = equal_weight_blend({"a": a, "b": b, "c": 3.0 * a})
    pd.testing.assert_frame_equal(blend, (a + b + 3.0 * a) / 3.0)
    pd.testing.assert_frame_equal(equal_weight_blend({"a": a, "b": b}, ["b"]), b)
    with pytest.raises(ValueError):
        equal_weight_blend({"a": a}, [])


def test_annual_turnover_counts_every_change_and_refuses_a_gap():
    index = pd.bdate_range("2015-01-01", periods=5)
    still = pd.DataFrame({"a": [0.5] * 5, "b": [-0.5] * 5}, index=index)
    assert annual_turnover(still) == 0.0
    flip = pd.DataFrame({"a": [0.5, -0.5] * 2 + [0.5], "b": [-0.5, 0.5] * 2 + [-0.5]}, index=index)
    assert annual_turnover(flip) == pytest.approx(2.0 * 252)
    with pytest.raises(ValueError, match="non-finite"):
        gap = flip.copy()
        gap.iloc[2, 0] = np.nan
        annual_turnover(gap)
    with pytest.raises(ValueError, match="two sessions"):
        annual_turnover(still.iloc[:1])


def test_load_panel_pivots_rescales_and_masks_sentinels(tmp_path):
    periods = pd.bdate_range("2020-01-01", periods=3)
    long = pd.DataFrame({
        "series_id": ["ind_b"] * 3 + ["ind_a"] * 3,
        "period": list(periods) * 2,
        "available_at": list(periods) * 2,
        "value": [1.0, -2.0, 0.5, -99.99, 3.0, 4.0],
    })
    (tmp_path / "panels").mkdir()
    long.to_parquet(tmp_path / "panels" / "toy.parquet")
    wide = load_panel("toy", root=tmp_path)
    assert list(wide.columns) == ["ind_a", "ind_b"]
    assert wide.index.is_monotonic_increasing
    np.testing.assert_allclose(wide["ind_b"].to_numpy(), [0.01, -0.02, 0.005])
    assert np.isnan(wide["ind_a"].iloc[0])
    np.testing.assert_allclose(wide["ind_a"].iloc[1:].to_numpy(), [0.03, 0.04])


# -------------------------------------------------------- the disclosed figure, if on disk

PANEL = RAW / "panels" / "industry_49.parquet"
FACTORS = RAW / "panels" / "factors_5.parquet"


@pytest.mark.skipif(not (PANEL.exists() and FACTORS.exists()), reason="panel not on disk")
def test_the_disclosed_p2_effective_rank_reproduces_from_committed_code():
    """§1 P2 is admission-critical. If a re-fetch of the panel moves it, this says so."""
    weights = industry_library(
        load_panel("industry_49"), load_panel("factors_5")["ff_mkt-rf"], INDUSTRY_CANDIDATES
    )
    twelve = geometry(weights)
    assert round(twelve.effective_rank, 2) == 8.37
    assert len(twelve.sessions) == 7_946
    assert round(twelve.pair("IND_ACCEL", "IND_MOM_12_1"), 3) == -0.912
    assert geometry(weights, LIBRARY).effective_rank >= 4.0

"""The Two Sigma portfolio protocol must be causal, blind and priced as §5 says.

Three properties matter more than the arithmetic. A state known at the close of t
must not steer the book on t. A power calculation must not let a mean through,
which is tested by adding a mean and checking nothing moves. And a missing input
must surface as a NaN, never as a zero that quietly shortens the sample.
All data here is synthetic; nothing reads `data/`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regime_lab.extensions.vehicle import portfolio_scalar
from regime_lab.selection.protocol import (
    BLIND_TOLERANCE,
    COST_BPS,
    TILT_D,
    annual_turnover,
    blend,
    blinded_mde,
    breakeven_bps,
    cost_in_sharpe,
    equal_mix,
    excess_returns,
    factor_series,
    map_states,
    market_excess,
    match_gross,
    mde_at,
    mde_z,
    net_of_costs,
    paired_hac_t,
    placebo_percentile,
    realised_beta,
    risk_free,
    risk_parity_mix,
    scale_to_target,
    standalone_sharpe_threshold,
    switch_table,
    target_volatility,
    tilt_mix,
    tilt_table,
    turnover_kill,
)

SESSIONS = pd.bdate_range("2012-01-02", periods=400)
INDUSTRIES = [f"ind{i:02d}" for i in range(8)]
NAMES = ["MOM", "REV", "LOWVOL"]


def _signals(seed: int = 0) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    out = {}
    for name in NAMES:
        raw = rng.normal(size=(len(SESSIONS), len(INDUSTRIES)))
        raw -= raw.mean(axis=1, keepdims=True)
        raw /= np.abs(raw).sum(axis=1, keepdims=True)
        out[name] = pd.DataFrame(raw, index=SESSIONS, columns=INDUSTRIES)
    return out


def _returns(seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = rng.normal(0.0003, 0.012, size=(len(SESSIONS), len(INDUSTRIES)))
    return pd.DataFrame(data, index=SESSIONS, columns=INDUSTRIES)


# --- the blend ---------------------------------------------------------------
def test_the_control_is_the_plain_average_of_the_signals():
    signals = _signals()
    expected = sum(signals.values()) / len(signals)
    pd.testing.assert_frame_equal(blend(signals), expected)


def test_a_mix_weights_each_signal_on_its_own_session():
    signals = _signals()
    mix = equal_mix(NAMES, SESSIONS)
    mix.iloc[5] = [1.0, 0.0, 0.0]
    out = blend(signals, mix)
    pd.testing.assert_series_equal(out.iloc[5], signals["MOM"].iloc[5])


def test_a_missing_signal_value_is_a_nan_not_a_zero():
    signals = _signals()
    signals["REV"].iloc[10, 2] = np.nan
    out = blend(signals)
    assert np.isnan(out.iloc[10, 2])
    assert np.isfinite(out.drop(index=SESSIONS[10])).all().all()


def test_misaligned_signals_and_bad_mixes_are_refused():
    signals = _signals()
    shifted = dict(signals)
    shifted["REV"] = signals["REV"].iloc[1:]
    with pytest.raises(ValueError, match="aligned"):
        blend(shifted)
    bad = equal_mix(NAMES, SESSIONS) * 2.0
    with pytest.raises(ValueError, match="sum to one"):
        blend(signals, bad)
    with pytest.raises(ValueError, match="no weight"):
        blend(signals, equal_mix(NAMES[:2], SESSIONS))


# --- the tilt ----------------------------------------------------------------
def test_the_tilt_is_the_formula_of_section_four():
    m = pd.DataFrame([[1.0, 0.0, -1.0], [0.4, -0.2, 0.3]], index=[0, 1], columns=NAMES)
    table = tilt_table(m, d=0.5)
    raw = (1.0 + 0.5 * m.to_numpy()) / 3.0
    np.testing.assert_allclose(table.to_numpy(), raw / raw.sum(axis=1, keepdims=True))
    np.testing.assert_allclose(table.sum(axis=1), 1.0)
    assert TILT_D == 0.50


def test_the_floor_comes_before_the_renormalisation():
    m = pd.DataFrame([[4.0, -4.0, 0.0]], index=[0], columns=NAMES)
    row = tilt_table(m, d=0.5).iloc[0]
    assert row["REV"] == 0.0
    assert row.sum() == pytest.approx(1.0)
    assert (row >= 0.0).all()


def test_a_state_with_nothing_left_or_no_estimate_abstains_at_equal_weight():
    m = pd.DataFrame([[-5.0, -5.0, -5.0], [np.nan, 0.2, 0.1]], index=[0, 1], columns=NAMES)
    table = tilt_table(m, d=1.0)
    np.testing.assert_allclose(table.to_numpy(), 1.0 / 3.0)


def test_zero_tilt_strength_is_the_control():
    m = pd.DataFrame(np.random.default_rng(2).normal(size=(4, 3)), columns=NAMES)
    np.testing.assert_allclose(tilt_table(m, d=0.0).to_numpy(), 1.0 / 3.0)


# --- states to dates: the T-1 rule ---------------------------------------------
def test_the_state_known_at_the_close_of_t_steers_the_book_on_t_plus_one():
    states = pd.Series([0, 0, 1, 1, 0], index=SESSIONS[:5])
    table = pd.DataFrame([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], index=[0, 1], columns=NAMES)
    mix = map_states(states, table)
    # session 0 has no previous state and holds the fallback
    np.testing.assert_allclose(mix.iloc[0], 1.0 / 3.0)
    assert mix.iloc[2]["MOM"] == 1.0, "state 1 appeared at t=2 and must not act before t=3"
    assert mix.iloc[3]["REV"] == 1.0
    assert mix.iloc[4]["REV"] == 1.0
    lagless = map_states(states, table, lag=0)
    assert lagless.iloc[2]["REV"] == 1.0


def test_changing_a_future_state_does_not_move_the_past():
    rng = np.random.default_rng(3)
    states = pd.Series(rng.integers(0, 4, size=len(SESSIONS)), index=SESSIONS)
    m = pd.DataFrame(rng.normal(size=(4, 3)), columns=NAMES)
    base = tilt_mix(states, m)
    changed = states.copy()
    changed.iloc[200:] = (changed.iloc[200:] + 1) % 4
    moved = tilt_mix(changed, m)
    pd.testing.assert_frame_equal(base.iloc[:201], moved.iloc[:201])
    assert not base.iloc[201:].equals(moved.iloc[201:])


def test_an_unknown_state_holds_the_fallback():
    states = pd.Series([0.0, np.nan, 7.0, 1.0], index=SESSIONS[:4])
    table = pd.DataFrame([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], index=[0, 1], columns=NAMES)
    pooled = pd.Series([0.2, 0.3, 0.5], index=NAMES)
    mix = map_states(states, table, fallback=pooled, lag=0)
    np.testing.assert_allclose(mix.iloc[1], pooled)
    np.testing.assert_allclose(mix.iloc[2], pooled)
    assert mix.iloc[3]["LOWVOL"] == 1.0
    with pytest.raises(ValueError, match="sum to one"):
        map_states(states, table * 2.0)


def test_the_lag_refuses_a_calendar_it_cannot_count_on():
    table = pd.DataFrame([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], index=[0, 1], columns=NAMES)
    shuffled = pd.Series([0, 1, 0], index=SESSIONS[[2, 0, 1]])
    with pytest.raises(ValueError, match="sorted"):
        map_states(shuffled, table)
    doubled = pd.Series([0, 1, 0], index=SESSIONS[[0, 1, 1]])
    with pytest.raises(ValueError, match="sorted"):
        map_states(doubled, table)


def test_the_hard_switch_holds_the_top_signal_of_each_state():
    m = pd.DataFrame(
        [[0.1, 0.9, -0.3], [2.0, 2.0, 0.0], [np.nan, 1.0, 0.0]], index=[0, 1, 2], columns=NAMES
    )
    table = switch_table(m)
    np.testing.assert_allclose(table.loc[0], [0.0, 1.0, 0.0])
    np.testing.assert_allclose(table.loc[1], [1.0, 0.0, 0.0])  # a tie goes to the first
    np.testing.assert_allclose(table.loc[2], 1.0 / 3.0)
    states = pd.Series([0, 0, 1], index=SESSIONS[:3])
    mix = map_states(states, table)
    np.testing.assert_allclose(mix.iloc[2], [0.0, 1.0, 0.0])


# --- risk parity (level B) -----------------------------------------------------
def test_risk_parity_equalises_the_risk_contributions():
    rng = np.random.default_rng(4)
    a = rng.normal(size=(500, 5)) @ rng.normal(size=(5, 5))
    cov = np.cov(a, rowvar=False)
    w = risk_parity_mix(cov)
    contrib = w * (cov @ w)
    np.testing.assert_allclose(contrib / contrib.sum(), 0.2, atol=1e-8)
    assert w.sum() == pytest.approx(1.0)
    assert (w > 0).all(), "no signal may be shorted"


def test_risk_parity_on_a_diagonal_is_inverse_volatility():
    vols = np.array([0.05, 0.10, 0.20])
    frame = pd.DataFrame(np.diag(vols**2), index=NAMES, columns=NAMES)
    w = risk_parity_mix(frame)
    assert list(w.index) == NAMES
    np.testing.assert_allclose(w.to_numpy(), (1 / vols) / (1 / vols).sum(), atol=1e-10)


def test_with_two_signals_risk_parity_ignores_the_correlation():
    for rho in (-0.6, 0.0, 0.8):
        cov = np.array([[0.04, rho * 0.2 * 0.1], [rho * 0.2 * 0.1, 0.01]])
        np.testing.assert_allclose(risk_parity_mix(cov), [1 / 3, 2 / 3], atol=1e-9)


def test_risk_parity_refuses_a_broken_covariance():
    with pytest.raises(ValueError):
        risk_parity_mix(np.array([[1.0, 0.0], [0.0, 0.0]]))
    with pytest.raises(ValueError):
        risk_parity_mix(np.array([[1.0, 2.0], [2.0, 1.0]]))
    with pytest.raises(ValueError):
        risk_parity_mix(np.array([[1.0, np.nan], [np.nan, 1.0]]))


def test_match_gross_copies_the_reference_gross_row_by_row():
    signals = _signals()
    reference = blend(signals)
    other = signals["MOM"] * 3.0
    matched = match_gross(other, reference)
    np.testing.assert_allclose(matched.abs().sum(axis=1), reference.abs().sum(axis=1))
    zero = other * 0.0
    assert (match_gross(zero, reference) == 0.0).all().all()


# --- volatility target -----------------------------------------------------------
def test_the_target_reuses_the_vehicle_multiplier_and_leaves_the_warm_up_empty():
    rng = np.random.default_rng(5)
    r = pd.Series(rng.normal(0.0, 0.004, size=len(SESSIONS)), index=SESSIONS)
    scaled, multiplier, _ = scale_to_target(r)
    reference, _ = portfolio_scalar(r)
    assert scaled.iloc[:63].isna().all()
    assert scaled.iloc[63:].notna().all()
    pd.testing.assert_series_equal(
        multiplier.iloc[63:], reference.iloc[63:], check_names=False
    )


def test_the_cap_binds_on_a_quiet_book():
    rng = np.random.default_rng(6)
    r = pd.Series(rng.normal(0.0, 0.0005, size=len(SESSIONS)), index=SESSIONS)
    _, multiplier, binds = scale_to_target(r)
    assert multiplier.iloc[63:].max() == pytest.approx(3.0)
    assert binds.iloc[63:].astype(bool).all()


def test_the_book_at_t_cannot_see_returns_at_t_or_after():
    weights = blend(_signals()).shift(1)
    returns = _returns()
    split = 250
    base = target_volatility(weights, returns).returns
    shocked = returns.copy()
    shocked.iloc[split:] *= 4.0
    moved = target_volatility(weights, shocked).returns
    pd.testing.assert_series_equal(base.iloc[:split], moved.iloc[:split])
    assert not base.iloc[split:].equals(moved.iloc[split:]), "the test perturbed nothing"


def test_the_held_weights_carry_the_multiplier():
    weights = blend(_signals()).shift(1)
    book = target_volatility(weights, _returns())
    live = book.multiplier.dropna().index
    pd.testing.assert_frame_equal(
        book.weights.loc[live], weights.loc[live].mul(book.multiplier.loc[live], axis=0)
    )
    assert book.returns.iloc[0:64].isna().all()


# --- costs and turnover -----------------------------------------------------------
def test_the_schedule_is_the_one_priced_in_section_five():
    assert COST_BPS == {"realistic": 5.0, "conservative": 10.0, "stress": 20.0}


def test_costs_are_charged_on_the_absolute_change_in_weights():
    idx = SESSIONS[:4]
    w = pd.DataFrame({"a": [0.0, 0.5, 0.5, -0.5], "b": [0.0, -0.5, 0.0, 0.0]}, index=idx)
    r = pd.Series(0.0, index=idx)
    net = net_of_costs(r, w, bps=10.0)
    np.testing.assert_allclose(net.to_numpy(), [0.0, -1.0e-3, -0.5e-3, -1.0e-3])


def test_turnover_skips_the_undefined_first_change():
    idx = SESSIONS[:3]
    w = pd.DataFrame({"a": [0.5, -0.5, -0.5], "b": [0.5, 0.5, 0.5]}, index=idx)
    assert annual_turnover(w) == pytest.approx((1.0 + 0.0) / 2 * 252)


def test_turnover_never_counts_a_partial_row():
    # One leg missing on session 2: the changes into and out of it are skipped, not
    # summed over the leg that happens to be present.
    idx = SESSIONS[:5]
    w = pd.DataFrame(
        {"a": [0.5, -0.5, np.nan, 0.5, 0.5], "b": [0.5, 0.5, 0.0, 0.5, -0.5]}, index=idx
    )
    assert annual_turnover(w) == pytest.approx((1.0 + 1.0) / 2 * 252)
    assert np.isnan(annual_turnover(w.iloc[:1]))
    warm_up = pd.DataFrame({"a": [np.nan, np.nan, 0.5, 0.0]}, index=SESSIONS[:4])
    assert annual_turnover(warm_up) == pytest.approx(0.5 * 252)


def test_the_cost_kill_arithmetic_of_section_five():
    assert breakeven_bps(0.241, 2.6) == pytest.approx(92.69, abs=0.01)
    assert turnover_kill(0.241, COST_BPS["stress"]) == pytest.approx(12.05, abs=0.01)
    assert cost_in_sharpe(2.6, 5.0) == pytest.approx(0.013)
    assert cost_in_sharpe(21.6, 5.0) == pytest.approx(0.108)
    assert cost_in_sharpe(2.6, breakeven_bps(0.241, 2.6)) == pytest.approx(0.241)
    assert breakeven_bps(0.3, 0.0) == float("inf")


# --- excess returns and the market ----------------------------------------------------
def _panel(delay_days: int) -> pd.DataFrame:
    periods = pd.bdate_range("2020-01-01", periods=6)
    rows = []
    for i, p in enumerate(periods):
        rows.append(("ff_rf", p, p + pd.Timedelta(days=delay_days), 0.01 * (i + 1)))
        rows.append(("ff_mkt-rf", p, p + pd.Timedelta(days=delay_days), 1.0 * (i + 1)))
    return pd.DataFrame(rows, columns=["series_id", "period", "available_at", "value"])


def test_the_risk_free_rate_is_taken_when_published_never_at_its_period():
    panel = _panel(delay_days=2)
    index = pd.bdate_range("2020-01-01", periods=8)
    rf = risk_free(panel, index)
    # the first value is published two calendar days after 2020-01-01
    assert np.isnan(rf.iloc[0])
    assert rf.loc["2020-01-03"] == pytest.approx(0.0001)
    assert rf.index.equals(index)
    same_day = risk_free(_panel(delay_days=0), index)
    assert same_day.iloc[0] == pytest.approx(0.0001)
    assert same_day.iloc[-1] == pytest.approx(0.0006), "a rate is carried forward"


def test_the_market_return_is_paired_on_its_own_session_and_never_filled():
    panel = _panel(delay_days=5)
    index = pd.bdate_range("2020-01-01", periods=8)
    mkt = market_excess(panel, index)
    assert mkt.iloc[0] == pytest.approx(0.01)
    assert mkt.iloc[-1:].isna().all()
    with pytest.raises(KeyError):
        factor_series(panel, "ff_mom", index)


def test_excess_is_taken_instrument_by_instrument():
    returns = _returns().iloc[:5]
    rf = pd.Series(0.0001, index=returns.index)
    ex = excess_returns(returns, rf)
    np.testing.assert_allclose(ex.to_numpy(), returns.to_numpy() - 0.0001)
    w = blend(_signals()).iloc[:5]
    # a book pays the cash rate on its signed net exposure only
    np.testing.assert_allclose(
        (w * ex).sum(axis=1), (w * returns).sum(axis=1) - w.sum(axis=1) * 0.0001
    )


# --- comparison helpers -----------------------------------------------------------------
def test_the_realised_beta_is_recovered():
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2000-01-03", periods=3000)
    mkt = pd.Series(rng.normal(0, 0.01, size=3000), index=idx)
    book = 0.3 * mkt + pd.Series(rng.normal(0, 0.002, size=3000), index=mkt.index)
    assert realised_beta(book, mkt) == pytest.approx(0.3, abs=0.01)
    assert np.isnan(realised_beta(book, mkt * 0.0))


def test_the_placebo_percentile_counts_ties_as_half():
    null = np.array([1.0, 2.0, 3.0, 4.0, np.nan])
    assert placebo_percentile(2.5, null) == pytest.approx(0.5)
    assert placebo_percentile(3.0, null) == pytest.approx(0.625)
    assert np.isnan(placebo_percentile(np.nan, null))


def _newey_west_t(x: np.ndarray, lags: int) -> float:
    """Bartlett-kernel HAC t of a mean, no small-sample correction, written out by hand."""
    n = len(x)
    e = x - x.mean()
    s = e @ e / n
    for lag in range(1, lags + 1):
        s += 2.0 * (1.0 - lag / (lags + 1.0)) * (e[lag:] @ e[:-lag]) / n
    return float(x.mean() / np.sqrt(s / n))


def test_the_paired_t_is_newey_west_lag_six_on_the_difference():
    # The first draft asserted t > 2 for a population t of about 3.2; seed 8 draws 1.70
    # (the plain t is 1.64). The assertion tested the seed, not the code. The code is
    # checked here against the estimator written out by hand, on a serially correlated
    # difference where HAC and the plain t disagree.
    rng = np.random.default_rng(8)
    idx = pd.bdate_range("2000-01-03", periods=2000)
    shock = rng.normal(0.0, 0.01, size=2001)
    a = pd.Series(0.0005 + shock[1:] + 0.5 * shock[:-1], index=idx)
    b = pd.Series(rng.normal(0.0, 0.01, size=2000), index=idx)
    t = paired_hac_t(a, b)
    assert t == pytest.approx(_newey_west_t((a - b).to_numpy(), 6), rel=1e-10)
    assert t == pytest.approx(-paired_hac_t(b, a))
    strong = b + 0.002 + pd.Series(rng.normal(0.0, 0.001, size=2000), index=idx)
    assert paired_hac_t(strong, b) > 10.0, "a mean difference far above the noise"


def test_the_paired_t_is_undefined_on_identical_or_short_legs():
    rng = np.random.default_rng(8)
    idx = pd.bdate_range("2000-01-03", periods=2000)
    a = pd.Series(rng.normal(0.001, 0.01, size=2000), index=idx)
    b = pd.Series(rng.normal(0.0, 0.01, size=2000), index=idx)
    assert np.isnan(paired_hac_t(a, a))
    assert np.isnan(paired_hac_t(a.iloc[:10], b.iloc[:10]))


# --- the power convention ---------------------------------------------------------------
def test_the_critical_multiples_of_the_tree():
    # z(1 - a/2) + z(0.80). The first draft carried 3.4796 and 3.8779, transcription
    # errors: z(1 - 0.05/12) = 2.63826 and z(1 - 0.05/42) = 3.03808, plus 0.84162.
    assert mde_z(0.05) == pytest.approx(2.8016, abs=1e-4)
    assert mde_z(0.05 / 6) == pytest.approx(3.4799, abs=1e-4)
    assert mde_z(0.05 / 21) == pytest.approx(3.8797, abs=1e-4)


def test_the_blinded_mde_reads_no_mean():
    rng = np.random.default_rng(9)
    a = rng.normal(0.0, 0.01, size=1500)
    b = 0.6 * a + rng.normal(0.0, 0.008, size=1500)
    base = blinded_mde(a, b, mean_block=21, draws=200, seed=0)
    shifted = blinded_mde(a + 0.004, b - 0.002, mean_block=21, draws=200, seed=0)
    assert abs(base.observed) < BLIND_TOLERANCE
    assert shifted.se == pytest.approx(base.se, rel=1e-9), "a mean moved the standard error"
    assert shifted.mde == pytest.approx(base.mde, rel=1e-9)


def test_the_blinded_mde_drops_sessions_missing_on_either_leg():
    rng = np.random.default_rng(10)
    a = rng.normal(0.0, 0.01, size=600)
    b = rng.normal(0.0, 0.01, size=600)
    a_gap = a.copy()
    a_gap[:50] = np.nan
    gapped = blinded_mde(a_gap, b, mean_block=21, draws=100, seed=0)
    trimmed = blinded_mde(a[50:], b[50:], mean_block=21, draws=100, seed=0)
    assert gapped.se == pytest.approx(trimmed.se)
    with pytest.raises(ValueError):
        blinded_mde(a, b[:-1], mean_block=21, draws=10)


def test_the_blinded_mde_refuses_a_leg_without_variance():
    rng = np.random.default_rng(12)
    a = rng.normal(0.0, 0.01, size=300)
    with pytest.raises(ValueError, match="variance"):
        blinded_mde(a, np.full(300, 0.001), mean_block=21, draws=10)
    with pytest.raises(ValueError, match="variance"):
        blinded_mde(a, np.full(300, np.nan), mean_block=21, draws=10)


def test_a_corrected_alpha_rescales_the_same_standard_error():
    rng = np.random.default_rng(11)
    a, b = rng.normal(size=800), rng.normal(size=800)
    loose = blinded_mde(a, b, mean_block=21, draws=100, seed=0)
    strict = blinded_mde(a, b, mean_block=21, draws=100, seed=0, alpha=0.05 / 6)
    assert mde_at(loose, 0.05 / 6) == pytest.approx(strict.mde)
    assert strict.mde / loose.mde == pytest.approx(3.4799 / 2.8016, abs=1e-4)


def test_the_standalone_convention_reproduces_the_programme_reference():
    assert standalone_sharpe_threshold(23.2) == pytest.approx(0.638, abs=1e-3)
    assert standalone_sharpe_threshold(31.57) == pytest.approx(0.533, abs=1e-3)
    assert standalone_sharpe_threshold(3.9) == float("inf")

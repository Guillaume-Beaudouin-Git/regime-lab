"""The Bridgewater sleeve book: funding, within-sleeve risk, causality, cost charging.

Synthetic data throughout, except the last block, which checks the stored panel and
skips cleanly when ``data/`` is absent. Nothing here reads a quadrant label.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regime_lab.construction import sleeves as S
from regime_lab.extensions.vehicle import COST_SCHEDULE

SESSIONS = pd.bdate_range("2010-01-04", periods=800)
TOY = {"eq": ("^AAA", "^BBB"), "bond": ("SHY", "TLT"), "cmd": ("CL=F",)}


def _returns(seed: int = 0, vols: dict[str, float] | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    vols = vols or {"^AAA": 0.012, "^BBB": 0.018, "SHY": 0.001, "TLT": 0.009, "CL=F": 0.022}
    data = {name: rng.normal(0.0, v, len(SESSIONS)) for name, v in vols.items()}
    return pd.DataFrame(data, index=SESSIONS)


def _cash(value: float = 0.02) -> pd.Series:
    return pd.Series(value / 252, index=SESSIONS, name="cash")


def _dates(step: int = 63, first: int = 300) -> list[pd.Timestamp]:
    return [SESSIONS[i] for i in range(first, len(SESSIONS) - 1, step)]


# ---------------------------------------------------------------------------
# returns and funding
# ---------------------------------------------------------------------------
def test_a_non_positive_price_gives_no_return_that_session_or_the_next():
    prices = pd.DataFrame({"CL=F": [20.0, 18.0, -37.0, 10.0, 11.0]},
                          index=SESSIONS[:5])
    r = S.price_returns(prices)["CL=F"]
    assert r.iloc[1] == pytest.approx(-0.1)
    assert np.isnan(r.iloc[2]) and np.isnan(r.iloc[3])
    assert r.iloc[4] == pytest.approx(0.1)


def test_a_gap_is_not_booked_on_the_session_after_it():
    prices = pd.DataFrame({"PL=F": [100.0, np.nan, np.nan, 130.0, 131.3]},
                          index=SESSIONS[:5])
    r = S.price_returns(prices)["PL=F"]
    assert r.iloc[1:4].isna().all()
    assert r.iloc[4] == pytest.approx(0.01)


def test_usd_base_quotes_are_inverted_to_the_foreign_currency():
    prices = pd.DataFrame({"JPY=X": [100.0, 125.0], "EURUSD=X": [1.0, 1.25]},
                          index=SESSIONS[:2])
    r = S.price_returns(prices)
    assert r["JPY=X"].iloc[1] == pytest.approx(100.0 / 125.0 - 1.0)
    assert r["EURUSD=X"].iloc[1] == pytest.approx(0.25)


def test_the_cash_rate_is_read_at_available_at_never_at_period():
    frame = pd.DataFrame({
        "series_id": "rate_cash_3m",
        "period": SESSIONS[:3],
        "available_at": SESSIONS[1:4],
        "value": [1.0, 2.0, 3.0],
    })
    cash = S.cash_rate(SESSIONS[:4], frame) * 252 * 100
    assert np.isnan(cash.iloc[0])
    assert cash.iloc[1:].tolist() == pytest.approx([1.0, 2.0, 3.0])


def test_only_the_total_return_etfs_are_funded():
    r = _returns()
    x = S.excess_returns(r, _cash())
    for name in ("SHY", "TLT"):
        assert np.allclose(r[name] - x[name], 0.02 / 252)
    for name in ("^AAA", "^BBB", "CL=F"):
        assert np.allclose(r[name], x[name])


def test_a_book_on_excess_returns_pays_cash_on_its_signed_funded_exposure():
    r = _returns()
    cash = _cash()
    held = pd.DataFrame(
        np.tile([0.2, 0.1, 0.4, 0.2, 0.1], (len(SESSIONS), 1)), index=SESSIONS, columns=r.columns
    )
    gross = S.build_book(r, held)
    excess = S.build_book(S.excess_returns(r, cash), held)
    live = excess.live
    paid = S.funding_charge(excess.weights, cash).loc[live]
    assert np.allclose((gross.returns - excess.returns).loc[live], paid)
    assert np.allclose(paid, (0.4 + 0.2) * excess.multiplier.loc[live] * 0.02 / 252)


def test_the_cash_rate_must_cover_every_funded_session():
    r = _returns()
    cash = _cash().copy()
    cash.iloc[400] = np.nan
    with pytest.raises(ValueError, match="cash rate is missing"):
        S.excess_returns(r, cash)


# ---------------------------------------------------------------------------
# within-sleeve risk and the blind weights
# ---------------------------------------------------------------------------
def test_within_a_sleeve_every_instrument_carries_equal_standalone_risk():
    r = _returns()
    dates = _dates()
    within = S.within_sleeve_weights(r, TOY, dates, lookback=252)
    tau = dates[2]
    window = r.loc[:tau].iloc[-252:]
    sd = window.std(ddof=1)
    for members in TOY.values():
        w = within.loc[tau, list(members)]
        assert w.sum() == pytest.approx(1.0)
        risk = (w * sd[list(members)]).to_numpy()
        assert np.allclose(risk, risk[0])


def test_the_within_weights_use_nothing_after_the_rebalance_date():
    r = _returns()
    dates = _dates()
    base = S.within_sleeve_weights(r, TOY, dates, lookback=252)
    tau = dates[1]
    shocked = r.copy()
    shocked.loc[shocked.index > tau] *= 5.0
    moved = S.within_sleeve_weights(shocked, TOY, dates, lookback=252)
    assert np.allclose(base.loc[:tau], moved.loc[:tau])
    assert not np.allclose(base.loc[dates[2]], moved.loc[dates[2]])


def test_an_instrument_with_too_few_returns_leaves_its_sleeve():
    r = _returns()
    r.loc[SESSIONS[100]:SESSIONS[290], "^BBB"] = np.nan
    within = S.within_sleeve_weights(r, TOY, [SESSIONS[300]], lookback=252, min_valid=0.5)
    assert within.loc[SESSIONS[300], "^BBB"] == 0.0
    assert within.loc[SESSIONS[300], "^AAA"] == pytest.approx(1.0)


def test_estimates_never_reach_before_the_declared_start():
    r = _returns()
    start = SESSIONS[200]
    within = S.within_sleeve_weights(r, TOY, [SESSIONS[300], SESSIONS[500]], lookback=252,
                                     start=start)
    assert within.loc[SESSIONS[300]].isna().all()
    assert within.loc[SESSIONS[500]].notna().all()


def test_erc_equalises_the_sleeves_risk_contributions():
    r = _returns(seed=3)
    dates = _dates()
    within = S.within_sleeve_weights(r, TOY, dates, lookback=252)
    across = S.blind_sleeve_weights(r, TOY, within, method="erc", lookback=252)
    tau = dates[3]
    window = r.loc[:tau].iloc[-252:]
    cov = S.sleeve_returns(window, within.loc[tau], TOY).cov(ddof=1).to_numpy()
    x = across.loc[tau].to_numpy()
    contributions = x * (cov @ x)
    assert x.sum() == pytest.approx(1.0)
    assert np.allclose(contributions / contributions.sum(), 1.0 / len(TOY), atol=1e-8)


def test_inverse_vol_across_sleeves_ignores_correlation():
    r = _returns(seed=4)
    dates = _dates()
    within = S.within_sleeve_weights(r, TOY, dates, lookback=252)
    across = S.blind_sleeve_weights(r, TOY, within, method="inverse_vol", lookback=252)
    tau = dates[0]
    window = r.loc[:tau].iloc[-252:]
    sd = S.sleeve_returns(window, within.loc[tau], TOY).std(ddof=1)
    expected = (1.0 / sd) / (1.0 / sd).sum()
    assert np.allclose(across.loc[tau], expected[across.columns])


def test_weekly_estimates_run_on_calendar_week_sums():
    r = _returns(seed=5)
    dates = _dates()
    within = S.within_sleeve_weights(r, TOY, dates, lookback=252, frequency="weekly")
    tau = dates[1]
    weekly = r.loc[:tau].iloc[-252:].groupby(pd.Grouper(freq="W-FRI")).sum(min_count=1)
    inv = 1.0 / weekly[list(TOY["eq"])].std(ddof=1)
    assert np.allclose(within.loc[tau, list(TOY["eq"])], inv / inv.sum())


# ---------------------------------------------------------------------------
# schedules and holding
# ---------------------------------------------------------------------------
def test_a_row_stamped_on_a_session_is_first_held_on_the_next_one():
    schedule = pd.DataFrame({"a": [0.3, 0.6]}, index=[SESSIONS[10], SESSIONS[20]])
    held = S.hold(schedule, SESSIONS[:30])["a"]
    assert held.iloc[:11].isna().all()
    assert (held.iloc[11:21] == 0.3).all()
    assert (held.iloc[21:] == 0.6).all()


def test_a_row_stamped_on_a_weekend_is_held_from_monday():
    saturday = pd.Timestamp("2010-01-16")
    schedule = pd.DataFrame({"a": [1.0]}, index=[saturday])
    held = S.hold(schedule, SESSIONS[:15])["a"]
    assert held.loc[:"2010-01-15"].isna().all()
    assert held.loc["2010-01-18"] == 1.0


def test_the_instrument_schedule_is_sleeve_weight_times_within_weight():
    r = _returns()
    dates = _dates()
    within = S.within_sleeve_weights(r, TOY, dates, lookback=252)
    sleeve_w = pd.DataFrame({"eq": 0.5, "bond": 0.3, "cmd": 0.2}, index=within.index)
    schedule = S.instrument_schedule(sleeve_w, within, TOY)
    assert np.allclose(schedule.sum(axis=1), 1.0)
    tau = dates[0]
    assert schedule.loc[tau, "SHY"] == pytest.approx(0.3 * within.loc[tau, "SHY"])


def test_a_sleeve_at_zero_weight_holds_nothing_even_if_undefined_inside():
    within = S.within_sleeve_weights(_returns(), TOY, _dates(), lookback=252)
    within.loc[:, list(TOY["cmd"])] = np.nan
    sleeve_w = pd.DataFrame({"eq": 0.6, "bond": 0.4, "cmd": 0.0}, index=within.index)
    schedule = S.instrument_schedule(sleeve_w, within, TOY)
    assert schedule.notna().all().all()
    assert (schedule["CL=F"] == 0.0).all()
    assert np.allclose(schedule.sum(axis=1), 1.0)


def test_the_instrument_schedule_refuses_rows_that_do_not_sum_to_one():
    within = S.within_sleeve_weights(_returns(), TOY, _dates(), lookback=252)
    bad = pd.DataFrame({"eq": 0.5, "bond": 0.3, "cmd": 0.3}, index=within.index)
    with pytest.raises(ValueError, match="sum to one"):
        S.instrument_schedule(bad, within, TOY)


# ---------------------------------------------------------------------------
# the volatility target
# ---------------------------------------------------------------------------
def _book(r: pd.DataFrame, **kw) -> S.SleeveBook:
    dates = _dates(first=260)
    within = S.within_sleeve_weights(r, TOY, dates, lookback=252)
    across = S.blind_sleeve_weights(r, TOY, within, lookback=252)
    return S.build_book(r, S.hold(S.instrument_schedule(across, within, TOY), r.index), **kw)


def test_the_book_before_t_cannot_see_returns_at_t_or_after():
    r = _returns(seed=7)
    k = 600
    base = _book(r)
    shocked = r.copy()
    shocked.iloc[k:] *= 4.0
    moved = _book(shocked)
    before = SESSIONS[:k]
    pd.testing.assert_series_equal(base.multiplier.loc[before], moved.multiplier.loc[before])
    pd.testing.assert_series_equal(base.returns.loc[before], moved.returns.loc[before])
    pd.testing.assert_frame_equal(base.weights.loc[before], moved.weights.loc[before])
    # The multiplier held on session k was fixed at the close of k-1.
    assert base.multiplier.iloc[k] == moved.multiplier.iloc[k]
    assert base.returns.iloc[k] != moved.returns.iloc[k]


def test_the_book_is_missing_during_warm_up_never_zero():
    book = _book(_returns(seed=8))
    first_held = book.unscaled_returns.first_valid_index()
    position = book.returns.index.get_loc(first_held)
    assert book.returns.iloc[: position + S.VOL_WINDOW].isna().all()
    assert book.returns.iloc[position + S.VOL_WINDOW:].notna().all()


def test_without_a_cap_the_target_is_met_on_stationary_returns():
    book = _book(_returns(seed=9), cap=np.inf)
    sd = book.returns.dropna().std(ddof=1) * np.sqrt(252)
    assert sd == pytest.approx(0.10, abs=0.01)


def test_on_a_long_only_unit_book_the_multiplier_is_the_gross_exposure():
    book = _book(_returns(seed=10))
    live = book.live
    gross = book.weights.loc[live].abs().sum(axis=1)
    assert np.allclose(gross, book.multiplier.loc[live])
    assert (book.multiplier.loc[live] <= 3.0 + 1e-12).all()
    binds = book.cap_binds.loc[live].astype(bool)
    assert np.allclose(book.multiplier.loc[live][binds], 3.0)


# ---------------------------------------------------------------------------
# costs
# ---------------------------------------------------------------------------
def test_costs_are_charged_on_held_weight_changes_at_the_vehicle_rates():
    idx = SESSIONS[:4]
    w = pd.DataFrame(
        {"^GSPC": [np.nan, 1.0, 1.5, 1.5], "CL=F": [np.nan, 0.5, 0.5, 0.0],
         "TIP": [np.nan, 2.0, 1.0, 1.0], "SHY": [np.nan, 0.0, 0.0, 3.0]},
        index=idx,
    )
    drag = S.cost_drag(w, "headline") * 10_000
    # Entry is paid: 1.0*1.0 + 0.5*1.5 + 2.0*7.5.
    assert drag.iloc[1] == pytest.approx(1.0 + 0.75 + 15.0)
    assert drag.iloc[2] == pytest.approx(0.5 * 1.0 + 1.0 * 7.5)
    assert drag.iloc[3] == pytest.approx(0.5 * 1.5 + 3.0 * 1.0)
    assert drag.iloc[0] == 0.0


def test_the_four_cost_columns():
    names = pd.Index(["^GSPC", "CL=F", "TIP", "AGG", "SHY", "GC=F"])
    head = S.cost_rates(names, "headline")
    assert head.to_dict() == {"^GSPC": 1.0, "CL=F": 1.5, "TIP": 7.5, "AGG": 7.5, "SHY": 1.0,
                              "GC=F": 1.5}
    for column in ("conservative", "stress"):
        rates = S.cost_rates(names, column)
        assert rates["^GSPC"] == COST_SCHEDULE["equity_index"][column]
        assert rates["TIP"] == COST_SCHEDULE["cash_vehicle"][column]
    assert (S.cost_rates(names, "all_cash") == 7.5).all()
    with pytest.raises(ValueError):
        S.cost_rates(names, "optimistic")


def test_net_returns_subtract_the_cost_of_the_weights_held():
    book = _book(_returns(seed=11))
    net = S.net_returns(book, "stress")
    live = book.live
    assert np.allclose((book.returns - net).loc[live], S.cost_drag(book.weights, "stress")
                       .loc[live])


def test_the_diagnostics_read_no_mean_return():
    book = _book(_returns(seed=12))
    shifted = S.SleeveBook(
        unscaled_weights=book.unscaled_weights, weights=book.weights,
        unscaled_returns=book.unscaled_returns, returns=book.returns + 0.01,
        multiplier=book.multiplier, cap_binds=book.cap_binds,
    )
    a, b = S.book_diagnostics(book), S.book_diagnostics(shifted)
    for key, value in a.items():
        if isinstance(value, float):
            assert b[key] == pytest.approx(value, rel=1e-9, abs=1e-12), key


def test_the_equal_weight_blend_of_the_headline_prices_agg_and_tip_as_cash_vehicles():
    names = S.instruments(S.HEADLINE_SLEEVES)
    assert len(names) == 30
    expected = (12 * 1.0 + 3 * 1.0 + 2 * 7.5 + 13 * 1.5) / 30
    assert S.blended_cost_bps(names) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# the universe
# ---------------------------------------------------------------------------
def test_the_declared_configurations():
    assert [len(S.instruments(S.sleeve_map(c))) for c in S.CONFIGURATIONS] == [30, 33, 44]
    with pytest.raises(ValueError):
        S.sleeve_map("everything")
    with pytest.raises(ValueError, match="two sleeves"):
        S.instruments({"a": ("X",), "b": ("X",)})


@pytest.mark.skipif(not S.PANEL.exists(), reason="data/ absent: stored panel not on disk")
def test_the_stored_panel_matches_the_declared_sample():
    stored = pd.read_parquet(S.PANEL.parent / "trend_universe.parquet")
    prices = S.load_prices()
    for name in prices.columns:
        assert np.array_equal(prices[name].to_numpy(), stored[name].to_numpy(), equal_nan=True)
    start = S.configuration_start(prices, S.HEADLINE_SLEEVES)
    assert start == pd.Timestamp("2003-12-05")
    assert (prices.index >= start).sum() == 5938

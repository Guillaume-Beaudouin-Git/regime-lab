"""Tests of the Bridgewater quadrant: surprise arithmetic, stamps, point-in-time labels.

Everything runs on synthetic inputs except the last block, which re-reads the stored
SPF files and skips when `data/raw/spf/` is absent. No test reads a return.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

from regime_lab.config import RAW
from regime_lab.construction import quadrant as qd

SPF = RAW / "spf"


def spf_frame(start: str, **columns: list[float]) -> pd.DataFrame:
    n = len(next(iter(columns.values())))
    index = pd.period_range(start, periods=n, freq="Q", name="survey")
    return pd.DataFrame(columns, index=index)


def xlsx_bytes(frames: dict[str, pd.DataFrame], *, header: bool = True) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for sheet, frame in frames.items():
            frame.to_excel(writer, sheet_name=sheet, index=False, header=header)
    return buffer.getvalue()


# ------------------------------------------------------------------- surprises


def test_growth_surprise_is_next_survey_level_minus_nowcast_over_base():
    rgdp = spf_frame("2000Q1", RGDP1=[100.0, 101.0, 102.5], RGDP2=[100.5, 102.0, 103.0])
    growth = qd.growth_surprise(rgdp)
    assert growth.loc[pd.Period("2000Q1")] == pytest.approx((101.0 - 100.5) / 100.0 * 400)
    assert growth.loc[pd.Period("2000Q2")] == pytest.approx((102.5 - 102.0) / 101.0 * 400)
    assert np.isnan(growth.loc[pd.Period("2000Q3")])  # survey 2000Q4 not yet out


def test_denominator_never_changes_the_sign():
    rng = np.random.default_rng(0)
    level = 100 + np.cumsum(rng.normal(0.5, 1.0, 40))
    rgdp = spf_frame("1990Q1", RGDP1=list(level), RGDP2=list(level + rng.normal(0.5, 1.0, 40)))
    base = qd.growth_surprise(rgdp, denominator="base").dropna()
    nowcast = qd.growth_surprise(rgdp, denominator="nowcast").dropna()
    assert (np.sign(base) == np.sign(nowcast)).all()
    assert not np.allclose(base, nowcast)


def test_inflation_surprise_is_next_survey_rate_minus_nowcast():
    cpi = spf_frame("2000Q1", CPI1=[2.0, 3.0, 1.0], CPI2=[2.5, 2.5, 2.5])
    infl = qd.inflation_surprise(cpi)
    assert infl.loc[pd.Period("2000Q1")] == pytest.approx(0.5)
    assert infl.loc[pd.Period("2000Q2")] == pytest.approx(-1.5)


def test_within_vintage_growth_is_first_release_minus_drgdp2():
    drgdp2 = pd.Series([2.0, 1.0], index=pd.period_range("2000Q1", periods=2, freq="Q"))
    first = pd.Series([2.5, np.nan, 9.0], index=pd.period_range("2000Q1", periods=3, freq="Q"))
    out = qd.growth_surprise_within_vintage(first, drgdp2)
    assert out.iloc[0] == pytest.approx(0.5)
    assert np.isnan(out.iloc[1])
    assert len(out) == 2


def test_surprise_panel_keeps_rows_with_both_axes():
    rgdp = spf_frame("2000Q1", RGDP1=[100.0, 101.0, 102.0], RGDP2=[100.5, 101.5, 102.5])
    cpi = spf_frame("2000Q1", CPI1=[np.nan, 3.0, 1.0], CPI2=[np.nan, 2.5, 2.5])
    panel = qd.surprise_panel(rgdp, cpi)
    assert list(panel.index.astype(str)) == ["2000Q2"]  # 2000Q1: no CPI; 2000Q3: no q+1


def test_spf_table_refuses_a_gap():
    frame = pd.DataFrame({"YEAR": [2000, 2000, 2000], "QUARTER": [1, 2, 4], "CPI1": [1, 2, 3]})
    with pytest.raises(qd.SpfFileError, match="consecutive"):
        qd.spf_table(frame)


# ---------------------------------------------------------------------- coding


def test_quadrant_code_and_ties():
    growth = pd.Series([-1.0, -1.0, 1.0, 1.0, 0.0, np.nan, 1.0])
    infl = pd.Series([-1.0, 1.0, -1.0, 1.0, 0.0, 1.0, np.nan])
    code = qd.quadrant_code(growth, infl)
    assert code.iloc[:5].tolist() == [0, 1, 2, 3, 0]  # a zero counts as negative
    assert code.iloc[5:].isna().all()


# ---------------------------------------------------------------- availability


def test_draft_stamp_is_the_fifteenth_of_the_second_month_of_q_plus_1():
    stamps = qd.availability(pd.PeriodIndex(["2003Q3", "2019Q4"], freq="Q"), rule="draft")
    assert stamps["available_at"].tolist() == [pd.Timestamp("2003-11-15"),
                                               pd.Timestamp("2020-02-15")]


def test_release_stamp_uses_survey_q_plus_1_and_falls_back_to_its_quarter_end():
    table = pd.DataFrame({
        "deadline": [pd.Timestamp("2000-05-13")], "release": [pd.Timestamp("2000-05-22")],
        "note": [""],
    }, index=pd.PeriodIndex(["2000Q2"], freq="Q", name="survey"))
    quarters = pd.PeriodIndex(["1999Q4", "2000Q1"], freq="Q")
    stamps = qd.availability(quarters, rule="release", release_dates=table)
    assert stamps.loc[pd.Period("2000Q1"), "available_at"] == pd.Timestamp("2000-05-22")
    assert stamps.loc[pd.Period("2000Q1"), "source"] == "release"
    assert stamps.loc[pd.Period("1999Q4"), "available_at"] == pd.Timestamp("2000-03-31")
    assert stamps.loc[pd.Period("1999Q4"), "source"] == "fallback_quarter_end"
    later = qd.availability(quarters, rule="release", release_dates=table, extra_months=1)
    assert later.loc[pd.Period("2000Q1"), "available_at"] == pd.Timestamp("2000-06-22")
    with pytest.raises(ValueError):
        qd.availability(quarters, rule="release")


# ---------------------------------------------------------------- daily labels


def quarterly(labels: list[int], stamps: list[str], start: str = "2000Q1") -> pd.DataFrame:
    index = pd.period_range(start, periods=len(labels), freq="Q", name="quarter")
    return pd.DataFrame({"label": pd.array(labels, dtype="Int64"),
                         "available_at": pd.to_datetime(stamps)}, index=index)


def test_no_label_is_visible_before_its_stamp_and_lag_one_trades_the_next_session():
    q = quarterly([1, 3], ["2000-05-17", "2000-08-16"])  # Wednesdays
    sessions = pd.bdate_range("2000-05-15", "2000-08-18")
    known = qd.daily_labels(q, sessions, lag=0)
    traded = qd.daily_labels(q, sessions, lag=1)
    assert known.loc[:"2000-05-16"].isna().all()
    assert known.loc["2000-05-17"] == 1  # known on its release day
    assert known.loc["2000-08-15"] == 1 and known.loc["2000-08-16"] == 3
    assert traded.loc[:"2000-05-17"].isna().all()  # traded from the next session only
    assert traded.loc["2000-05-18"] == 1
    assert traded.loc["2000-08-16"] == 1 and traded.loc["2000-08-17"] == 3
    # every traded label was known on an earlier session
    stamps = q.set_index("label")["available_at"]
    for day, lab in traded.dropna().items():
        assert stamps.loc[lab] < day


def test_weekend_stamp_is_first_seen_on_the_next_session():
    q = quarterly([2], ["2000-05-20"])  # a Saturday
    sessions = pd.bdate_range("2000-05-18", "2000-05-24")
    known = qd.daily_labels(q, sessions, lag=0)
    assert known.loc[:"2000-05-19"].isna().all() and known.loc["2000-05-22"] == 2
    assert qd.daily_labels(q, sessions, lag=1).loc["2000-05-23"] == 2


def test_two_quarters_on_one_stamp_hold_the_later_quarter():
    q = quarterly([0, 3], ["1990-08-31", "1990-08-31"], start="1990Q1")
    sessions = pd.bdate_range("1990-08-30", "1990-09-05")
    known = qd.daily_labels(q, sessions, lag=0)
    assert known.dropna().unique().tolist() == [3]


def test_quarterly_labels_refuse_decreasing_stamps():
    panel = pd.DataFrame({"growth": [1.0, -1.0], "inflation": [1.0, 1.0]},
                         index=pd.period_range("2000Q1", periods=2, freq="Q", name="quarter"))
    stamps = pd.DataFrame({"available_at": pd.to_datetime(["2000-08-01", "2000-05-01"]),
                           "source": ["release", "release"]}, index=panel.index)
    with pytest.raises(ValueError, match="decrease"):
        qd.quarterly_labels(panel, stamps)


def test_quarterly_labels_drop_undefined_quarters():
    panel = pd.DataFrame({"growth": [1.0, np.nan, -1.0], "inflation": [1.0, 1.0, -1.0]},
                         index=pd.period_range("2000Q1", periods=3, freq="Q", name="quarter"))
    stamps = qd.availability(panel.index, rule="draft")
    out = qd.quarterly_labels(panel, stamps)
    assert out["label"].tolist() == [3, 0]


# ------------------------------------------------------------------ statistics


def test_clock_counts_spells_and_both_persistence_denominators():
    labels = pd.Series([0, 0, 1, 1, 1, 2, 0],
                       index=pd.date_range("2000-01-01", periods=7, freq="91D"))
    stats = qd.clock(labels)
    assert stats.transitions == 3
    assert stats.spell_median == 1.5 and stats.spell_max == 3
    assert stats.same_share == pytest.approx(3 / 6)
    assert stats.same_share_n == pytest.approx(3 / 7)
    assert stats.counts == {0: 3, 1: 3, 2: 1, 3: 0}
    assert stats.independent_same == pytest.approx((9 + 9 + 1) / 49)
    assert stats.per_year == pytest.approx(3 / (6 * 91 / 365.25))


def test_clock_drops_a_missing_head_and_refuses_an_interior_gap():
    head = pd.Series(pd.array([pd.NA, 1, 1, 2], dtype="Int64"),
                     index=pd.date_range("2000-01-03", periods=4))
    assert qd.clock(head).n == 3
    gap = pd.Series(pd.array([1, pd.NA, 2], dtype="Int64"),
                    index=pd.date_range("2000-01-03", periods=3))
    with pytest.raises(ValueError):
        qd.clock(gap)


def test_composition_decade_and_quarter_end_sample():
    index = pd.bdate_range("2009-12-28", "2010-01-08")
    labels = pd.Series(pd.array([0, 1, 1, 2, 2, 3, 3, 3, 3, 3], dtype="Int64"), index=index)
    table = qd.composition(labels, qd.decade(index))
    assert table.loc["2000s"].tolist() == [1, 2, 1, 0]  # 28-31 Dec 2009
    assert table.loc["2010s"].tolist() == [0, 0, 1, 5]  # 1 Jan to 8 Jan 2010
    sample = qd.quarter_end_sample(labels)
    assert sample.index.tolist() == [pd.Timestamp("2009-12-31"), pd.Timestamp("2010-01-08")]


# --------------------------------------------------------------------------- B1


def test_b1_change_labels_sign_conventions_and_lag():
    sessions = pd.bdate_range("2010-01-04", periods=6)
    panel = pd.DataFrame({"breakeven": [2.0, 2.1, 2.2, 2.1, 2.0, 2.0],
                          "baa_spread": [3.0, 2.9, 2.8, 2.9, 3.0, 3.0]}, index=sessions)
    known = qd.b1_change_labels(panel, 1, lag=0)
    # spread falling -> growth+, breakeven rising -> inflation+ ; unchanged -> negative
    assert known.iloc[1:].tolist() == [3, 3, 0, 0, 0]
    traded = qd.b1_change_labels(panel, 1, lag=1)
    assert traded.iloc[2:].tolist() == known.iloc[1:-1].tolist()
    assert traded.iloc[:2].isna().all()


def test_b1_level_labels_against_expanding_median():
    sessions = pd.bdate_range("2010-01-04", periods=5)
    panel = pd.DataFrame({"breakeven": [1.0, 2.0, 3.0, 4.0, 0.5],
                          "baa_spread": [5.0, 4.0, 3.0, 2.0, 9.0]}, index=sessions)
    out = qd.b1_level_labels(panel, min_periods=3, lag=0)
    assert out.iloc[:2].isna().all()
    assert out.iloc[2:].tolist() == [3, 3, 0]


def test_market_panel_holds_the_previous_close():
    sessions = pd.bdate_range("2010-01-04", periods=3)
    def pit(sid: str, values: list[float]) -> pd.DataFrame:
        return pd.DataFrame({"series_id": sid, "period": sessions,
                             "available_at": sessions + pd.Timedelta(days=1), "value": values})
    panel = qd.market_panel(pit("T10YIE", [2.0, 2.1, 2.2]),
                            pit("fin_baa_spread", [3.0, 3.1, 3.2]), sessions)
    assert np.isnan(panel["breakeven"].iloc[0])
    assert panel["breakeven"].iloc[1:].tolist() == [2.0, 2.1]


# ------------------------------------------------------------------ file checks


RELEASE_TEXT = (
    "Deadline and Release Dates for the Survey of Professional Forecasters\r\n"
    "True deadline and news release dates for surveys prior to 1990:Q2 are not known.\r\n\r\n"
    "Survey         True Deadline Date       News Release Date\r\n\r\n"
    "1999 Q3             8/14/99             8/23/99\r\n"
    "     Q4             11/13/99            11/19/99\r\n\r\n"
    "2000 Q1             2/12/00             2/22/00\r\n"
    "     Q2\t\t    5/13/00\t\t5/22/00  \r\n"
    "     Q3             8/12/00**           8/21/00**\r\n\r\n"
    "**Delayed by a shutdown.\r\n"
)


def test_parse_release_dates():
    table = qd.parse_release_dates(RELEASE_TEXT)
    assert list(table.index.astype(str)) == ["1999Q3", "1999Q4", "2000Q1", "2000Q2", "2000Q3"]
    assert table.loc[pd.Period("2000Q1"), "release"] == pd.Timestamp("2000-02-22")
    assert table.loc[pd.Period("2000Q2"), "deadline"] == pd.Timestamp("2000-05-13")
    assert table.loc[pd.Period("2000Q3"), "note"] == "**"
    csv = qd.release_dates_csv(table, "https://example.org/x.txt")
    assert csv["source_url"].unique().tolist() == ["https://example.org/x.txt"]


def test_release_dates_round_trip(tmp_path):
    table = qd.parse_release_dates(RELEASE_TEXT)
    path = tmp_path / "dates.csv"
    qd.release_dates_csv(table, "u").to_csv(path, index=False)
    back = qd.read_release_dates(path)
    pd.testing.assert_frame_equal(back, table, check_freq=False)


@pytest.mark.parametrize("text,match", [
    ("<!DOCTYPE html><html><title>Error - 404</title></html>", "HTML"),
    ("Some other table\n1999 Q3  8/14/99  8/23/99\n", "first line"),
    (RELEASE_TEXT.replace("     Q4             11/13/99            11/19/99\r\n", ""),
     "consecutive"),
    (RELEASE_TEXT.replace("8/23/99", "7/23/99"), "deadline"),
])
def test_parse_release_dates_refuses(text, match):
    with pytest.raises(qd.SpfFileError, match=match):
        qd.parse_release_dates(text)


def test_verify_workbook_accepts_a_real_xlsx_and_refuses_the_404_page():
    frame = pd.DataFrame({"YEAR": [1968, 1969], "QUARTER": [4, 1], "CPI1": [1.0, 2.0],
                          "CPI2": [1.5, 2.5]})
    out = qd.verify_workbook(xlsx_bytes({"Median_Level": frame}), required=("CPI1", "CPI2"),
                             label="ok.xlsx")
    assert out.shape == (2, 4)
    page = b"<!DOCTYPE html>\n<html><head><title>Error - 404</title></head></html>"
    with pytest.raises(qd.SpfFileError, match="Error - 404"):
        qd.verify_workbook(page, required=("CPI1",), label="bad.xlsx")
    with pytest.raises(qd.SpfFileError, match="ZIP"):
        qd.verify_workbook(b"garbage bytes", required=("CPI1",), label="bad.xlsx")
    with pytest.raises(qd.SpfFileError, match="missing columns"):
        qd.verify_workbook(xlsx_bytes({"s": frame}), required=("RGDP1",), label="x.xlsx")


def test_read_rtdsm_first_release():
    notes = pd.DataFrame({0: ["Real GNP/GDP", "Variable ID: ROUTPUT"]})
    data = pd.DataFrame([
        ["Real GNP/GDP (ROUTPUT)", None, None, None, None],
        [None, None, None, None, None],
        ["Date", "First", "Second", "Third", "Most_Recent"],
        ["2000:Q1", 5.0, 5.1, 5.2, 4.0],
        ["2000:Q2", None, None, None, 1.0],
        ["2000:Q3", -1.0, -1.1, -1.2, -0.5],
    ])
    content = xlsx_bytes({"NOTES": notes, "DATA": data}, header=False)
    first = qd.read_rtdsm_first_release(content)
    assert list(first.index.astype(str)) == ["2000Q1", "2000Q2", "2000Q3"]
    assert first.iloc[0] == 5.0 and np.isnan(first.iloc[1]) and first.iloc[2] == -1.0


# -------------------------------------------------------- stored files (skip)

needs_spf = pytest.mark.skipif(
    not (SPF / "median_rgdp_level.xlsx").exists() or not (SPF / qd.RELEASE_DATES_CSV).exists(),
    reason="data/raw/spf absent (run scripts/fetch_spf.py)",
)


@needs_spf
def test_stored_spf_reproduces_the_draft_panel():
    rgdp = qd.read_spf(SPF / "median_rgdp_level.xlsx", ("RGDP1", "RGDP2"))
    cpi = qd.read_spf(SPF / "median_cpi_level.xlsx", ("CPI1", "CPI2"))
    panel = qd.surprise_panel(rgdp, cpi)
    assert len(panel) >= 180
    first180 = panel.iloc[:180]
    assert str(first180.index[0]) == "1981Q3" and str(first180.index[-1]) == "2026Q2"
    assert round(first180["growth"].std(), 3) == 35.481
    assert round(first180["inflation"].std(), 3) == 1.333


@needs_spf
def test_stored_release_dates_and_the_declared_object_is_point_in_time():
    table = qd.read_release_dates(SPF / qd.RELEASE_DATES_CSV)
    assert str(table.index[0]) == "1990Q2"
    assert table.loc[pd.Period("2019Q1"), "release"] == pd.Timestamp("2019-03-22")
    rgdp = qd.read_spf(SPF / "median_rgdp_level.xlsx", ("RGDP1", "RGDP2"))
    cpi = qd.read_spf(SPF / "median_cpi_level.xlsx", ("CPI1", "CPI2"))
    panel = qd.surprise_panel(rgdp, cpi)
    stamps = qd.availability(panel.index, rule="release", release_dates=table)
    ql = qd.quarterly_labels(panel, stamps)
    sessions = pd.bdate_range("2003-12-01", "2026-09-10")
    traded = qd.daily_labels(ql, sessions, lag=1)
    assert ql["available_at"].is_monotonic_increasing
    held = ql.reset_index().drop_duplicates("available_at", keep="last")
    stamps = held["available_at"].to_numpy()
    # the label traded on session t is the latest one released on or before session t-1,
    # hence strictly before t
    previous = sessions[:-1].to_numpy()
    pos = np.searchsorted(stamps, previous, side="right") - 1
    got = traded.iloc[1:].to_numpy()
    assert (pos >= 0).all() and not pd.isna(got).any()
    assert (got.astype(int) == held["label"].to_numpy()[pos].astype(int)).all()
    assert (stamps[pos] < sessions[1:].to_numpy()).all()

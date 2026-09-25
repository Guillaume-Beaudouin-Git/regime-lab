"""The Bridgewater declared objects: content hashes, quarter folds, masks, B1, the witness.

Synthetic data only, except the last test, which skips cleanly when ``data/`` is absent.
What is tested:
- the content hash of an input ignores rows appended beyond the cut-off and sees a
  change to a used value;
- the contemporaneous path gives every session its calendar quarter's label;
- quarter folds are cut at quarter ends, count only public quarters, and refuse an
  impossible rule;
- a fold fits only on labels public by its last training session, and reads every test
  session;
- the quarterly blocks, B1's quarterly sampling, the witness's training cut-offs and the
  traded blind driver.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

from regime_lab.config import ROOT
from regime_lab.construction import declared as D
from regime_lab.construction import evaluate as E
from regime_lab.construction import quadrant as Q

# ---------------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def sessions() -> pd.DatetimeIndex:
    return pd.bdate_range("2004-01-02", "2019-12-31")


@pytest.fixture(scope="module")
def partition(sessions) -> D.Partition:
    """iid quarterly labels, each public 45 days after its quarter ends."""
    quarters = pd.period_range("2004Q1", "2019Q4", freq="Q")
    rng = np.random.default_rng(0)
    labels = pd.Series(rng.choice(4, len(quarters)).astype(float), index=quarters, name="q")
    available = pd.Series(quarters.end_time.normalize() + pd.Timedelta(days=45), index=quarters)
    return D.Partition("q", labels, available)


def _t10yie_bytes(periods: pd.DatetimeIndex, values: np.ndarray) -> bytes:
    frame = pd.DataFrame({"series_id": "T10YIE", "period": periods,
                          "available_at": periods + pd.Timedelta(days=1), "value": values})
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False)
    return buffer.getvalue()


def _spf_bytes(quarters: pd.PeriodIndex, values: np.ndarray) -> bytes:
    frame = pd.DataFrame({"YEAR": quarters.year, "QUARTER": quarters.quarter,
                          "RGDP1": values, "RGDP2": values + 1.0})
    buffer = io.BytesIO()
    frame.to_excel(buffer, index=False)
    return buffer.getvalue()


# ------------------------------------------------------------------ content hashes


def test_the_content_hash_ignores_appended_rows_and_sees_revisions():
    periods = pd.bdate_range("2026-09-01", D.T10YIE_CUTOFF)
    values = np.linspace(2.0, 2.3, len(periods))
    name = "data/raw/spf/fred_t10yie.parquet"
    base = D.content_of(name, _t10yie_bytes(periods, values))
    later = pd.bdate_range("2026-09-01", D.T10YIE_CUTOFF + pd.Timedelta(days=5))
    longer = np.r_[values, np.full(len(later) - len(periods), 9.9)]
    assert D.content_of(name, _t10yie_bytes(later, longer)) == base
    revised = values.copy()
    revised[3] += 1e-12
    assert D.content_of(name, _t10yie_bytes(periods, revised)) != base
    # The SPF medians: surveys after the cut-off do not count, a used value does.
    quarters = pd.period_range("2025Q1", D.SURVEY_CUTOFF, freq="Q")
    spf = "data/raw/spf/median_rgdp_level.xlsx"
    first = D.content_of(spf, _spf_bytes(quarters, np.arange(len(quarters), dtype=float)))
    grown = pd.period_range("2025Q1", D.SURVEY_CUTOFF + 1, freq="Q")
    assert D.content_of(spf, _spf_bytes(grown, np.arange(len(grown), dtype=float))) == first
    moved = np.arange(len(quarters), dtype=float)
    moved[0] = 99.0
    assert D.content_of(spf, _spf_bytes(quarters, moved)) != first
    # A cached input is hashed on its bytes.
    assert D.content_of("data/cache/x.parquet", b"abc") == b"abc"


def test_stamped_series_keeps_the_later_quarter_on_a_shared_stamp():
    quarterly = pd.DataFrame(
        {"label": [1, 2, 3], "available_at": pd.to_datetime(["1990-08-31", "1990-08-31",
                                                            "1990-11-20"])},
        index=pd.period_range("1990Q1", periods=3, freq="Q"))
    stamped = D.stamped_series(quarterly)
    assert stamped.tolist() == [2.0, 3.0]
    assert stamped.index.is_monotonic_increasing and stamped.index.is_unique


# ------------------------------------------------------------- the partition


def test_the_contemporaneous_path_is_the_calendar_quarter(sessions, partition):
    path = partition.path(sessions)
    for day in ("2005-02-14", "2011-09-30", "2019-12-31"):
        quarter = pd.Timestamp(day).to_period("Q")
        assert path.loc[day] == partition.labels.loc[quarter]
    short = D.Partition("q", partition.labels.loc[:"2010Q4"], partition.available)
    assert short.path(sessions).loc["2011-01-03":].isna().all()


def test_quarter_folds_cut_at_quarter_ends_and_count_public_quarters(sessions, partition):
    folds = D.quarter_folds(partition.labels, partition.available, sessions)
    assert len(folds) == 5
    previous = None
    for fold in folds:
        assert fold.test_start.to_period("Q").start_time.normalize() <= fold.test_start
        assert fold.test_start == sessions[sessions >= fold.test_start.to_period("Q")
                                           .start_time][0]
        assert fold.test_end == sessions[sessions <= fold.test_end.to_period("Q")
                                         .end_time][-1]
        assert fold.train_end == sessions[sessions < fold.test_start][-1]
        if previous is not None:
            assert fold.test_start == sessions[sessions > previous][0]
        previous = fold.test_end
    assert folds[-1].test_end == sessions[-1]
    # The rule: four PUBLIC quarters of every cell before the first test quarter, and not
    # one quarter earlier.
    labels, avail = partition.labels, partition.available
    cut = folds[0].train_end
    public = labels[(avail <= cut) & (labels.index < folds[0].test_start.to_period("Q"))]
    assert (public.value_counts().reindex(range(4), fill_value=0) >= 4).all()
    prior = folds[0].test_start.to_period("Q") - 1
    cut_before = sessions[sessions < prior.start_time][-1]
    earlier = labels[(avail <= cut_before) & (labels.index < prior)]
    assert (earlier.value_counts().reindex(range(4), fill_value=0) < 4).any()
    with pytest.raises(ValueError, match="every cell"):
        D.quarter_folds(labels, avail, sessions, min_quarters=40)


def test_usable_sessions_delay_the_counted_quarters(sessions, partition):
    base = D.quarter_folds(partition.labels, partition.available, sessions, min_quarters=3)
    later = D.quarter_folds(partition.labels, partition.available, sessions, min_quarters=3,
                            usable=sessions[sessions >= "2008-01-01"])
    assert later[0].test_start > base[0].test_start


def test_a_fold_fits_only_on_public_labels_and_reads_every_test_session(sessions, partition):
    folds = D.quarter_folds(partition.labels, partition.available, sessions)
    paths = D.fold_paths(partition.labels, partition.available, sessions, folds)
    full = partition.path(sessions)
    for fold in folds:
        path = paths[fold.number]
        test = fold.test(sessions)
        pd.testing.assert_series_equal(path.loc[test], full.loc[test], check_names=False)
        train = sessions[sessions <= fold.train_end]
        quarters = D.session_quarters(train)
        public = np.asarray(pd.DatetimeIndex(partition.available.reindex(quarters))
                            <= fold.train_end)
        assert path.loc[train][public].notna().all()
        assert path.loc[train][~public].isna().all()
        # The quarter just before the test is released 45 days into the test: masked.
        last = fold.test_start.to_period("Q") - 1
        assert path.loc[last.start_time:fold.train_end].isna().all()
        assert path.loc[fold.test_end:].iloc[1:].isna().all()


def test_quarter_blocks_follow_the_test_windows(sessions, partition):
    folds = D.quarter_folds(partition.labels, partition.available, sessions)
    blocks = D.quarter_blocks(partition.labels.index, sessions, folds)
    first_test = folds[0].test_start.to_period("Q")
    assert (blocks[blocks.index < first_test] == "0:train").all()
    for fold in folds:
        inside = pd.period_range(fold.test_start, fold.test_end, freq="Q")
        assert (blocks.loc[inside] == f"{fold.number}:test").all()
    assert D.last_test_quarter(folds) == pd.Period("2019Q4", freq="Q")


def test_through_cuts_a_partition_after_a_quarter(partition):
    cut = partition.through(pd.Period("2010Q2", freq="Q"))
    assert cut.labels.index[-1] == pd.Period("2010Q2", freq="Q")
    assert cut.available.index.equals(cut.labels.index)


# ------------------------------------------------------------------------ B1


def test_b1_is_sampled_on_each_quarter_last_session(sessions):
    rng = np.random.default_rng(1)
    days = pd.bdate_range("2003-06-02", sessions[-1])

    def frame(series_id: str, values: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame({"series_id": series_id, "period": days,
                             "available_at": days + pd.Timedelta(days=1), "value": values})

    t10 = frame("T10YIE", 2.0 + np.cumsum(rng.normal(0, 0.02, len(days))))
    baa = frame("BAA10Y", 2.5 + np.cumsum(rng.normal(0, 0.02, len(days))))
    b1 = D.b1_partition(t10, baa, sessions)
    panel = Q.market_panel(t10, baa, sessions)
    daily = Q.b1_change_labels(panel, D.B1_WINDOW, lag=0)
    for quarter in b1.labels.index[::7]:
        last = sessions[sessions <= quarter.end_time][-1]
        assert b1.labels.loc[quarter] == float(daily.loc[last])
        assert b1.available.loc[quarter] == last
    first_defined = daily.dropna().index[0]
    assert b1.labels.index[0] == first_defined.to_period("Q")
    cut = D.b1_partition(t10, baa, sessions, through=pd.Period("2015Q4", freq="Q"))
    assert cut.labels.index[-1] == pd.Period("2015Q4", freq="Q")


# ------------------------------------------------------------------- the witness


def test_the_quarterly_witness_uses_public_training_quarters(sessions, partition):
    folds = D.quarter_folds(partition.labels, partition.available, sessions)
    rng = np.random.default_rng(2)
    driver = pd.Series(rng.standard_normal(len(sessions))
                       * np.repeat(rng.uniform(0.5, 2.0, 16 * 4), 70)[: len(sessions)] * 0.01,
                       index=sessions)
    last = D.quarter_last_sessions(partition.labels.index, sessions)
    per_fold, stitched = D.quarterly_volatility_labels(driver, sessions, folds, last)
    # Changing a quarter after fold 1's cut leaves fold 1's cut-offs, hence the bins of
    # every earlier quarter, unchanged.
    moved = driver.copy()
    moved.loc[folds[0].test_start:] *= 5.0
    again, _ = D.quarterly_volatility_labels(moved, sessions, folds, last)
    before = per_fold[1].index < folds[0].test_start.to_period("Q")
    pd.testing.assert_series_equal(again[1][before], per_fold[1][before])
    assert set(np.unique(per_fold[1])) <= {0.0, 1.0, 2.0, 3.0}
    blocks = D.quarter_blocks(stitched.index, sessions, folds)
    for fold in folds:
        mine = blocks == f"{fold.number}:test"
        pd.testing.assert_series_equal(stitched[mine], per_fold[fold.number][mine])


def test_the_traded_blind_driver_is_the_blind_path_times_the_sleeves(sessions, partition):
    folds = D.quarter_folds(partition.labels, partition.available, sessions)
    rng = np.random.default_rng(3)
    returns = pd.DataFrame(rng.standard_normal((len(sessions), 5)) * 0.01, index=sessions,
                           columns=list(E.SLEEVES))
    returns.iloc[10, 2] = np.nan
    engine = E.LevelA(returns, folds)
    driver = D.traded_blind_driver(engine)
    w = engine.blind_weights
    t = folds[2].test(sessions)[5]
    assert driver.loc[t] == pytest.approx(float(returns.loc[t] @ w.loc[3]))
    t0 = sessions[20]
    assert driver.loc[t0] == pytest.approx(float(returns.loc[t0] @ w.loc[1]))
    assert np.isnan(driver.iloc[10])
    assert driver.loc[folds[-1].test_end:].iloc[1:].isna().all()


# ------------------------------------------------------------------ on the data


DATA = ROOT / "data" / "raw" / "spf" / "median_rgdp_level.xlsx"


@pytest.mark.skipif(not DATA.exists(), reason="data/ is absent (gitignored)")
def test_the_declared_objects_on_the_stored_data():
    inputs = D.load_quadrant_inputs()
    obj = D.contemporaneous_object(inputs=inputs)
    s = obj.sessions
    assert len(s) == 5938 and len(obj.folds) == 5
    assert all(f.test_start == s[s >= f.test_start.to_period("Q").start_time][0]
               for f in obj.folds)
    assert obj.folds[-1].test_end == pd.Timestamp("2026-06-30")
    paths = obj.paths()
    assert all(paths[f.number].loc[f.test(s)].notna().all() for f in obj.folds)
    stamped = D.declared_object(inputs=inputs)
    assert len(stamped.test_sessions()) == 4126
    assert stamped.stamped.index.equals(obj.stamped.index)

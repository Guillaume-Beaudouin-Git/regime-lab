"""M1 must move the right columns, in the right direction, and no others.

The direction is the part worth a test: shifting the wrong way would double the
defect instead of removing it, and both directions produce a plausible-looking
panel. The anchor is a dated event — the pound fell on 24 June 2016, and the
stored series records that fall on the 27th.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regime_lab.extensions.repair import (
    FX_ALIGNED,
    FX_LATE,
    FX_LATE_INFERRED,
    FX_LATE_VERIFIED,
    realign_fx,
    realignment_report,
)

SESSIONS = pd.bdate_range("2016-06-20", periods=8)


def _panel() -> pd.DataFrame:
    """A panel whose GBP column carries the Brexit fall one session late.

    The levels are the ones actually stored for GBPUSD=X, so the dates line up
    with the event: sessions run Mon 20 June to Wed 29 June 2016, and the stored
    series puts -1.6% on Friday the 24th and -7.6% on Monday the 27th.
    """
    gbp = pd.Series(
        [148.00, 147.20, 146.70, 147.89, 145.58, 134.51, 132.35, 133.00], index=SESSIONS
    )
    return pd.DataFrame(
        {
            "GBPUSD=X": gbp,
            "CAD=X": pd.Series(np.linspace(128.0, 131.0, len(SESSIONS)), index=SESSIONS),
            "^GSPC": pd.Series(np.linspace(2100.0, 2050.0, len(SESSIONS)), index=SESSIONS),
        }
    )


def test_the_brexit_fall_moves_onto_the_session_it_happened():
    panel = _panel()
    friday, monday = pd.Timestamp("2016-06-24"), pd.Timestamp("2016-06-27")

    stored = panel["GBPUSD=X"].pct_change()
    assert stored.loc[friday] > -0.02, "the stored series should be flat on the day itself"
    assert stored.loc[monday] < -0.07, "the stored series should carry the fall on the Monday"

    repaired = realign_fx(panel)["GBPUSD=X"].pct_change()
    assert repaired.loc[friday] < -0.07, "the repair must put the fall on the Friday"


def test_the_shift_is_backwards_and_not_forwards():
    panel = _panel()
    repaired = realign_fx(panel)
    shifted_the_wrong_way = panel["GBPUSD=X"].shift(1)
    assert not repaired["GBPUSD=X"].equals(shifted_the_wrong_way)
    pd.testing.assert_series_equal(
        repaired["GBPUSD=X"], panel["GBPUSD=X"].shift(-1), check_names=False
    )


def test_the_aligned_series_are_left_exactly_alone():
    panel = _panel()
    repaired = realign_fx(panel)
    for column in (*FX_ALIGNED, "^GSPC"):
        if column in panel.columns:
            pd.testing.assert_series_equal(repaired[column], panel[column])


def test_the_verified_scope_leaves_the_inferred_series_unshifted():
    index = pd.bdate_range("2020-01-01", periods=10)
    panel = pd.DataFrame(
        {c: pd.Series(np.arange(10.0), index=index) for c in FX_LATE_VERIFIED + FX_LATE_INFERRED}
    )
    repaired = realign_fx(panel, scope="verified")
    for column in FX_LATE_VERIFIED:
        assert repaired[column].iloc[0] == 1.0, "verified columns must still be shifted"
    for column in FX_LATE_INFERRED:
        assert repaired[column].iloc[0] == 0.0, "inferred columns must be left in place"


def test_none_returns_the_panel_untouched_and_the_input_is_never_mutated():
    panel = _panel()
    before = panel.copy()
    pd.testing.assert_frame_equal(realign_fx(panel, scope="none"), before)
    realign_fx(panel)
    pd.testing.assert_frame_equal(panel, before, obj="the input panel was mutated")


def test_an_unknown_scope_is_refused_rather_than_silently_ignored():
    with pytest.raises(ValueError, match="scope must be"):
        realign_fx(_panel(), scope="everything")


def test_the_report_accounts_for_every_fx_column_it_knows_about():
    panel = pd.DataFrame(
        {c: pd.Series([1.0, 2.0]) for c in FX_LATE + FX_ALIGNED}
    )
    report = realignment_report(panel)
    assert len(report) == len(FX_LATE) + len(FX_ALIGNED)
    assert (report.loc[list(FX_LATE), "action"] == "shifted").all()
    assert (report.loc[list(FX_ALIGNED), "action"] == "left alone").all()

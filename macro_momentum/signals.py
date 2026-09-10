"""The four macro themes, each measured as a change and never as a level.

That distinction is the study's premise. A companion project established, on the
same data and with the same point-in-time contract, that the *state* of the
economy carries information about forward variance and essentially none about
forward direction: two to four points of incremental R-squared on volatility,
0.02 to 0.03 on returns. Practitioners who trade macro directionally do not
classify states; they trade the rate of change. This module builds that object.

Every input arrives through the point-in-time panel, so a signal on a given date
uses only figures that had been published by then — first releases at their real
publication dates for the revised series, and a conservative lag for the market
rates that are never restated.
"""

from __future__ import annotations

import pandas as pd

#: One year of business days. Signals are changes over this span.
YEAR = 252


def _year_on_year(level: pd.Series) -> pd.Series:
    """Year-on-year growth of a level series.

    Levels are not comparable across vintages — an index gets rebased and the
    published number moves by ten percent without anything happening in the
    economy — so every macro input becomes a growth rate before it is used.
    """
    return level.pct_change(YEAR)


def growth(panel: pd.DataFrame) -> pd.Series:
    """Change in the pace of real activity over the past year.

    Industrial production and payrolls averaged: the two disagree often enough
    that either alone is noisy, and they are the two series with vintage history
    reaching back to 1990.
    """
    parts = [
        _year_on_year(panel[c]).diff(YEAR)
        for c in ("indpro", "payems")
        if c in panel
    ]
    if not parts:
        raise KeyError("no activity series in the panel")
    return pd.concat(parts, axis=1).mean(axis=1).rename("growth")


def inflation(panel: pd.DataFrame) -> pd.Series:
    """Change in the pace of consumer prices over the past year."""
    return _year_on_year(panel["cpi"]).diff(YEAR).rename("inflation")


def policy(panel: pd.DataFrame) -> pd.Series:
    """Change in the short policy rate over the past year.

    Rising means tightening. The three-month bill is used rather than the target
    rate because it is quoted daily, never revised, and moves ahead of the
    committee.
    """
    return panel["bill_3m"].diff(YEAR).rename("policy")


def risk(panel: pd.DataFrame) -> pd.Series:
    """Change in the credit spread over the past year.

    Rising means the market is charging more for default risk. Moody's Baa over
    the ten-year Treasury, because the option-adjusted spreads a desk would use
    are served by the data provider for the trailing two years only.
    """
    return panel["baa_spread"].diff(YEAR).rename("risk")


def all_themes(panel: pd.DataFrame) -> pd.DataFrame:
    """The four themes as one frame, in the order of the frozen sign map."""
    return pd.concat(
        [growth(panel), inflation(panel), policy(panel), risk(panel)], axis=1
    )


def standardise(themes: pd.DataFrame, *, min_periods: int = 756) -> pd.DataFrame:
    """Scale each theme by its own expanding history, then clip.

    Expanding rather than full-sample: a 2008 reading must not know how extreme
    2008 turned out to be. Clipping at three keeps a single macro shock from
    dominating the book, and is applied after scaling so it cannot change the
    scale itself.
    """
    mean = themes.expanding(min_periods=min_periods).mean()
    std = themes.expanding(min_periods=min_periods).std()
    return ((themes - mean) / std.replace(0.0, pd.NA)).clip(-3.0, 3.0)

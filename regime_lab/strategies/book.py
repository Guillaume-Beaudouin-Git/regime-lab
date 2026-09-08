"""One definition of the base book, loaded the same way everywhere.

Phase 0 measured power on returns in excess of cash; phase 2's first run did
not, and reported Sharpe ratios gross of the risk-free rate. The two numbers
were not comparable and one of them was wrong. Both now come through here.

Excess is the right convention twice over. It is the project's stated rule, and
it is what makes the on/off rule mean what it says: gating an excess series to
zero represents holding cash, which is what a flat book actually does. Gating a
gross series to zero represents holding nothing at all — not even a Treasury
bill — which no one does, and which quietly penalises the overlay.
"""

from __future__ import annotations

import pandas as pd

from regime_lab.data import build_panel, store
from regime_lab.data.universe import MACRO_LAGGED, MACRO_VINTAGED
from regime_lab.strategies.base import sixty_forty


def panels(start: str = "1990-01-01") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the price panel and the macro panel on a common business-day grid."""
    prices = store.read("prices", "cross_asset")
    dates = pd.date_range(start, prices["period"].max(), freq="B")
    price_panel = build_panel(prices, dates)

    macro = store.load_many([("macro", n) for n in list(MACRO_VINTAGED) + list(MACRO_LAGGED)])
    macro_panel = build_panel(macro, dates, max_staleness=pd.Timedelta(days=120))
    return price_panel, macro_panel


def cash_rate(macro_panel: pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series:
    """Daily risk-free rate from the three-month bill, forward filled."""
    return (macro_panel["rate_cash_3m"].reindex(index).ffill() / 100.0) / 252.0


def base_book(*, excess: bool = True) -> pd.Series:
    """The frozen 60/40 book, in excess of cash unless asked otherwise.

    ``excess=False`` exists only so a report can show what the convention is
    worth; nothing in the study evaluates on gross returns.
    """
    price_panel, macro_panel = panels()
    book = sixty_forty(price_panel).dropna()
    if not excess:
        return book.rename("sixty_forty_gross")
    return (book - cash_rate(macro_panel, book.index)).dropna().rename("sixty_forty")

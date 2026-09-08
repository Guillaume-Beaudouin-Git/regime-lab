"""Transaction costs, priced before the results were seen.

The charter requires the cost level to be written down in advance, because
otherwise it becomes a free parameter: on strategies of this kind the sign of
the answer moves between two and three basis points, and an author who picks the
level after seeing the table is choosing the conclusion.

The book is a 60/40 traded as two liquid futures legs. Round-trip costs there
are of the order of half a basis point for the equity leg and one for the note,
so a blended round trip of about two basis points is the realistic figure and
five is a conservative one. Both are reported, alongside ten as a stress.

Cost is charged on the change in exposure, which is what actually trades: a book
gated from one to zero and back pays two crossings.
"""

from __future__ import annotations

import pandas as pd

#: Round-trip cost in basis points. Declared before any result was read.
COST_BPS = {"realistic": 2.0, "conservative": 5.0, "stress": 10.0}


def charge(returns: pd.Series, position: pd.Series, *, bps: float) -> pd.Series:
    """Subtract the cost of trading ``position`` from ``returns``.

    ``position`` is the exposure actually held, already lagged. Its absolute
    change is the fraction of the book crossed on that session.
    """
    traded = position.reindex(returns.index).fillna(0.0).diff().abs().fillna(0.0)
    return (returns - traded * bps / 10_000.0).rename(returns.name)


def breakeven_bps(gross_edge: float, mean_turnover: float) -> float:
    """Cost level at which an annual edge in return terms disappears.

    Returns the round-trip basis points that exactly consume ``gross_edge``
    (annualised, in return units) given ``mean_turnover`` per session.
    """
    annual_turnover = mean_turnover * 252
    if annual_turnover <= 0:
        return float("inf")
    return gross_edge / annual_turnover * 10_000.0

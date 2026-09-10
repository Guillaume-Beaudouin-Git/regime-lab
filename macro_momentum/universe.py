"""What the study trades, and what drives it.

The macro side is deliberately United States only. A proper implementation
prices growth and inflation momentum in every country and trades the resulting
cross-section, but vintaged macro outside the US is not freely available, and
using today's revised foreign series would reintroduce exactly the leak this
project's infrastructure exists to prevent. The limitation is declared here
rather than discovered later.
"""

from __future__ import annotations

#: Assets traded, ``{ticker: asset class}``. Classes match the frozen sign map.
ASSETS = {
    "^GSPC": "equities", "^NDX": "equities", "^RUT": "equities",
    "^STOXX50E": "equities", "^N225": "equities", "^FTSE": "equities",
    "^GDAXI": "equities", "EEM": "equities",
    "TLT": "bonds", "IEF": "bonds", "SHY": "bonds", "AGG": "bonds", "TIP": "bonds",
    "GC=F": "commodities", "SI=F": "commodities", "CL=F": "commodities",
    "HG=F": "commodities", "ZC=F": "commodities", "ZS=F": "commodities",
    "DX-Y.NYB": "dollar",
}

#: Macro inputs taken as first releases, with their real publication dates.
VINTAGED = {
    "indpro": "INDPRO",
    "payems": "PAYEMS",
    "cpi": "CPIAUCSL",
}

#: Market-based inputs, never revised, so a publication lag is exact.
LAGGED = {
    "bill_3m": ("DTB3", "daily"),
    "baa_spread": ("BAA10Y", "daily"),
}

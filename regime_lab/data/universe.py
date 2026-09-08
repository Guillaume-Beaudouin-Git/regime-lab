"""The instruments and macro series the study draws on.

Every entry has to earn its place by feeding a feature the thesis needs. The
catalogue grows against that test, not to look comprehensive.
"""

from __future__ import annotations

#: Cross-asset daily prices, ``{series_id: yahoo_symbol}``.
PRICES = {
    "eq_us_large": "^GSPC",
    "eq_us_small": "^RUT",
    "eq_us_tech": "^IXIC",
    "eq_jp": "^N225",
    "bond_us_long": "^TYX",
    "bond_us_10y": "^TNX",
    "cmd_gold": "GC=F",
    "cmd_oil": "CL=F",
    "fx_dollar": "DX-Y.NYB",
    "vol_vix": "^VIX",
}

#: Series taken as first releases from ALFRED, ``{series_id: fred_id}``.
#:
#: These four carry the revision study. All have vintage history back to 1990,
#: which most FRED series do not: initial claims start in 2009 and the fourth
#: revision of the St. Louis stress index in 2022, so asking for their vintages
#: buys a short series rather than a real-time one.
MACRO_VINTAGED = {
    "macro_indpro": "INDPRO",
    "macro_payems": "PAYEMS",
    "macro_unrate": "UNRATE",
    "macro_cpi": "CPIAUCSL",
}

#: Series taken as current values plus a publication lag,
#: ``{series_id: (fred_id, frequency, revised)}``.
#:
#: ``revised=False`` means the lag path is exact, not an approximation: a market
#: rate printed on a given day is never restated.
MACRO_LAGGED = {
    "macro_claims": ("ICSA", "weekly", True),
    "fin_nfci": ("NFCI", "weekly", True),
    "fin_stlfsi": ("STLFSI4", "weekly", True),
    "fin_baa_spread": ("BAA10Y", "daily", False),
    "fin_aaa_spread": ("AAA10Y", "daily", False),
    "rate_curve_10y2y": ("T10Y2Y", "daily", False),
    "rate_curve_10y3m": ("T10Y3M", "daily", False),
    "rate_cash_3m": ("DTB3", "daily", False),
}
